"""
豬哥亮聲音分離 - Phase 1: 切段比對
逐檔處理，處理完一個存一個，防止中斷遺失
"""
import sys, os, signal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import warnings
warnings.filterwarnings('ignore')

# 忽略 SIGINT（防止被終端中斷）
signal.signal(signal.SIGINT, signal.SIG_IGN)

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import json
from pathlib import Path

REF_DIR = Path('ref_clips/zgl')
SOURCES = [
    Path('raw_audio/zgl/zgl_interview1_16k.wav'),
    Path('raw_audio/zgl/zgl_interview2_16k.wav'),
]
OUTPUT_DIR = Path('voice_data/zgl')
CAMPPLUS_PATH = 'pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx'
THRESHOLD = 0.70
MIN_SEC = 3.0
MAX_SEC = 15.0
SEGMENT_SEC = 8.0
OVERLAP_SEC = 2.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# campplus
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 4
ort_session = onnxruntime.InferenceSession(
    CAMPPLUS_PATH, sess_options=option, providers=["CPUExecutionProvider"]
)

def extract_emb(audio_16k):
    if audio_16k.dim() == 1:
        audio_16k = audio_16k.unsqueeze(0)
    feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()})[0].flatten()
    return emb / (np.linalg.norm(emb) + 1e-8)

def cosine_sim(a, b):
    return float(np.dot(a, b))

# 參考嵌入
print('提取參考嵌入...')
ref_embs = []
for rf in sorted(REF_DIR.glob('*.wav')):
    audio, sr = torchaudio.load(str(rf))
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    ref_embs.append(extract_emb(audio.squeeze(0)))
    print(f'  {rf.name} OK')
ref_mean = np.mean(ref_embs, axis=0)
ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-8)

# 逐檔處理
results_path = OUTPUT_DIR / '_segments.json'

# 載入已有結果（斷點續接）
if results_path.exists():
    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
    done_files = set(all_results.get('done_files', []))
    segments = all_results.get('segments', [])
    print(f'已有結果: {len(segments)} 段, 已完成: {done_files}')
else:
    done_files = set()
    segments = []

for src in SOURCES:
    if src.name in done_files:
        print(f'\n跳過已完成: {src.name}')
        continue
    if not src.exists():
        print(f'\n跳過不存在: {src.name}')
        continue

    print(f'\n處理: {src.name}')
    audio_16k, sr = torchaudio.load(str(src))
    if audio_16k.shape[0] > 1:
        audio_16k = audio_16k.mean(dim=0, keepdim=True)
    audio_16k = audio_16k.squeeze(0)

    total_samples = audio_16k.shape[0]
    total_dur = total_samples / 16000
    print(f'  長度: {total_dur:.1f}s ({total_dur/60:.1f}min)')

    seg_samples = int(SEGMENT_SEC * 16000)
    step_samples = int((SEGMENT_SEC - OVERLAP_SEC) * 16000)
    n_segs = max(1, int(np.ceil((total_samples - seg_samples) / step_samples)) + 1)

    kept = 0
    rejected = 0

    for i in range(n_segs):
        start = i * step_samples
        end = min(start + seg_samples, total_samples)
        if (end - start) / 16000 < MIN_SEC:
            continue

        seg = audio_16k[start:end]
        if seg.abs().max().item() < 0.01:
            continue

        try:
            emb = extract_emb(seg)
            sim = cosine_sim(emb, ref_mean)
        except Exception:
            rejected += 1
            continue

        if sim >= THRESHOLD:
            segments.append({
                'src': src.name,
                'start': round(start / 16000, 2),
                'end': round(end / 16000, 2),
                'sim': round(sim, 4),
            })
            kept += 1
        else:
            rejected += 1

        # 每 100 段存一次
        if (kept + rejected) % 100 == 0:
            print(f'  ... {kept + rejected}/{n_segs}  kept={kept}')

    done_files.add(src.name)
    # 每完成一個檔案就存結果
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({'done_files': list(done_files), 'segments': segments}, f, ensure_ascii=False)
    print(f'  完成: 保留 {kept}, 丟棄 {rejected} → 已存檔')

    # 釋放記憶體
    del audio_16k
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

print(f'\n全部完成: {len(segments)} 段')
print(f'結果存在: {results_path}')
print('下一步: python separate_zgl_phase2.py')
