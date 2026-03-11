"""
豬哥亮聲音分離 - 簡化版
先用 ffmpeg 切成小段，再用 campplus 比對，最後用 Whisper 轉寫
"""
import sys, os, shutil, subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
from pathlib import Path
from tqdm import tqdm

# ============ 設定 ============
REF_DIR = Path('ref_clips/zgl')
SOURCES = [
    Path('raw_audio/zgl/zgl_interview1_16k.wav'),
    Path('raw_audio/zgl/zgl_interview2_16k.wav'),
]
OUTPUT_DIR = Path('voice_data/zgl')
TEMP_DIR = Path('voice_data/zgl/_temp_segments')
CAMPPLUS_PATH = 'pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx'
THRESHOLD = 0.70
MIN_SEC = 3.0
MAX_SEC = 15.0
SEGMENT_SEC = 8.0   # 固定切段長度
OVERLAP_SEC = 2.0   # 重疊秒數
TARGET_SR = 24000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ============ 載入 campplus ============
print('[1/6] 載入 campplus...')
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

# ============ 提取參考嵌入 ============
print('[2/6] 提取參考嵌入...')
ref_embs = []
for rf in sorted(REF_DIR.glob('*.wav')):
    audio, sr = torchaudio.load(str(rf))
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    emb = extract_emb(audio.squeeze(0))
    ref_embs.append(emb)
    print(f'  {rf.name} -> OK')

ref_mean = np.mean(ref_embs, axis=0)
ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-8)
print(f'  參考嵌入數: {len(ref_embs)}')

# ============ 預先載入全部音檔 ============
print('[3/6] 預先載入音檔...')
source_audio = {}  # src_name -> 1D tensor (16kHz)
for src in SOURCES:
    if not src.exists():
        print(f'  [SKIP] {src} 不存在')
        continue
    audio_16k, sr = torchaudio.load(str(src))
    if audio_16k.shape[0] > 1:
        audio_16k = audio_16k.mean(dim=0, keepdim=True)
    source_audio[src.name] = audio_16k.squeeze(0)
    print(f'  {src.name}: {audio_16k.shape[-1]/16000:.1f}s loaded')

# ============ 切段 + 比對 ============
print('[4/6] 切段並比對...')
all_kept = []  # [(sim, src_name, start_s, end_s)]
total_kept = 0
total_rejected = 0

for src in SOURCES:
    if src.name not in source_audio:
        continue

    audio_16k = source_audio[src.name]

    total_samples = audio_16k.shape[0]
    total_dur = total_samples / 16000
    print(f'\n  {src.name}: {total_dur:.1f}s ({total_dur/60:.1f}min)')

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

        # 跳過太安靜的片段（靜音/音樂）
        if seg.abs().max().item() < 0.01:
            continue

        try:
            emb = extract_emb(seg)
            sim = cosine_sim(emb, ref_mean)
        except Exception:
            rejected += 1
            continue

        if sim >= THRESHOLD:
            all_kept.append((sim, src.name, start / 16000, end / 16000))
            kept += 1
        else:
            rejected += 1

    total_kept += kept
    total_rejected += rejected
    print(f'  保留 {kept} 段, 丟棄 {rejected} 段')

print(f'\n總計: 保留 {total_kept}, 丟棄 {total_rejected}')

if total_kept == 0:
    print('[ERROR] 沒有找到任何符合的片段！嘗試降低 THRESHOLD')
    sys.exit(1)

# ============ 合併重疊片段 + 儲存 ============
print(f'\n[5/6] 合併重疊片段並儲存...')

# 按來源和時間排序
all_kept.sort(key=lambda x: (x[1], x[2]))

# 合併重疊的片段（同一來源檔案中）
merged = []
for sim, src_name, start_s, end_s in all_kept:
    if merged and merged[-1][1] == src_name and start_s < merged[-1][3] + 0.5:
        # 合併：擴展結束時間，保留較高的 sim
        prev = merged[-1]
        new_end = max(prev[3], end_s)
        new_dur = new_end - prev[2]
        if new_dur <= MAX_SEC:
            merged[-1] = (max(prev[0], sim), src_name, prev[2], new_end)
            continue
    merged.append((sim, src_name, start_s, end_s))

# 再按相似度排序（高到低）
merged.sort(key=lambda x: -x[0])

print(f'  合併後: {len(merged)} 段')

# 儲存 WAV (resample 到 24kHz)
resampler = torchaudio.transforms.Resample(16000, TARGET_SR)
total_duration = 0.0
manifest = []

for i, (sim, src_name, start_s, end_s) in enumerate(merged):
    clip_name = f'clip_{i:04d}'
    wav_path = OUTPUT_DIR / f'{clip_name}.wav'
    
    # 從預載音檔中切出片段
    full_audio = source_audio[src_name]
    start_sample = int(start_s * 16000)
    end_sample = int(end_s * 16000)
    seg_audio = full_audio[start_sample:end_sample]
    dur = seg_audio.shape[0] / 16000
    total_duration += dur

    # resample 到 24kHz
    seg_24k = resampler(seg_audio.unsqueeze(0))
    torchaudio.save(str(wav_path), seg_24k, TARGET_SR)

    manifest.append({
        'clip': clip_name,
        'src': src_name,
        'start': f'{start_s:.2f}',
        'end': f'{end_s:.2f}',
        'dur': f'{dur:.2f}',
        'sim': f'{sim:.4f}',
    })

# 寫入清單
with open(OUTPUT_DIR / '_manifest.txt', 'w', encoding='utf-8') as f:
    f.write(f'speaker: zgl\n')
    f.write(f'threshold: {THRESHOLD}\n')
    f.write(f'clips: {len(manifest)}\n')
    f.write(f'total: {total_duration:.1f}s ({total_duration/60:.1f}min)\n')
    f.write('=' * 70 + '\n')
    for m in manifest:
        f.write(f"{m['clip']}  {m['src']}  {m['start']:>8}  {m['end']:>8}  {m['dur']:>6}  {m['sim']:>8}\n")

print(f'  已儲存 {len(manifest)} 段到 {OUTPUT_DIR}')
print(f'  總時長: {total_duration:.1f}s ({total_duration/60:.1f}min)')

# ============ Whisper 轉寫 ============
print(f'\n[6/6] Whisper 轉寫...')
import whisper
wmodel = whisper.load_model('base')
success = 0
for m in tqdm(manifest, desc='  轉寫'):
    wav_path = OUTPUT_DIR / f"{m['clip']}.wav"
    txt_path = OUTPUT_DIR / f"{m['clip']}.txt"
    try:
        result = wmodel.transcribe(str(wav_path), language='zh',
                                   initial_prompt='以下是正體中文語音轉寫。')
        text = result['text'].strip()
        if text:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            success += 1
    except Exception as e:
        print(f'  {m["clip"]} 失敗: {e}')

print(f'  轉寫完成: {success}/{len(manifest)}')

# 清理
shutil.rmtree(str(TEMP_DIR), ignore_errors=True)

print('\n' + '=' * 60)
print(f'  完成！')
print(f'  輸出: {OUTPUT_DIR}')
print(f'  片段: {len(manifest)}')
print(f'  時長: {total_duration:.1f}s ({total_duration/60:.1f}min)')
print(f'  轉寫: {success}/{len(manifest)}')
print(f'\n  下一步：')
print(f'    python finetune_voice.py --voice_dir voice_data/zgl --speaker zgl')
print('=' * 60)
