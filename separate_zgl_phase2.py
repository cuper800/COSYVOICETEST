"""
豬哥亮聲音分離 - Phase 2: 合併 + 儲存 WAV + Whisper 轉寫
讀取 Phase 1 產生的 _segments.json
"""
import sys, os, signal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import warnings
warnings.filterwarnings('ignore')

signal.signal(signal.SIGINT, signal.SIG_IGN)

import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

SOURCES = {
    'zgl_interview1_16k.wav': Path('raw_audio/zgl/zgl_interview1_16k.wav'),
    'zgl_interview2_16k.wav': Path('raw_audio/zgl/zgl_interview2_16k.wav'),
}
OUTPUT_DIR = Path('voice_data/zgl')
TARGET_SR = 24000
MAX_SEC = 15.0

segments_path = OUTPUT_DIR / '_segments.json'
if not segments_path.exists():
    print('找不到 _segments.json，請先跑 phase1')
    sys.exit(1)

with open(segments_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
segments = data['segments']
print(f'讀取 {len(segments)} 個片段')

# 合併重疊片段
segments.sort(key=lambda x: (x['src'], x['start']))
merged = []
for seg in segments:
    if merged and merged[-1]['src'] == seg['src'] and seg['start'] < merged[-1]['end'] + 0.5:
        prev = merged[-1]
        new_end = max(prev['end'], seg['end'])
        new_dur = new_end - prev['start']
        if new_dur <= MAX_SEC:
            merged[-1] = {
                'src': seg['src'],
                'start': prev['start'],
                'end': new_end,
                'sim': max(prev['sim'], seg['sim']),
            }
            continue
    merged.append(dict(seg))

# 按相似度排序
merged.sort(key=lambda x: -x['sim'])
print(f'合併後: {len(merged)} 段')

# 預載音檔
print('載入音檔...')
source_audio = {}
for name, path in SOURCES.items():
    if path.exists():
        audio, sr = torchaudio.load(str(path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        source_audio[name] = audio.squeeze(0)
        print(f'  {name}: {audio.shape[-1]/16000:.1f}s')

# 儲存 WAV
print('儲存 WAV...')
resampler = torchaudio.transforms.Resample(16000, TARGET_SR)
total_duration = 0.0
manifest = []

# 清理舊 clip 檔案
for old in OUTPUT_DIR.glob('clip_*.wav'):
    old.unlink()
for old in OUTPUT_DIR.glob('clip_*.txt'):
    old.unlink()

for i, seg in enumerate(merged):
    clip_name = f'clip_{i:04d}'
    wav_path = OUTPUT_DIR / f'{clip_name}.wav'

    audio = source_audio[seg['src']]
    start_sample = int(seg['start'] * 16000)
    end_sample = int(seg['end'] * 16000)
    clip_audio = audio[start_sample:end_sample]
    dur = clip_audio.shape[0] / 16000
    total_duration += dur

    seg_24k = resampler(clip_audio.unsqueeze(0))
    torchaudio.save(str(wav_path), seg_24k, TARGET_SR)

    manifest.append({
        'clip': clip_name,
        'src': seg['src'],
        'start': f"{seg['start']:.2f}",
        'end': f"{seg['end']:.2f}",
        'dur': f'{dur:.2f}',
        'sim': f"{seg['sim']:.4f}",
    })

# 寫清單
with open(OUTPUT_DIR / '_manifest.txt', 'w', encoding='utf-8') as f:
    f.write(f'speaker: zgl\n')
    f.write(f'clips: {len(manifest)}\n')
    f.write(f'total: {total_duration:.1f}s ({total_duration/60:.1f}min)\n')
    f.write('=' * 70 + '\n')
    for m in manifest:
        f.write(f"{m['clip']}  {m['src']}  {m['start']:>8}  {m['end']:>8}  {m['dur']:>6}  {m['sim']:>8}\n")

print(f'已儲存 {len(manifest)} 個 WAV')
print(f'總時長: {total_duration:.1f}s ({total_duration/60:.1f}min)')

# 釋放音檔記憶體
del source_audio
import gc; gc.collect()

# Whisper 轉寫
print('\nWhisper 轉寫...')
import whisper
wmodel = whisper.load_model('base')
success = 0
for m in tqdm(manifest, desc='轉寫'):
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

print(f'\n轉寫完成: {success}/{len(manifest)}')
print(f'\n{"="*60}')
print(f'  完成！')
print(f'  輸出: {OUTPUT_DIR}')
print(f'  片段: {len(manifest)}')
print(f'  時長: {total_duration:.1f}s ({total_duration/60:.1f}min)')
print(f'  轉寫: {success}/{len(manifest)}')
print(f'{"="*60}')
