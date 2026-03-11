"""只做 Whisper 轉寫（跳過已有 .txt 的）"""
import sys, os, signal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
signal.signal(signal.SIGINT, signal.SIG_IGN)

from pathlib import Path

OUTPUT_DIR = Path('voice_data/zgl')
wavs = sorted(OUTPUT_DIR.glob('clip_*.wav'))
already = set(p.stem for p in OUTPUT_DIR.glob('clip_*.txt'))
todo = [w for w in wavs if w.stem not in already]
print(f'WAV: {len(wavs)}, 已轉寫: {len(already)}, 待轉寫: {len(todo)}')

if not todo:
    print('全部完成！')
    sys.exit(0)

import whisper
print('載入 Whisper base...')
model = whisper.load_model('base')
success = 0

for i, wav_path in enumerate(todo):
    txt_path = wav_path.with_suffix('.txt')
    try:
        result = model.transcribe(str(wav_path), language='zh',
                                  initial_prompt='以下是正體中文語音轉寫。')
        text = result['text'].strip()
        if text:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            success += 1
    except Exception as e:
        print(f'  {wav_path.stem} 失敗: {e}')

    if (i + 1) % 50 == 0:
        print(f'  進度: {i+1}/{len(todo)}  ok={success}')

print(f'\n轉寫完成: {success}/{len(todo)}')
total = len(already) + success
print(f'總計: {total}/{len(wavs)} 個 txt')
