"""從豬哥亮 14s 片段裁切出 3s/5s/7s 短版本作為 prompt"""
import torchaudio
from pathlib import Path

voice_dir = Path('voice_data/zgl')

# 白名單片段
WHITELIST = [
    'clip_0002', 'clip_0003', 'clip_0004', 'clip_0017', 'clip_0019',
    'clip_0030', 'clip_0033', 'clip_0039', 'clip_0045', 'clip_0049',
    'clip_0052', 'clip_0059', 'clip_0067', 'clip_0075', 'clip_0078',
    'clip_0086', 'clip_0096', 'clip_0114', 'clip_0136', 'clip_0151',
    'clip_0155', 'clip_0183', 'clip_0198', 'clip_0200', 'clip_0268',
]

DURATIONS = [5, 7, 10]  # 秒

created = 0
for name in WHITELIST:
    wav_path = voice_dir / f'{name}.wav'
    txt_path = voice_dir / f'{name}.txt'
    if not wav_path.exists():
        continue
    wav, sr = torchaudio.load(str(wav_path))
    full_dur = wav.shape[1] / sr
    text = txt_path.read_text(encoding='utf-8').strip() if txt_path.exists() else ''

    for dur in DURATIONS:
        if dur >= full_dur - 0.5:
            continue  # 跳過太接近原長的
        samples = int(dur * sr)
        trimmed = wav[:, :samples]
        # 估算對應文字長度（按比例裁切）
        char_count = int(len(text) * dur / full_dur)
        trimmed_text = text[:max(char_count, 4)]

        out_name = f'{name}_{dur}s'
        out_wav = voice_dir / f'{out_name}.wav'
        out_txt = voice_dir / f'{out_name}.txt'
        torchaudio.save(str(out_wav), trimmed, sr)
        out_txt.write_text(trimmed_text, encoding='utf-8')
        created += 1
        print(f'  {out_name}.wav ({dur}s) -> {trimmed_text[:30]}...')

print(f'\n完成！共產生 {created} 個短片段')
