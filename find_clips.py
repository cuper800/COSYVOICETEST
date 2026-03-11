import torchaudio
from pathlib import Path

clips = sorted(Path('voice_data/韓國瑜').glob('clip_*.wav'))
good = []
for c in clips[:300]:
    txt = c.with_suffix('.txt')
    if not txt.exists():
        continue
    wav, sr = torchaudio.load(str(c))
    dur = wav.shape[1] / sr
    amp = wav.abs().max().item()
    text = txt.read_text(encoding='utf-8').strip()
    if 3 <= dur <= 10 and amp > 0.1 and len(text) > 5:
        good.append((c.name, dur, amp, text))

print(f"Found {len(good)} good clips")
for name, dur, amp, text in good[:15]:
    print(f"{name} ({dur:.1f}s, amp={amp:.2f}) => {text[:80]}")
