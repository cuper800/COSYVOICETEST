"""
準備訓練用音檔的輔助工具

功能：
  1. 從 YouTube 下載音訊（需安裝 yt-dlp）
  2. 自動切割為 5~15 秒片段
  3. 自動用 Whisper 轉寫
  4. 自動去除靜音段

用法：
  # 方法一：從 YouTube 下載
  python prepare_voice_data.py --youtube "https://www.youtube.com/watch?v=XXXXX" --speaker 韓國瑜

  # 方法二：從本地長音檔
  python prepare_voice_data.py --audio_file speech.mp3 --speaker 韓國瑜

  # 方法三：從一個資料夾的多個音檔
  python prepare_voice_data.py --audio_dir raw_audio/ --speaker 韓國瑜

輸出：
  voice_data/<speaker>/
    clip_001.wav + clip_001.txt
    clip_002.wav + clip_002.txt
    ...
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append('third_party/Matcha-TTS')

parser = argparse.ArgumentParser(description='準備聲音訓練資料')
parser.add_argument('--youtube', type=str, default=None,
                    help='YouTube 影片網址')
parser.add_argument('--audio_file', type=str, default=None,
                    help='本地音檔路徑')
parser.add_argument('--audio_dir', type=str, default=None,
                    help='本地音檔資料夾')
parser.add_argument('--speaker', type=str, required=True,
                    help='說話人名稱')
parser.add_argument('--output_dir', type=str, default=None,
                    help='輸出目錄（預設 voice_data/<speaker>）')
parser.add_argument('--max_clip_sec', type=float, default=15.0)
parser.add_argument('--min_clip_sec', type=float, default=3.0)
parser.add_argument('--silence_thresh_db', type=float, default=-35.0,
                    help='靜音門檻 (dB)')
parser.add_argument('--skip_transcribe', action='store_true',
                    help='跳過自動轉寫')
args = parser.parse_args()

import torch
import torchaudio

output_dir = Path(args.output_dir) if args.output_dir else Path(f'voice_data/{args.speaker}')
output_dir.mkdir(parents=True, exist_ok=True)


def download_youtube(url):
    """用 yt-dlp 下載 YouTube 音訊"""
    print(f'📥 從 YouTube 下載：{url}')
    output_path = output_dir / 'raw_youtube.wav'
    cmd = f'yt-dlp -x --audio-format wav -o "{output_path}" "{url}"'
    print(f'  指令：{cmd}')
    ret = os.system(cmd)
    if ret != 0:
        print('❌ 下載失敗！請確認已安裝 yt-dlp：')
        print('   pip install yt-dlp')
        sys.exit(1)
    return output_path


def split_on_silence(audio_path, max_sec=15.0, min_sec=3.0):
    """基於能量的簡單切割"""
    print(f'🔪 切割音檔：{audio_path}')

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 轉為 16kHz mono
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    audio = waveform[0].numpy()
    total_duration = len(audio) / sr
    print(f'  總長度：{total_duration / 60:.1f} 分鐘')

    # 計算短時能量
    frame_size = int(0.02 * sr)  # 20ms
    hop = frame_size // 2
    n_frames = (len(audio) - frame_size) // hop + 1
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_size]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

    # 轉 dB
    energy_db = 20 * np.log10(energy + 1e-10)
    silence_mask = energy_db < args.silence_thresh_db

    # 找靜音段的邊界作為切割點
    clips = []
    clip_start = 0
    max_samples = int(max_sec * sr)
    min_samples = int(min_sec * sr)

    i = 0
    while i < n_frames:
        current_pos = i * hop
        elapsed = current_pos - clip_start

        # 如果到了最大長度，或者遇到靜音段，就切
        if elapsed >= max_samples:
            # 往回找最近的靜音點
            best_cut = current_pos
            for j in range(i, max(0, i - int(2 * sr / hop)), -1):
                if silence_mask[j]:
                    best_cut = j * hop
                    break
            if best_cut - clip_start >= min_samples:
                clips.append((clip_start, best_cut))
            clip_start = best_cut
        elif elapsed >= min_samples and silence_mask[i]:
            # 找到靜音段末尾
            end_silence = i
            while end_silence < n_frames and silence_mask[end_silence]:
                end_silence += 1
            cut_point = ((i + end_silence) // 2) * hop
            if cut_point - clip_start >= min_samples:
                clips.append((clip_start, cut_point))
            clip_start = cut_point
            i = end_silence
            continue
        i += 1

    # 最後一段
    if len(audio) - clip_start >= min_samples:
        clips.append((clip_start, len(audio)))

    print(f'  切割為 {len(clips)} 個片段')

    # 儲存
    saved = []
    for idx, (start, end) in enumerate(clips):
        clip = waveform[:, start:end]
        clip_path = output_dir / f'clip_{idx + 1:04d}.wav'
        torchaudio.save(str(clip_path), clip, sr)
        duration = (end - start) / sr
        saved.append((clip_path, duration))
        print(f'  ✅ {clip_path.name} ({duration:.1f}s)')

    return saved


def transcribe_all():
    """用 Whisper 轉寫所有音檔"""
    import whisper
    print('\n🗣️ 載入 Whisper 進行轉寫...')
    model = whisper.load_model('base')
    print('✅ Whisper 載入完成\n')

    wav_files = sorted(output_dir.glob('clip_*.wav'))
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix('.txt')
        if txt_path.exists():
            continue

        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            audio_np = waveform[0].numpy().astype(np.float32)
            result = model.transcribe(audio_np, language='zh')
            text = result['text'].strip()
            if text:
                txt_path.write_text(text, encoding='utf-8')
                print(f'  ✅ {wav_path.name} → {text[:60]}')
            else:
                print(f'  ⚠️ {wav_path.name} → （空）')
        except Exception as e:
            print(f'  ⚠️ {wav_path.name} 失敗：{e}')


def main():
    print('🎤 聲音訓練資料準備工具')
    print('=' * 60)

    # Step 1: 取得原始音檔
    if args.youtube:
        raw_audio = download_youtube(args.youtube)
        clips = split_on_silence(raw_audio)
    elif args.audio_file:
        clips = split_on_silence(Path(args.audio_file))
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        all_clips = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']:
            for f in sorted(audio_dir.glob(ext)):
                all_clips.extend(split_on_silence(f))
        clips = all_clips
    else:
        print('❌ 請指定 --youtube、--audio_file 或 --audio_dir')
        sys.exit(1)

    # Step 2: 轉寫
    if not args.skip_transcribe:
        transcribe_all()

    # 統計
    total = len(list(output_dir.glob('clip_*.wav')))
    with_text = len(list(output_dir.glob('clip_*.txt')))
    print()
    print('=' * 60)
    print(f'✅ 完成！{output_dir} 內共 {total} 個音檔，{with_text} 個有文字')
    print()
    print('📌 下一步：執行微調訓練')
    print(f'   python finetune_voice.py --voice_dir {output_dir} --speaker {args.speaker}')
    print('=' * 60)


if __name__ == '__main__':
    main()
