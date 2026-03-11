"""
快速錄音工具 - 在本機錄製一段 prompt 音訊供 WebUI 上傳使用

用法：python record_prompt.py
"""
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

SAMPLE_RATE = 16000
DURATION = 8  # 錄 8 秒
DEVICE = 1     # Realtek Audio（你之前用的裝置）
OUTPUT_FILE = "my_prompt.wav"

print("=" * 50)
print(f"🎤 準備錄音（{DURATION} 秒）")
print(f"   裝置：{DEVICE}  取樣率：{SAMPLE_RATE}")
print(f"   輸出：{OUTPUT_FILE}")
print("=" * 50)

# 列出裝置
devices = sd.query_devices()
print(f"\n使用的裝置：{devices[DEVICE]['name']}")
print(f"\n按 Enter 開始錄音...")
input()

print(f"🔴 正在錄音... （請說一段話，例如：「今天天氣真好，我們一起去公園走走吧」）")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=1, dtype='float32', device=DEVICE)
sd.wait()
print("✅ 錄音完成！")

# 檢查
audio = audio.flatten()
max_amp = np.abs(audio).max()
rms = np.sqrt(np.mean(audio ** 2))
print(f"\n📊 音訊診斷：")
print(f"   長度：{len(audio) / SAMPLE_RATE:.2f}s")
print(f"   最大振幅：{max_amp:.6f}")
print(f"   RMS：{rms:.6f}")

if max_amp < 0.001:
    print("⚠️  警告：錄到的是靜音！請檢查麥克風裝置。")
    print("   嘗試列出所有輸入裝置：")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"   [{i}] {d['name']} (channels={d['max_input_channels']})")
else:
    # 正規化
    audio = audio / max_amp * 0.9
    # 存檔
    wav.write(OUTPUT_FILE, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    full_path = os.path.abspath(OUTPUT_FILE)
    print(f"\n✅ 已存檔：{full_path}")
    print(f"   你可以在 WebUI 用「上傳」按鈕上傳這個檔案作為 prompt 音訊")
