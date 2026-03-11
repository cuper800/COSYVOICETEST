import sys
sys.path.append('third_party/Matcha-TTS')

import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
# ========== 設定 ==========
RECORD_SECONDS = 10          # 錄音秒數（建議 5~15 秒，內容越豐富越好）
SAMPLE_RATE = 16000           # 取樣率
MIC_DEVICE = 1               # 麥克風設備編號（Realtek Audio）
PROMPT_WAV = './my_voice.wav'           # 原始錄音
PROMPT_WAV_CLEAN = './my_voice_clean.wav'  # 前處理後的錄音
SPK_CACHE = './my_speaker.pt'           # 說話人特徵快取

# ⚠️ 重要：PROMPT_TEXT 必須和你錄音時「實際說的話」完全一致！
# CosyVoice3 格式：前面加 'You are a helpful assistant.<|endofprompt|>'
# 👇👇👇 請改成你錄音時「實際說的話」👇👇👇
PROMPT_TEXT = 'You are a helpful assistant.<|endofprompt|>大家好，我是一個測試語音，今天天氣真不錯，希望大家都有美好的一天。'

# 要合成的目標文字（廣東話測試）
TARGET_TEXTS = [
    '今日天氣真係好好，我好耐之前，流行嘅版本係屌那媽。',
    '你食咗飯未呀？要唔要一齊去食嘢？',
    '呢樣嘢幾多錢呀？可唔可以平啲呀？',
    '我好鍾意飲茶，香港嘅茶樓係全世界最好嘅。',
]

# 關閉文字前端正規化（避免被轉成大陸腔）
TEXT_FRONTEND = False

# instruct 風格指令（廣東話）
INSTRUCT_STYLES = [
    'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
]

# =========================


def trim_silence(audio, sr, threshold_db=-35, min_silence_ms=200):
    """去除開頭和結尾的靜音段"""
    threshold = 10 ** (threshold_db / 20)
    frame_len = int(sr * min_silence_ms / 1000)

    # 計算每個 frame 的能量
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, frame_len // 2)
    ])

    # 找到開頭和結尾的有聲位置
    active = np.where(energy > threshold)[0]
    if len(active) == 0:
        return audio  # 全靜音，不裁切

    start = max(0, active[0] * (frame_len // 2) - frame_len)  # 保留一小段前導
    end = min(len(audio), (active[-1] + 1) * (frame_len // 2) + frame_len)
    return audio[start:end]


def normalize_audio(audio, target_peak=0.9):
    """正規化音量到目標峰值"""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def preprocess_audio(input_path, output_path, sr=SAMPLE_RATE):
    """音訊前處理：去靜音 + 正規化音量"""
    audio, file_sr = sf.read(input_path)

    # 如果是多聲道，取平均
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 重取樣（如果需要）
    if file_sr != sr:
        import resampy
        audio = resampy.resample(audio, file_sr, sr)

    original_len = len(audio) / sr
    print(f"   原始長度: {original_len:.1f} 秒")

    # 去靜音
    audio = trim_silence(audio, sr)
    trimmed_len = len(audio) / sr
    print(f"   去靜音後: {trimmed_len:.1f} 秒")

    # 正規化音量
    audio = normalize_audio(audio)

    # 分析音質指標
    rms = np.sqrt(np.mean(audio ** 2))
    snr_estimate = 20 * np.log10(np.max(np.abs(audio)) / (rms + 1e-8))
    print(f"   RMS 能量: {rms:.4f}")
    print(f"   峰值 SNR: {snr_estimate:.1f} dB")

    if trimmed_len < 3:
        print("   ⚠️ 有效語音太短（<3秒），建議重新錄音！")
    elif trimmed_len > 15:
        print("   ⚠️ 語音偏長（>15秒），只取前 15 秒")
        audio = audio[:int(15 * sr)]

    sf.write(output_path, audio, sr)
    print(f"   ✅ 已儲存前處理後的音訊: {output_path}")
    return audio


def record_voice():
    """錄製你的聲音"""
    print(f"\n🎤 準備錄音！")
    print(f"   設備: {sd.query_devices(MIC_DEVICE)['name']}")
    print(f"   時長: {RECORD_SECONDS} 秒")
    print(f"   取樣率: {SAMPLE_RATE} Hz")
    print(f"\n💡 錄音建議：")
    print(f"   • 安靜環境，減少背景噪音")
    print(f"   • 嘴巴離麥克風 15~30 公分")
    print(f"   • 用自然語速說話，句子要完整")
    print(f"   • 內容越豐富（有逗號、句號停頓）效果越好")
    print()
    input("👉 按 Enter 開始錄音...")

    print(f"🔴 正在錄音（{RECORD_SECONDS} 秒）...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32', device=MIC_DEVICE)
    sd.wait()
    audio = audio.flatten()
    print("✅ 錄音完成！")

    # 儲存原始 WAV
    sf.write(PROMPT_WAV, audio, SAMPLE_RATE)
    print(f"💾 已儲存原始錄音: {PROMPT_WAV}")

    # 音訊前處理
    print("\n🔧 音訊前處理...")
    clean_audio = preprocess_audio(PROMPT_WAV, PROMPT_WAV_CLEAN)

    # 播放處理後的版本
    print("\n🔊 播放處理後的錄音...")
    sd.play(clean_audio, SAMPLE_RATE)
    sd.wait()

    ok = input("\n👉 錄音效果滿意嗎？(y=繼續合成 / n=重新錄音): ").strip().lower()
    return ok == 'y' or ok == ''


def synthesize():
    """用你的聲音做 zero-shot 語音合成（多模式）"""
    print("\n🤖 載入 CosyVoice3 模型...")
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

    prompt_wav = PROMPT_WAV_CLEAN if os.path.exists(PROMPT_WAV_CLEAN) else PROMPT_WAV

    # --- 儲存說話人特徵（下次可直接使用，不需重新錄音）---
    print("\n💾 儲存說話人特徵...")
    cosyvoice.add_zero_shot_spk(PROMPT_TEXT, prompt_wav, 'my_voice')
    cosyvoice.save_spkinfo()
    print("   ✅ 說話人特徵已快取（下次可跳過錄音直接合成）")

    # ==================== 模式 1: Zero-Shot ====================
    print("\n" + "=" * 50)
    print("  📢 模式 1: Zero-Shot 語音克隆")
    print("=" * 50)

    for idx, text in enumerate(TARGET_TEXTS):
        print(f"\n🎯 合成文字 [{idx+1}]: {text[:40]}...")
        for i, j in enumerate(cosyvoice.inference_zero_shot(
            text, PROMPT_TEXT, prompt_wav, stream=False, text_frontend=TEXT_FRONTEND
        )):
            output_file = f'output_zeroshot_{idx}_{i}.wav'
            torchaudio.save(output_file, j['tts_speech'], cosyvoice.sample_rate)
            duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
            print(f"   ✅ {output_file} ({duration:.1f} 秒)")

    # ============= 模式 2: Instruct2（風格控制）=============
    print("\n" + "=" * 50)
    print("  📢 模式 2: Instruct2 風格控制")
    print("=" * 50)

    for idx, (style, text) in enumerate(zip(INSTRUCT_STYLES, TARGET_TEXTS)):
        style_name = style.split('请用')[1].split('。')[0] if '请用' in style else f'風格{idx}'
        print(f"   (使用廣東話指令)")
        print(f"\n🎭 風格: {style_name}")
        print(f"🎯 文字: {text[:40]}...")
        for i, j in enumerate(cosyvoice.inference_instruct2(
            text, style, prompt_wav, stream=False, text_frontend=TEXT_FRONTEND
        )):
            output_file = f'output_instruct_{idx}_{i}.wav'
            torchaudio.save(output_file, j['tts_speech'], cosyvoice.sample_rate)
            duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
            print(f"   ✅ {output_file} ({duration:.1f} 秒)")

    # ============= 模式 3: 用快取特徵合成（示範）=============
    print("\n" + "=" * 50)
    print("  📢 模式 3: 使用快取特徵（無需音訊）")
    print("=" * 50)

    text = TARGET_TEXTS[0]
    print(f"\n🎯 文字: {text[:40]}...")
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        text, '', '', zero_shot_spk_id='my_voice', stream=False, text_frontend=TEXT_FRONTEND
    )):
        output_file = f'output_cached_{i}.wav'
        torchaudio.save(output_file, j['tts_speech'], cosyvoice.sample_rate)
        duration = j['tts_speech'].shape[1] / cosyvoice.sample_rate
        print(f"   ✅ {output_file} ({duration:.1f} 秒)")

    # 播放第一個結果
    print("\n🔊 播放 zero-shot 合成結果...")
    audio, sr = sf.read('output_zeroshot_0_0.wav')
    sd.play(audio, sr)
    sd.wait()

    print("\n🎉 全部完成！所有合成檔案：")
    for f in sorted(os.listdir('.')):
        if f.startswith('output_') and f.endswith('.wav'):
            size = os.path.getsize(f) / 1024
            print(f"   🎵 {f} ({size:.0f} KB)")


def main():
    print("=" * 50)
    print("  CosyVoice3 Zero-Shot 廣東話測試")
    print("=" * 50)

    # 檢查是否有之前的錄音
    if os.path.exists(PROMPT_WAV):
        print(f"\n📂 偵測到之前的錄音: {PROMPT_WAV}")
        choice = input("👉 要使用之前的錄音嗎？(y=使用 / n=重新錄音): ").strip().lower()
        if choice == 'y' or choice == '':
            if not os.path.exists(PROMPT_WAV_CLEAN):
                print("\n🔧 音訊前處理...")
                preprocess_audio(PROMPT_WAV, PROMPT_WAV_CLEAN)
            synthesize()
            return

    # 步驟 1: 錄音
    while True:
        if record_voice():
            break
        print("\n🔄 重新錄音...\n")

    # 步驟 2: 合成
    synthesize()


if __name__ == '__main__':
    main()
