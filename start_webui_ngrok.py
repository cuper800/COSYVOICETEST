"""
CosyVoice3 多功能 WebUI + ngrok

功能：
  1. 名人 TTS - 韓國瑜 / 豬哥亮 聲音合成
  2. 語音克隆 - 上傳你的聲音進行克隆
  3. 進階模式 - 原有 4 種推理模式

使用：
  python start_webui_ngrok.py
  python start_webui_ngrok.py --port 7860
"""
import sys
import os
import signal

# Windows SIGINT 保護（防止被終端中斷）
signal.signal(signal.SIGINT, signal.SIG_IGN)

# Windows cp950 無法輸出簡體字，強制 UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import numpy as np
import random
from pathlib import Path

sys.path.append('third_party/Matcha-TTS')

from pyngrok import ngrok, conf

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=7860)
parser.add_argument('--model-dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
args = parser.parse_args()

import torch
import torchaudio
import gradio as gr
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

import whisper

# ==================== 載入模型 ====================
print(f'[*] 載入模型：{args.model_dir}')
cosyvoice = AutoModel(model_dir=args.model_dir)

print('[*] 載入 Whisper 語音辨識模型...')
whisper_model = whisper.load_model('base')
print('[OK] Whisper 載入完成')

sft_spk = cosyvoice.list_available_spks()
if len(sft_spk) == 0:
    sft_spk = ['']


# ==================== 說話人設定 ====================
SPEAKER_CONFIG = {
    '韓國瑜': {
        'voice_dir': Path('voice_data/韓國瑜'),
        'llm_path': Path('pretrained_models/Fun-CosyVoice3-0.5B/llm_hanyu.pt'),
        'blacklist': {'clip_0042.wav', 'clip_0171.wav', 'clip_0017.wav'},
    },
    '豬哥亮': {
        'voice_dir': Path('voice_data/zgl'),
        'llm_path': Path('pretrained_models/Fun-CosyVoice3-0.5B/llm_zgl.pt'),
        'blacklist': set(),
        'max_prompts': 20,
        # 白名單基礎片段（14s 原版 + 自動包含 _5s/_7s/_10s 短版）
        'whitelist_base': [
            'clip_0155',  # 大家好,我是朱哥亮
            'clip_0075',  # 這是我老婆的功勞
            'clip_0067',  # 現在我還會想念它
            'clip_0198',  # 別人講過的話,你不可以講
            'clip_0183',  # 我人生的路程走過的路
            'clip_0019',  # 他這個七年當中,讓我體會以前的生活
            'clip_0039',  # 我們在策劃這一部電影
            'clip_0086',  # 我當然要看你的面子去演
            'clip_0004',  # 我們在交朋友不一樣
            'clip_0078',  # 就停在那邊應該不會再回舞台了
            'clip_0114',  # 別人會再看,再看,再看
            'clip_0200',  # 我才知道,吸引這個吸那麼重要
            'clip_0268',  # 主要是什麼角色,你的音,我都知道
        ],
    },
}
SPEAKER_NAMES = list(SPEAKER_CONFIG.keys())
SPEAKER_PROMPTS = {}   # speaker_name -> {display: (wav_path, prompt_text)}
SPEAKER_PROMPT_NAMES = {}  # speaker_name -> [display1, display2, ...]
CURRENT_SPEAKER = [None]  # 目前載入的說話人（用 list 方便 closure 修改）


def scan_prompts_for_speaker(speaker_name):
    """掃描某個說話人的訓練音檔，篩選品質好的作為內建 prompt"""
    cfg = SPEAKER_CONFIG[speaker_name]
    voice_dir = cfg['voice_dir']
    blacklist = cfg.get('blacklist', set())
    # 白名單：支援 whitelist（直接指定）或 whitelist_base（自動展開 _3s/_5s/_7s）
    whitelist = cfg.get('whitelist', None)
    whitelist_base = cfg.get('whitelist_base', None)
    if whitelist_base:
        whitelist = set()
        for base in whitelist_base:
            for suffix in ['_5s', '_7s', '_10s', '']:
                whitelist.add(f'{base}{suffix}.wav')
    prompts = {}
    if not voice_dir.exists():
        print(f'[WARN] 找不到 {voice_dir} 目錄')
        SPEAKER_PROMPTS[speaker_name] = prompts
        SPEAKER_PROMPT_NAMES[speaker_name] = ['(找不到內建音檔)']
        return
    clips = sorted(voice_dir.glob('clip_*.wav'))
    candidates = []
    for c in clips[:500]:
        if whitelist and c.name not in whitelist:
            continue
        if c.name in blacklist:
            continue
        txt = c.with_suffix('.txt')
        if not txt.exists():
            continue
        try:
            wav, sr = torchaudio.load(str(c))
            dur = wav.shape[1] / sr
            amp = wav.abs().max().item()
            text = txt.read_text(encoding='utf-8').strip()
            min_dur = 1.0 if whitelist else 3.0
            max_dur = 30.0 if whitelist else 8.0
            min_amp = 0.05 if whitelist else 0.15
            min_len = 4 if whitelist else 6
            if min_dur <= dur <= max_dur and amp > min_amp and len(text) >= min_len:
                candidates.append((c, dur, amp, text))
        except Exception:
            continue
    max_prompts = cfg.get('max_prompts', 8)

    # 若有 whitelist_base，排除 14s 原版，保留 5s/7s/10s 全部版本，按片段分組排列
    if whitelist_base:
        import re
        from collections import defaultdict
        base_groups = defaultdict(list)  # base_name -> [(dur_tag, candidate), ...]
        keep_durs = {5, 7, 10}
        for cand in candidates:
            fname = cand[0].stem
            m = re.match(r'(clip_\d+?)(?:_(\d+)s)?$', fname)
            if not m:
                continue
            base = m.group(1)
            dur_tag = int(m.group(2)) if m.group(2) else 14
            if dur_tag in keep_durs:
                base_groups[base].append((dur_tag, cand))
        # 每組內按時長排序，組間按振幅排序
        sorted_bases = sorted(base_groups.items(),
                              key=lambda kv: max(c[2] for _, c in kv[1]), reverse=True)
        candidates = []
        for base, items in sorted_bases:
            for dur_tag, cand in sorted(items, key=lambda x: x[0]):
                candidates.append(cand)

    candidates = candidates[:max_prompts * 3] if whitelist_base else candidates
    if not whitelist_base:
        candidates.sort(key=lambda x: (-x[2], -x[1]))
    for i, (c, dur, amp, text) in enumerate(candidates[:max_prompts * 3 if whitelist_base else max_prompts]):
        short_text = text[:30] + '...' if len(text) > 30 else text
        display = f'片段 {i+1} ({dur:.1f}s): {short_text}'
        prompt_text = f'You are a helpful assistant.<|endofprompt|>{text}'
        prompts[display] = (str(c), prompt_text)
    SPEAKER_PROMPTS[speaker_name] = prompts
    SPEAKER_PROMPT_NAMES[speaker_name] = list(prompts.keys()) if prompts else ['(找不到內建音檔)']
    print(f'[OK] {speaker_name}: 找到 {len(prompts)} 個 prompt 音檔')


for spk in SPEAKER_NAMES:
    scan_prompts_for_speaker(spk)


def switch_speaker(speaker_name):
    """切換說話人：重載 LLM 權重"""
    if CURRENT_SPEAKER[0] == speaker_name:
        return True
    cfg = SPEAKER_CONFIG[speaker_name]
    llm_src = cfg['llm_path']
    if not llm_src.exists():
        print(f'[ERROR] 找不到 {llm_src}')
        return False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] 切換說話人: {CURRENT_SPEAKER[0]} → {speaker_name}')
    print(f'    載入 {llm_src.name} ...')
    state_dict = torch.load(str(llm_src), map_location=device, weights_only=True)
    cosyvoice.model.llm.load_state_dict(state_dict, strict=True)
    cosyvoice.model.llm.to(device).eval()
    CURRENT_SPEAKER[0] = speaker_name
    print(f'[OK] 切換完成: {speaker_name}')
    return True


# 啟動時載入韓國瑜
switch_speaker('韓國瑜')


# ==================== 進階模式設定 ====================
MODE_LIST = ['Zero-Shot 聲音克隆', '跨語言克隆', '風格/方言控制', '預訓練音色']
MODE_HELP = {
    'Zero-Shot 聲音克隆':
        '1. 上傳 prompt 音訊\n2. 填 prompt 文字\n3. 填要合成的文字\n4. 點生成',
    '跨語言克隆':
        '1. 上傳 prompt 音訊\n2. 填要合成的文字（可跨語言）\n3. 點生成',
    '風格/方言控制':
        '1. 上傳 prompt 音訊\n2. 從快速選單選方言/情緒\n3. 填要合成的文字\n4. 點生成',
    '預訓練音色':
        '1. 選預訓練音色\n2. 填要合成的文字\n3. 點生成',
}

INSTRUCT_PRESETS = {
    '閩南話': 'You are a helpful assistant. 请用闽南话表达。<|endofprompt|>',
    '廣東話': 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
    '四川話': 'You are a helpful assistant. 请用四川话表达。<|endofprompt|>',
    '上海話': 'You are a helpful assistant. 请用上海话表达。<|endofprompt|>',
    '東北話': 'You are a helpful assistant. 请用东北话表达。<|endofprompt|>',
    '開心': 'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',
    '傷心': 'You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>',
    '生氣': 'You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>',
    '快速': 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
    '慢速': 'You are a helpful assistant. 请用尽可能慢地语速说一句话。<|endofprompt|>',
    '佩奇': 'You are a helpful assistant. 我想体验一下小猪佩奇风格，可以吗？<|endofprompt|>',
    '機器人': 'You are a helpful assistant. 你可以尝试用机器人的方式解答吗？<|endofprompt|>',
    '大聲': 'You are a helpful assistant. Please say a sentence as loudly as possible.<|endofprompt|>',
    '小聲': 'You are a helpful assistant. Please say a sentence in a very soft voice.<|endofprompt|>',
}

MIN_PROMPT_SEC = 1.5
ENDOFPROMPT = '<|endofprompt|>'


# ==================== 工具函數 ====================
def diagnose_audio(filepath):
    try:
        wav, sr = torchaudio.load(filepath)
        duration = wav.shape[1] / sr
        max_amp = wav.abs().max().item()
        rms = wav.float().pow(2).mean().sqrt().item()
        info_str = (f'音訊診斷：{os.path.basename(filepath)}\n'
                    f'  取樣率={sr}, 長度={duration:.2f}s, '
                    f'通道={wav.shape[0]}, 樣本數={wav.shape[1]}\n'
                    f'  最大振幅={max_amp:.6f}, RMS={rms:.6f}')
        print(info_str)
        return duration, max_amp, info_str
    except Exception as e:
        print(f'[WARN] 無法分析音訊：{e}')
        return 0, 0, f'無法分析音訊：{e}'


def auto_transcribe(audio_path):
    if not audio_path:
        return ''
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        audio_np = waveform[0].numpy().astype(np.float32)
        result = whisper_model.transcribe(audio_np, language='zh')
        text = result['text'].strip()
        print(f'[ASR] Whisper: {text}')
        return text
    except Exception as e:
        print(f'[WARN] Whisper 轉寫失敗：{e}')
        return ''


def validate_prompt_wav(prompt_wav):
    if not prompt_wav:
        return False, '請提供 prompt 音訊！'
    try:
        duration, max_amp, info_str = diagnose_audio(prompt_wav)
        if duration == 0:
            return False, 'Prompt 音訊是空的，請重新錄製。'
        if duration < MIN_PROMPT_SEC:
            return False, f'Prompt 音訊太短（{duration:.1f}s），至少需要 {MIN_PROMPT_SEC}s。'
        if max_amp < 0.001:
            return False, (f'Prompt 音訊似乎是靜音（最大振幅={max_amp:.6f}）！\n'
                           f'建議改用「上傳」方式提供音訊。')
        return True, info_str
    except Exception as e:
        return False, f'無法讀取 prompt 音訊：{e}'


def ensure_endofprompt(text):
    if ENDOFPROMPT not in text:
        text = f'You are a helpful assistant.{ENDOFPROMPT}{text}'
    return text


def ensure_trailing_punct(text):
    """確保文字結尾有標點，避免模型提早停止導致最後一個字被截斷"""
    text = text.strip()
    if not text:
        return text
    end_puncts = set('。！？!?.，,；;：:、…—～）)】」』》〉')
    if text[-1] not in end_puncts:
        text += '。'
    return text


def postprocess_audio(all_speech):
    if not all_speech:
        return None
    audio = np.concatenate(all_speech)
    sr = cosyvoice.sample_rate
    trim_samples = int(sr * 0.05)
    if len(audio) > trim_samples + sr:
        audio = audio[trim_samples:]
    fade_len = int(sr * 0.05)
    if len(audio) > fade_len:
        fade_curve = np.linspace(0.0, 1.0, fade_len, dtype=audio.dtype)
        audio[:fade_len] *= fade_curve
    # 尾端加 0.3s 靜音保護，避免最後一個字被截斷
    tail_pad = np.zeros(int(sr * 0.3), dtype=audio.dtype)
    audio = np.concatenate([audio, tail_pad])
    return (sr, audio)


# ==================== 功能 1：名人 TTS ====================
def get_speaker_ref(speaker_name, prompt_choice=None):
    """取得說話人參考音檔"""
    prompts = SPEAKER_PROMPTS.get(speaker_name, {})
    names = SPEAKER_PROMPT_NAMES.get(speaker_name, [])
    key = prompt_choice if (prompt_choice and prompt_choice in prompts) else (names[0] if names and names[0] in prompts else None)
    if key and key in prompts:
        return prompts[key]
    return None, None


def generate_celebrity_tts(speaker_name, tts_text, prompt_choice, speed, seed):
    if not tts_text or not tts_text.strip():
        gr.Warning('請輸入要合成的文字！')
        return None

    # 切換說話人模型
    if not switch_speaker(speaker_name):
        gr.Warning(f'切換說話人 {speaker_name} 失敗！')
        return None

    wav_path, prompt_text = get_speaker_ref(speaker_name, prompt_choice)
    if not wav_path or not os.path.exists(wav_path):
        gr.Warning(f'請選擇一個{speaker_name}音檔片段！')
        return None

    tts_text = ensure_trailing_punct(tts_text)
    print(f'[TTS] [{speaker_name}] 參考片段：{os.path.basename(wav_path)}')
    print(f'[TTS] 文字：{tts_text}')
    all_speech = []
    try:
        set_all_random_seed(int(seed))
        for i in cosyvoice.inference_zero_shot(
            tts_text, prompt_text, wav_path,
            stream=False, speed=speed, text_frontend=False
        ):
            all_speech.append(i['tts_speech'].numpy().flatten())
    except RuntimeError as e:
        if 'Kernel size' in str(e):
            gr.Warning('生成失敗：請換一段文字再試。')
        else:
            gr.Warning(f'生成失敗：{e}')
        return None
    except Exception as e:
        gr.Warning(f'生成失敗：{e}')
        return None

    return postprocess_audio(all_speech)


def preview_builtin_prompt(speaker_name, prompt_choice):
    prompts = SPEAKER_PROMPTS.get(speaker_name, {})
    if prompt_choice not in prompts:
        return None
    wav_path, _ = prompts[prompt_choice]
    if os.path.exists(wav_path):
        wav, sr = torchaudio.load(wav_path)
        return (sr, wav[0].numpy())
    return None


def on_speaker_change(speaker_name):
    """切換說話人時更新 prompt 選單"""
    names = SPEAKER_PROMPT_NAMES.get(speaker_name, ['(找不到內建音檔)'])
    return gr.update(choices=names, value=names[0] if names else None)


# ==================== 功能 2：語音克隆 ====================
def generate_clone(clone_text, clone_wav, clone_mode, clone_speaker, clone_ref_choice, speed, seed):
    if not clone_wav:
        gr.Warning('請先上傳或錄製你的聲音！')
        return None, ''

    ok, msg = validate_prompt_wav(clone_wav)
    if not ok:
        gr.Warning(msg)
        return None, ''

    user_text = auto_transcribe(clone_wav)
    status = f'辨識到你說：「{user_text}」' if user_text else '未能辨識語音內容'

    spk = clone_speaker or SPEAKER_NAMES[0]

    if clone_mode == '用名人聲音重複你說的話':
        if not user_text:
            gr.Warning('無法辨識你的語音內容，請確認錄音品質。')
            return None, status

        # 切換說話人
        if not switch_speaker(spk):
            gr.Warning(f'切換說話人 {spk} 失敗！')
            return None, status

        ref_wav, ref_text = get_speaker_ref(spk, clone_ref_choice)
        if not ref_wav:
            gr.Warning(f'沒有內建{spk}音檔！')
            return None, status

        all_speech = []
        try:
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_zero_shot(
                ensure_trailing_punct(user_text), ref_text, ref_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())
        except Exception as e:
            gr.Warning(f'生成失敗：{e}')
            return None, status
        return postprocess_audio(all_speech), status

    elif clone_mode == '用名人聲音說自訂文字':
        if not clone_text or not clone_text.strip():
            gr.Warning(f'請輸入要讓{spk}說的文字！')
            return None, status

        if not switch_speaker(spk):
            gr.Warning(f'切換說話人 {spk} 失敗！')
            return None, status

        ref_wav, ref_text = get_speaker_ref(spk, clone_ref_choice)
        if not ref_wav:
            gr.Warning(f'沒有內建{spk}音檔！')
            return None, status

        all_speech = []
        try:
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_zero_shot(
                ensure_trailing_punct(clone_text), ref_text, ref_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())
        except Exception as e:
            gr.Warning(f'生成失敗：{e}')
            return None, status
        return postprocess_audio(all_speech), status

    elif clone_mode == '克隆你的聲音（用你的音色說自訂文字）':
        if not clone_text or not clone_text.strip():
            gr.Warning('請輸入要說的文字！')
            return None, status

        prompt_text_for_user = ensure_endofprompt(user_text) if user_text else ''
        if not prompt_text_for_user:
            gr.Warning('無法辨識你的語音，克隆需要 prompt 文字。')
            return None, status

        all_speech = []
        try:
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_zero_shot(
                clone_text, prompt_text_for_user, clone_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())
        except Exception as e:
            gr.Warning(f'生成失敗：{e}')
            return None, status
        return postprocess_audio(all_speech), status

    return None, status


# ==================== 功能 3：進階模式 ====================
def generate_advanced(tts_text, mode, sft_dropdown, prompt_text, prompt_wav,
                      instruct_text, seed, speed):
    if not tts_text or not tts_text.strip():
        gr.Warning('請輸入要合成的文字！')
        return None

    all_speech = []
    try:
        if mode == '預訓練音色':
            if not sft_dropdown or sft_dropdown == '':
                gr.Warning('沒有可用的預訓練音色！')
                return None
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False, speed=speed):
                all_speech.append(i['tts_speech'].numpy().flatten())

        elif mode == 'Zero-Shot 聲音克隆':
            ok, msg = validate_prompt_wav(prompt_wav)
            if not ok:
                gr.Warning(msg)
                return None
            if not prompt_text:
                gr.Warning('請輸入 prompt 文字！')
                return None
            prompt_text = ensure_endofprompt(prompt_text)
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())

        elif mode == '跨語言克隆':
            ok, msg = validate_prompt_wav(prompt_wav)
            if not ok:
                gr.Warning(msg)
                return None
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_cross_lingual(
                tts_text, prompt_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())

        elif mode == '風格/方言控制':
            ok, msg = validate_prompt_wav(prompt_wav)
            if not ok:
                gr.Warning(msg)
                return None
            if not instruct_text:
                gr.Warning('請輸入或選擇 instruct 風格！')
                return None
            instruct_text = ensure_endofprompt(instruct_text)
            set_all_random_seed(int(seed))
            for i in cosyvoice.inference_instruct2(
                tts_text, instruct_text, prompt_wav,
                stream=False, speed=speed, text_frontend=False
            ):
                all_speech.append(i['tts_speech'].numpy().flatten())

    except RuntimeError as e:
        err_msg = str(e)
        if 'Kernel size' in err_msg:
            gr.Warning('生成失敗：音訊太短，請提供更長的 prompt（建議 3~10 秒）。')
        else:
            gr.Warning(f'生成失敗：{err_msg}')
        return None
    except Exception as e:
        gr.Warning(f'生成失敗：{e}')
        return None

    return postprocess_audio(all_speech)


def on_preset_change(preset_name):
    if preset_name and preset_name in INSTRUCT_PRESETS:
        return (
            INSTRUCT_PRESETS[preset_name],
            '風格/方言控制',
            MODE_HELP['風格/方言控制'],
        )
    return '', gr.update(), gr.update()


# ==================== UI 介面 ====================
DEFAULT_SPK = SPEAKER_NAMES[0]  # 韓國瑜
DEFAULT_PROMPTS = SPEAKER_PROMPT_NAMES.get(DEFAULT_SPK, ['(找不到內建音檔)'])

with gr.Blocks(title='CosyVoice3 名人語音合成', theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        '# CosyVoice3 名人語音合成 WebUI\n'
        '> 微調版模型 -- 支援韓國瑜、豬哥亮 TTS、語音克隆、進階模式'
    )

    with gr.Tabs():
        # ======== Tab 1：名人 TTS ========
        with gr.Tab('名人 TTS'):
            gr.Markdown(
                '### 直接打字，名人幫你說！\n'
                '選一個說話人和參考音檔片段，輸入文字即可生成聲音。'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    celeb_speaker = gr.Dropdown(
                        choices=SPEAKER_NAMES,
                        value=DEFAULT_SPK,
                        label='🎤 選擇說話人',
                        info='切換時會自動載入對應的模型'
                    )
                    celeb_text = gr.Textbox(
                        label='要說的文字',
                        lines=3,
                        value='各位校長、各位老師、各位同學大家好，今天天氣真好，我們一起來聊聊天吧。',
                        placeholder='輸入任何你想讓名人說的話...'
                    )
                    celeb_prompt = gr.Dropdown(
                        choices=DEFAULT_PROMPTS,
                        value=DEFAULT_PROMPTS[0] if DEFAULT_PROMPTS else None,
                        label='選擇參考音檔（影響語氣和音色）',
                        info='從訓練資料中挑選的高品質片段'
                    )
                with gr.Column(scale=1):
                    gr.Markdown(
                        '**使用提示**\n\n'
                        '- 切換說話人時會自動\n  載入對應模型權重\n'
                        '- 選不同參考片段可得到\n  不同語氣效果\n'
                        '- 點「試聽」聽原始聲音\n'
                        '- 文字越長生成越慢\n'
                        '- 換種子可換語調變化'
                    )
                    celeb_preview = gr.Audio(label='參考音檔試聽', interactive=False)

            with gr.Row():
                celeb_speed = gr.Number(value=1.0, label='語速', minimum=0.5, maximum=2.0, step=0.1)
                celeb_seed = gr.Number(value=42, label='種子')
                celeb_seed_btn = gr.Button('隨機種子', scale=0)

            celeb_gen_btn = gr.Button('🎙️ 生成語音', variant='primary', size='lg')
            celeb_output = gr.Audio(label='合成結果', autoplay=True)

            # Tab 1 事件
            celeb_seed_btn.click(fn=lambda: random.randint(1, 99999999), outputs=celeb_seed)
            celeb_speaker.change(fn=on_speaker_change, inputs=celeb_speaker, outputs=celeb_prompt)
            celeb_prompt.change(
                fn=preview_builtin_prompt,
                inputs=[celeb_speaker, celeb_prompt],
                outputs=celeb_preview
            )
            celeb_gen_btn.click(
                fn=generate_celebrity_tts,
                inputs=[celeb_speaker, celeb_text, celeb_prompt, celeb_speed, celeb_seed],
                outputs=celeb_output
            )

        # ======== Tab 2：語音克隆 ========
        with gr.Tab('語音克隆'):
            gr.Markdown(
                '### 上傳/錄製你的聲音，進行語音克隆\n'
                '- **名人重複你說的話** -- 你說什麼，名人就說什麼\n'
                '- **名人說自訂文字** -- 你上傳聲音當參考，名人說你指定的文字\n'
                '- **克隆你的聲音** -- 用你的音色說自訂文字'
            )

            clone_speaker = gr.Dropdown(
                choices=SPEAKER_NAMES,
                value=DEFAULT_SPK,
                label='🎤 選擇說話人（模式 1 和 2 使用）',
                info='切換時自動更新參考音檔列表'
            )

            clone_mode = gr.Radio(
                choices=[
                    '用名人聲音重複你說的話',
                    '用名人聲音說自訂文字',
                    '克隆你的聲音（用你的音色說自訂文字）'
                ],
                label='克隆模式',
                value='用名人聲音重複你說的話'
            )

            clone_ref = gr.Dropdown(
                choices=DEFAULT_PROMPTS,
                value=DEFAULT_PROMPTS[0] if DEFAULT_PROMPTS else None,
                label='參考音檔（模式 1 和 2 使用，影響聲音）',
                info='選不同片段可得到不同語氣效果'
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('> 建議用「上傳」模式，ngrok 線上錄音可能錄到靜音。')
                    clone_wav = gr.Audio(
                        sources=['upload', 'microphone'], type='filepath',
                        label='上傳或錄製你的聲音'
                    )
                    clone_status = gr.Textbox(label='語音辨識結果', interactive=False, lines=2)
                with gr.Column(scale=1):
                    clone_text = gr.Textbox(
                        label='自訂文字（模式 2 和 3 需要填）',
                        lines=3,
                        placeholder='模式 1「重複你說的話」不需要填這裡',
                        value=''
                    )

            with gr.Row():
                clone_speed = gr.Number(value=1.0, label='語速', minimum=0.5, maximum=2.0, step=0.1)
                clone_seed = gr.Number(value=42, label='種子')
                clone_seed_btn = gr.Button('隨機', scale=0)

            clone_gen_btn = gr.Button('生成克隆語音', variant='primary', size='lg')
            clone_output = gr.Audio(label='克隆結果', autoplay=True)

            # Tab 2 事件
            clone_seed_btn.click(fn=lambda: random.randint(1, 99999999), outputs=clone_seed)
            clone_speaker.change(fn=on_speaker_change, inputs=clone_speaker, outputs=clone_ref)
            clone_gen_btn.click(
                fn=generate_clone,
                inputs=[clone_text, clone_wav, clone_mode, clone_speaker, clone_ref, clone_speed, clone_seed],
                outputs=[clone_output, clone_status]
            )

        # ======== Tab 3：進階模式 ========
        with gr.Tab('進階模式'):
            gr.Markdown('### 進階推理模式（4 種）')
            with gr.Row():
                with gr.Column(scale=2):
                    adv_text = gr.Textbox(
                        label='要合成的文字', lines=3,
                        value='今天天氣真好，我們一起去公園走走吧。'
                    )
                    adv_mode = gr.Radio(
                        choices=MODE_LIST, label='推理模式',
                        value=MODE_LIST[0]
                    )
                with gr.Column(scale=1):
                    adv_help = gr.Textbox(
                        label='操作說明',
                        value=MODE_HELP[MODE_LIST[0]],
                        interactive=False, lines=5
                    )

            with gr.Row():
                adv_sft = gr.Dropdown(
                    choices=sft_spk, label='預訓練音色', value=sft_spk[0]
                )
                adv_speed = gr.Number(value=1.0, label='語速', minimum=0.5, maximum=2.0, step=0.1)
                adv_seed = gr.Number(value=42, label='種子')
                adv_seed_btn = gr.Button('隨機', scale=0)

            gr.Markdown('#### Prompt 音訊')
            gr.Markdown('> 建議用「上傳」模式。')
            adv_wav = gr.Audio(
                sources=['upload', 'microphone'], type='filepath',
                label='上傳或錄製 prompt 音訊'
            )
            adv_info = gr.Textbox(label='音訊診斷', interactive=False, lines=2)
            adv_prompt_text = gr.Textbox(
                label='Prompt 文字（Zero-Shot 用）',
                placeholder='上傳音訊後自動辨識...', lines=2
            )

            gr.Markdown('#### 風格/方言')
            with gr.Row():
                adv_preset = gr.Dropdown(
                    choices=list(INSTRUCT_PRESETS.keys()),
                    label='快速選單', value=None
                )
                adv_instruct = gr.Textbox(
                    label='Instruct 文字',
                    placeholder='You are a helpful assistant. ...<|endofprompt|>',
                    lines=1
                )

            adv_gen_btn = gr.Button('生成音訊', variant='primary', size='lg')
            adv_output = gr.Audio(label='合成結果', autoplay=True)

            # Tab 3 事件
            def on_adv_audio(audio_path):
                if not audio_path:
                    return '(尚未提供音訊)', ''
                ok, msg = validate_prompt_wav(audio_path)
                prefix = '[OK]' if ok else '[X]'
                diag = f'{prefix} {msg}'
                if ok:
                    text = auto_transcribe(audio_path)
                    pt = f'You are a helpful assistant.<|endofprompt|>{text}' if text else ''
                else:
                    pt = ''
                return diag, pt

            adv_wav.change(fn=on_adv_audio, inputs=adv_wav, outputs=[adv_info, adv_prompt_text])
            adv_wav.stop_recording(fn=on_adv_audio, inputs=adv_wav, outputs=[adv_info, adv_prompt_text])
            adv_wav.upload(fn=on_adv_audio, inputs=adv_wav, outputs=[adv_info, adv_prompt_text])

            adv_seed_btn.click(fn=lambda: random.randint(1, 99999999), outputs=adv_seed)
            adv_mode.change(fn=lambda m: MODE_HELP[m], inputs=adv_mode, outputs=adv_help)
            adv_preset.change(
                fn=on_preset_change, inputs=adv_preset,
                outputs=[adv_instruct, adv_mode, adv_help]
            )
            adv_gen_btn.click(
                fn=generate_advanced,
                inputs=[adv_text, adv_mode, adv_sft, adv_prompt_text, adv_wav,
                        adv_instruct, adv_seed, adv_speed],
                outputs=adv_output
            )

# ==================== ngrok 隧道 ====================
print('[*] 正在建立 ngrok 隧道...')
import time as _time
try:
    ngrok.kill()
    _time.sleep(2)
except Exception:
    pass
conf.get_default().region = 'ap'

NGROK_DOMAIN = 'unferried-milo-unphlegmatically.ngrok-free.dev'
try:
    public_url = ngrok.connect(
        args.port, 'http',
        hostname=NGROK_DOMAIN
    ).public_url
except Exception:
    public_url = ngrok.connect(args.port, 'http').public_url

print('=' * 60)
print(f'公開連結：{public_url}')
print(f'本地連結：http://localhost:{args.port}')
print('=' * 60)

demo.queue(max_size=4, default_concurrency_limit=2)
demo.launch(
    server_name='0.0.0.0',
    server_port=args.port,
    share=False,
    show_error=True,
    root_path=public_url
)
