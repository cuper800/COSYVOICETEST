"""
CosyVoice3 聲音微調腳本（Windows 版）

用法：
  1. 準備音檔資料夾，結構如下：
     voice_data/
       韓國瑜/
         clip01.wav
         clip01.txt  ← 對應的逐字稿（UTF-8）
         clip02.wav
         clip02.txt
         ...

  2. 執行此腳本：
     python finetune_voice.py --voice_dir voice_data/韓國瑜 --speaker 韓國瑜

  音檔要求：
    - WAV 格式（16kHz 或更高皆可，腳本會自動轉換）
    - 每段 3~30 秒，建議 5~15 秒最佳
    - 總量建議 30 分鐘以上（越多越好）
    - 盡量乾淨無背景音樂/雜音
    - 每個 .wav 搭配一個同名 .txt（裡面是該段音檔的文字內容）

  或者用「自動切割+轉寫」模式（只需提供長音檔，不需 .txt）：
     python finetune_voice.py --voice_dir voice_data/韓國瑜 --speaker 韓國瑜 --auto_transcribe
"""
import os
import sys
import signal
import argparse
import shutil
import torch
import torchaudio
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Windows: 忽略 SIGINT 避免終端意外中斷
if sys.platform == 'win32':
    signal.signal(signal.SIGINT, signal.SIG_IGN)

sys.path.append('third_party/Matcha-TTS')

# ==================== 參數 ====================
parser = argparse.ArgumentParser(description='CosyVoice3 聲音微調')
parser.add_argument('--voice_dir', type=str, required=True,
                    help='音檔資料夾路徑，裡面放 .wav + .txt')
parser.add_argument('--speaker', type=str, required=True,
                    help='說話人名稱，例如 韓國瑜')
parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                    help='預訓練模型路徑')
parser.add_argument('--output_dir', type=str, default=None,
                    help='輸出目錄（預設為 finetune_output/<speaker>）')
parser.add_argument('--auto_transcribe', action='store_true',
                    help='自動用 Whisper 轉寫（不需要 .txt 檔案）')
parser.add_argument('--auto_split', action='store_true',
                    help='自動切割長音檔為短片段（搭配 --auto_transcribe）')
parser.add_argument('--max_clip_sec', type=float, default=15.0,
                    help='自動切割時每段最長秒數')
parser.add_argument('--min_clip_sec', type=float, default=3.0,
                    help='自動切割時每段最短秒數')
parser.add_argument('--stage', type=int, default=0,
                    help='從第幾步開始（0=全部, 1=embedding, 2=token, 3=parquet, 5=train）')
parser.add_argument('--stop_stage', type=int, default=5,
                    help='在第幾步停止')
parser.add_argument('--train_model', type=str, default='llm',
                    choices=['llm', 'flow', 'hifigan', 'all'],
                    help='訓練哪個模型（建議先訓 llm）')
parser.add_argument('--max_epoch', type=int, default=50,
                    help='最大訓練 epoch 數')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='學習率')
parser.add_argument('--num_workers', type=int, default=2,
                    help='資料載入線程數')
parser.add_argument('--dev_ratio', type=float, default=0.1,
                    help='驗證集比例')
args = parser.parse_args()

# ==================== 路徑設定 ====================
voice_dir = Path(args.voice_dir).resolve()
model_dir = Path(args.model_dir).resolve()
output_dir = Path(args.output_dir) if args.output_dir else Path(f'finetune_output/{args.speaker}')
output_dir = output_dir.resolve()

data_dir = output_dir / 'data'
train_dir = data_dir / 'train'
dev_dir = data_dir / 'dev'
exp_dir = output_dir / 'exp'

for d in [train_dir, dev_dir, exp_dir]:
    d.mkdir(parents=True, exist_ok=True)

print(f'📁 音檔來源：{voice_dir}')
print(f'📁 輸出目錄：{output_dir}')
print(f'🤖 預訓練模型：{model_dir}')
print(f'🎤 說話人：{args.speaker}')
print()


# ==================== Stage 0: 資料準備 ====================
def stage0_prepare_data():
    """準備 wav.scp / text / utt2spk / spk2utt / instruct"""
    print('=' * 60)
    print('📋 Stage 0: 準備訓練資料')
    print('=' * 60)

    wav_files = sorted(voice_dir.glob('*.wav'))
    if not wav_files:
        print(f'❌ 在 {voice_dir} 找不到 .wav 檔案！')
        sys.exit(1)

    # 如果需要自動切割長音檔
    if args.auto_split:
        wav_files = auto_split_audio(wav_files)

    # 如果需要自動轉寫
    if args.auto_transcribe:
        auto_transcribe_all(wav_files)

    # 收集 (wav_path, text) 對
    entries = []
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix('.txt')
        if not txt_path.exists():
            print(f'  ⚠️ 跳過 {wav_path.name}（找不到對應 .txt）')
            continue

        text = txt_path.read_text(encoding='utf-8').strip()
        if not text:
            print(f'  ⚠️ 跳過 {wav_path.name}（.txt 是空的）')
            continue

        # 檢查音檔長度
        try:
            info = torchaudio.info(str(wav_path))
            duration = info.num_frames / info.sample_rate
            if duration < args.min_clip_sec:
                print(f'  ⚠️ 跳過 {wav_path.name}（太短 {duration:.1f}s < {args.min_clip_sec}s）')
                continue
            if duration > 30:
                print(f'  ⚠️ 跳過 {wav_path.name}（太長 {duration:.1f}s > 30s）')
                continue
        except Exception as e:
            print(f'  ⚠️ 跳過 {wav_path.name}（無法讀取：{e}）')
            continue

        entries.append((wav_path, text, duration))

    if len(entries) < 3:
        print(f'❌ 有效音檔太少（只有 {len(entries)} 個），至少需要 3 個！')
        sys.exit(1)

    total_duration = sum(e[2] for e in entries)
    print(f'\n✅ 找到 {len(entries)} 個有效音檔，總計 {total_duration / 60:.1f} 分鐘')
    if total_duration < 60:
        print(f'  ⚠️ 警告：訓練資料只有 {total_duration:.0f} 秒，建議至少 30 分鐘以獲得好效果')

    # 分割 train / dev
    np.random.seed(42)
    indices = np.random.permutation(len(entries))
    n_dev = max(1, int(len(entries) * args.dev_ratio))
    dev_indices = set(indices[:n_dev])
    train_indices = set(indices[n_dev:])

    for split_name, split_dir, split_indices in [
        ('train', train_dir, train_indices),
        ('dev', dev_dir, dev_indices)
    ]:
        wav_scp = []
        texts = []
        utt2spk = []
        instructs = []

        for idx in sorted(split_indices):
            wav_path, text, duration = entries[idx]
            utt_id = f'{args.speaker}_{wav_path.stem}'
            wav_scp.append(f'{utt_id} {wav_path}')
            texts.append(f'{utt_id} {text}')
            utt2spk.append(f'{utt_id} {args.speaker}')
            instructs.append(f'{utt_id} You are a helpful assistant.<|endofprompt|>')

        # 寫入檔案
        (split_dir / 'wav.scp').write_text('\n'.join(wav_scp) + '\n', encoding='utf-8')
        (split_dir / 'text').write_text('\n'.join(texts) + '\n', encoding='utf-8')
        (split_dir / 'utt2spk').write_text('\n'.join(utt2spk) + '\n', encoding='utf-8')
        (split_dir / 'instruct').write_text('\n'.join(instructs) + '\n', encoding='utf-8')

        # spk2utt
        spk_utts = ' '.join(line.split()[0] for line in utt2spk)
        (split_dir / 'spk2utt').write_text(f'{args.speaker} {spk_utts}\n', encoding='utf-8')

        print(f'  📂 {split_name}: {len(wav_scp)} 個音檔')

    print()


def auto_split_audio(wav_files):
    """自動切割長音檔為短片段"""
    print('🔪 自動切割長音檔...')
    split_dir = voice_dir / '_splits'
    split_dir.mkdir(exist_ok=True)

    all_splits = []
    for wav_path in wav_files:
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            duration = waveform.shape[1] / sr
        except Exception:
            continue

        if duration <= args.max_clip_sec:
            all_splits.append(wav_path)
            continue

        # 切割
        max_samples = int(args.max_clip_sec * sr)
        n_chunks = int(np.ceil(waveform.shape[1] / max_samples))
        for i in range(n_chunks):
            start = i * max_samples
            end = min((i + 1) * max_samples, waveform.shape[1])
            chunk = waveform[:, start:end]
            chunk_duration = chunk.shape[1] / sr
            if chunk_duration < args.min_clip_sec:
                continue
            chunk_path = split_dir / f'{wav_path.stem}_part{i:03d}.wav'
            torchaudio.save(str(chunk_path), chunk, sr)
            all_splits.append(chunk_path)
            print(f'  ✂️ {wav_path.name} → part{i:03d} ({chunk_duration:.1f}s)')

    print(f'  共 {len(all_splits)} 個片段')
    return all_splits


def auto_transcribe_all(wav_files):
    """用 Whisper 自動轉寫所有音檔"""
    import whisper
    print('🗣️ 載入 Whisper 進行自動轉寫...')
    model = whisper.load_model('base')
    print('✅ Whisper 載入完成')

    for wav_path in tqdm(wav_files, desc='轉寫中'):
        txt_path = wav_path.with_suffix('.txt')
        if txt_path.exists():
            continue  # 已有文字檔就跳過
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            audio_np = waveform[0].numpy().astype(np.float32)
            result = model.transcribe(audio_np, language='zh')
            text = result['text'].strip()
            if text:
                txt_path.write_text(text, encoding='utf-8')
                print(f'  ✅ {wav_path.name} → {text[:50]}...')
            else:
                print(f'  ⚠️ {wav_path.name} → 轉寫為空')
        except Exception as e:
            print(f'  ⚠️ {wav_path.name} 轉寫失敗：{e}')
    print()


# ==================== Stage 1: 提取 Speaker Embedding ====================
def stage1_extract_embedding():
    print('=' * 60)
    print('🎯 Stage 1: 提取 Speaker Embedding')
    print('=' * 60)

    import onnxruntime
    import torchaudio.compliance.kaldi as kaldi

    onnx_path = str(model_dir / 'campplus.onnx')
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=option,
        providers=["CPUExecutionProvider"]
    )

    for split_name, split_dir in [('train', train_dir), ('dev', dev_dir)]:
        wav_scp = {}
        utt2spk_map = {}

        with open(split_dir / 'wav.scp', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_scp[parts[0]] = parts[1]

        with open(split_dir / 'utt2spk', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt2spk_map[parts[0]] = parts[1]

        utt2embedding = {}
        spk2embedding = {}

        for utt, wav_path in tqdm(wav_scp.items(), desc=f'Embedding ({split_name})'):
            try:
                audio, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
                feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                embedding = ort_session.run(
                    None,
                    {ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
                )[0].flatten().tolist()
                utt2embedding[utt] = embedding
                spk = utt2spk_map.get(utt, args.speaker)
                if spk not in spk2embedding:
                    spk2embedding[spk] = []
                spk2embedding[spk].append(embedding)
            except Exception as e:
                print(f'  ⚠️ {utt} embedding 提取失敗：{e}')

        for k, v in spk2embedding.items():
            spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

        torch.save(utt2embedding, str(split_dir / 'utt2embedding.pt'))
        torch.save(spk2embedding, str(split_dir / 'spk2embedding.pt'))
        print(f'  ✅ {split_name}: {len(utt2embedding)} embeddings')
    print()


# ==================== Stage 2: 提取 Speech Token ====================
def stage2_extract_speech_token():
    print('=' * 60)
    print('🔤 Stage 2: 提取 Speech Token')
    print('=' * 60)

    import onnxruntime
    import whisper

    onnx_path = str(model_dir / 'speech_tokenizer_v3.onnx')
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    providers = ["CUDAExecutionProvider"]
    try:
        ort_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=option, providers=providers
        )
        print('  ✅ 使用 CUDA 提取 speech token')
    except Exception:
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=option, providers=providers
        )
        print('  ⚠️ CUDA 不可用，改用 CPU（速度較慢）')

    for split_name, split_dir in [('train', train_dir), ('dev', dev_dir)]:
        wav_scp = {}
        with open(split_dir / 'wav.scp', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_scp[parts[0]] = parts[1]

        utt2speech_token = {}
        for utt, wav_path in tqdm(wav_scp.items(), desc=f'Speech Token ({split_name})'):
            try:
                audio, sr = torchaudio.load(wav_path, backend='soundfile')
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                if sr != 16000:
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio)

                if audio.shape[1] / 16000 > 30:
                    print(f'  ⚠️ {utt} 超過 30 秒，跳過')
                    continue

                # 用 Whisper 提取 128 維 log-mel spectrogram（這是 speech_tokenizer_v3 的必要輸入）
                feat = whisper.log_mel_spectrogram(audio, n_mels=128)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)

                speech_token = ort_session.run(
                    None,
                    {
                        ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                        ort_session.get_inputs()[1].name: feat_len
                    }
                )[0].flatten().tolist()
                utt2speech_token[utt] = speech_token
            except Exception as e:
                print(f'  ⚠️ {utt} token 提取失敗：{e}')

        torch.save(utt2speech_token, str(split_dir / 'utt2speech_token.pt'))
        print(f'  ✅ {split_name}: {len(utt2speech_token)} utterances')
    print()


# ==================== Stage 3: 建立 Parquet ====================
def stage3_make_parquet():
    print('=' * 60)
    print('📦 Stage 3: 建立 Parquet 資料集')
    print('=' * 60)

    import pandas as pd

    for split_name, split_dir in [('train', train_dir), ('dev', dev_dir)]:
        parquet_dir = split_dir / 'parquet'
        parquet_dir.mkdir(exist_ok=True)

        # 讀取所有資料
        wav_scp = {}
        with open(split_dir / 'wav.scp', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_scp[parts[0]] = parts[1]

        text_map = {}
        with open(split_dir / 'text', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    text_map[parts[0]] = parts[1]

        utt2spk_map = {}
        with open(split_dir / 'utt2spk', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt2spk_map[parts[0]] = parts[1]

        instruct_map = {}
        instruct_file = split_dir / 'instruct'
        if instruct_file.exists():
            with open(instruct_file, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        instruct_map[parts[0]] = parts[1]

        # 載入 embedding 和 token
        utt2embedding = torch.load(str(split_dir / 'utt2embedding.pt'), map_location='cpu') \
            if (split_dir / 'utt2embedding.pt').exists() else {}
        spk2embedding = torch.load(str(split_dir / 'spk2embedding.pt'), map_location='cpu') \
            if (split_dir / 'spk2embedding.pt').exists() else {}
        utt2speech_token = torch.load(str(split_dir / 'utt2speech_token.pt'), map_location='cpu') \
            if (split_dir / 'utt2speech_token.pt').exists() else {}

        # 用 pandas 建立 DataFrame（和官方 make_parquet_list.py 一致）
        utt_list = []
        audio_data_list = []
        wav_list = []
        text_list = []
        spk_list = []
        utt_emb_list = []
        spk_emb_list = []
        speech_tok_list = []
        instruct_list = []

        for utt in wav_scp:
            wav_path = wav_scp[utt]
            try:
                with open(wav_path, 'rb') as f:
                    audio_data = f.read()
            except Exception:
                continue

            utt_list.append(utt)
            audio_data_list.append(audio_data)
            wav_list.append(wav_path)
            text_list.append(text_map.get(utt, ''))
            spk_list.append(utt2spk_map.get(utt, args.speaker))

            # embedding 和 token 存原始 list（不要用 str()）
            if utt in utt2embedding:
                emb = utt2embedding[utt]
                utt_emb_list.append(emb if isinstance(emb, list) else emb)
            else:
                utt_emb_list.append(None)

            spk = utt2spk_map.get(utt, args.speaker)
            if spk in spk2embedding:
                emb = spk2embedding[spk]
                spk_emb_list.append(emb if isinstance(emb, list) else emb)
            else:
                spk_emb_list.append(None)

            if utt in utt2speech_token:
                speech_tok_list.append(utt2speech_token[utt])
            else:
                speech_tok_list.append([])

            instruct_list.append(instruct_map.get(utt, ''))

        if not utt_list:
            print(f'  ⚠️ {split_name}: 沒有有效資料')
            continue

        df = pd.DataFrame()
        df['utt'] = utt_list
        df['audio_data'] = audio_data_list
        df['wav'] = wav_list
        df['text'] = text_list
        df['spk'] = spk_list
        df['utt_embedding'] = utt_emb_list
        df['spk_embedding'] = spk_emb_list
        df['speech_token'] = speech_tok_list
        df['instruct'] = instruct_list

        parquet_path = parquet_dir / 'parquet_00000000.parquet'
        df.to_parquet(str(parquet_path))

        # 建立 data.list
        (parquet_dir / 'data.list').write_text(str(parquet_path.resolve()) + '\n', encoding='utf-8')

        print(f'  ✅ {split_name}: {len(utt_list)} 筆資料 → {parquet_path.name}')
    print()


# ==================== Stage 5: 訓練 ====================
def stage5_train():
    print('=' * 60)
    print('🚀 Stage 5: 開始訓練')
    print('=' * 60)

    train_data_list = str(train_dir / 'parquet' / 'data.list')
    dev_data_list = str(dev_dir / 'parquet' / 'data.list')

    if not Path(train_data_list).exists():
        print('❌ 找不到訓練資料 data.list，請先完成 Stage 0~3')
        return

    # 準備 config（複製 + 修改學習率等）
    src_config = model_dir / 'cosyvoice3.yaml'
    if not src_config.exists():
        src_config = Path('examples/libritts/cosyvoice3/conf/cosyvoice3.yaml')
    config_path = output_dir / 'cosyvoice3_finetune.yaml'
    shutil.copy2(str(src_config), str(config_path))

    # 修改 config 以適配 SFT
    config_text = config_path.read_text(encoding='utf-8')
    # SFT 需要 use_spk_embedding: True
    config_text = config_text.replace('use_spk_embedding: False', 'use_spk_embedding: True')
    # 設定 max_epoch
    import re
    config_text = re.sub(r'max_epoch:\s*\d+', f'max_epoch: {args.max_epoch}', config_text)
    config_path.write_text(config_text, encoding='utf-8')

    models_to_train = ['llm', 'flow', 'hifigan'] if args.train_model == 'all' else [args.train_model]

    # 確保 PYTHONPATH 包含工作目錄（torchrun 子程序需要）
    project_root = str(Path('.').resolve())
    third_party = str(Path('third_party/Matcha-TTS').resolve())
    existing = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f'{project_root};{third_party};{existing}'

    for model_name in models_to_train:
        print(f'\n--- 訓練 {model_name} ---')
        model_exp_dir = exp_dir / model_name
        model_exp_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = str(model_dir / f'{model_name}.pt')
        if not Path(checkpoint).exists():
            if model_name == 'hifigan':
                checkpoint = str(model_dir / 'hift.pt')
            if not Path(checkpoint).exists():
                print(f'  ⚠️ 找不到預訓練 checkpoint {checkpoint}，跳過')
                continue

        cmd = (
            f'torchrun --nnodes=1 --nproc_per_node=1 '
            f'--rdzv_id=1986 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234 '
            f'cosyvoice/bin/train.py '
            f'--train_engine torch_ddp '
            f'--config "{config_path}" '
            f'--train_data "{train_data_list}" '
            f'--cv_data "{dev_data_list}" '
            f'--qwen_pretrain_path "{model_dir / "CosyVoice-BlankEN"}" '
            f'--onnx_path "{model_dir}" '
            f'--model {model_name} '
            f'--checkpoint "{checkpoint}" '
            f'--model_dir "{model_exp_dir}" '
            f'--tensorboard_dir "{output_dir / "tensorboard" / model_name}" '
            f'--ddp.dist_backend gloo '
            f'--num_workers {args.num_workers} '
            f'--prefetch 100 '
            f'--pin_memory '
            f'--use_amp'
        )

        print(f'  📌 指令：\n    {cmd}\n')
        print(f'  💡 提示：訓練開始後可以用 Ctrl+C 中斷')
        print(f'  💡 checkpoint 會存在 {model_exp_dir}')
        print()

        # Windows 下用 subprocess + CREATE_NEW_PROCESS_GROUP 隔離 SIGINT
        import subprocess, sys
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        proc = subprocess.Popen(
            cmd,
            shell=True,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        proc.wait()

    print('\n✅ 訓練完成！')
    print(f'   checkpoint 位於：{exp_dir}')
    print()
    print('📌 下一步：將訓練好的模型複製到預訓練目錄使用')
    print(f'   例如：copy {exp_dir}\\llm\\llm.pt {model_dir}\\llm.pt')
    print()


# ==================== 主流程 ====================
def main():
    print('🎙️ CosyVoice3 聲音微調工具')
    print('=' * 60)
    print()

    if args.stage <= 0 and args.stop_stage >= 0:
        stage0_prepare_data()

    if args.stage <= 1 and args.stop_stage >= 1:
        stage1_extract_embedding()

    if args.stage <= 2 and args.stop_stage >= 2:
        stage2_extract_speech_token()

    if args.stage <= 3 and args.stop_stage >= 3:
        stage3_make_parquet()

    if args.stage <= 5 and args.stop_stage >= 5:
        stage5_train()

    print('🎉 全部完成！')
    print()
    print('=' * 60)
    print('📖 使用微調後的模型：')
    print(f'   1. 備份原始模型：')
    print(f'      copy {model_dir}\\llm.pt {model_dir}\\llm.pt.bak')
    print(f'   2. 覆蓋模型：')
    print(f'      copy {exp_dir}\\llm\\llm.pt {model_dir}\\llm.pt')
    print(f'   3. 重新啟動 WebUI 即可使用')
    print('=' * 60)


if __name__ == '__main__':
    main()
