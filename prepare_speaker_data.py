"""
說話人分離腳本 — 從多人音檔中只提取目標說話人的語音

使用方式：
  步驟 1: 準備一個參考音檔資料夾，裡面放 2~5 段「只有目標說話人在講話」的短音檔（3~10 秒）
          例如：ref_clips/豬哥亮/ref01.wav, ref02.wav ...

  步驟 2: 準備要處理的長音檔（可以是多人對話的影片/音檔）
          例如：raw_audio/豬哥亮/show01.wav, show02.mp3, show03.mp4 ...

  步驟 3: 執行此腳本：
     python prepare_speaker_data.py \
       --ref_dir ref_clips/豬哥亮 \
       --audio_dir raw_audio/豬哥亮 \
       --output_dir voice_data/豬哥亮 \
       --speaker 豬哥亮

  步驟 4: 腳本會：
     a) 把影片/音檔轉換為 WAV
     b) 用 Silero-VAD 偵測語音段落
     c) 用 campplus 模型提取每段的說話人嵌入向量
     d) 與參考音檔比對（cosine similarity）
     e) 只保留高相似度（= 目標說話人）的片段
     f) 用 Whisper 自動轉寫
     g) 輸出到 voice_data/豬哥亮/clip_XXXX.wav + clip_XXXX.txt

  步驟 5: 接下來就可以用 finetune_voice.py 訓練：
     python finetune_voice.py --voice_dir voice_data/豬哥亮 --speaker 豬哥亮

  可調參數：
    --threshold    相似度閾值（預設 0.65，越高越嚴格，越低保留越多片段）
    --min_sec      最短片段秒數（預設 2.0）
    --max_sec      最長片段秒數（預設 20.0）
    --preview      預覽模式，只處理前 N 段讓你聽聽看，不做轉寫
"""
import os
import sys
import argparse
import shutil
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
from tqdm import tqdm

# --- Windows 編碼修正 ---
if sys.platform == 'win32':
    for s in [sys.stdout, sys.stderr]:
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

warnings.filterwarnings('ignore')

# ============================================================
#  參數解析
# ============================================================
parser = argparse.ArgumentParser(description='說話人分離：從多人音檔中提取目標說話人語音')
parser.add_argument('--ref_dir', type=str, required=True,
                    help='參考音檔資料夾（放 2~5 段只有目標說話人的短音檔）')
parser.add_argument('--audio_dir', type=str, required=True,
                    help='要處理的長音檔/影片資料夾')
parser.add_argument('--output_dir', type=str, required=True,
                    help='輸出資料夾（clip_XXXX.wav + clip_XXXX.txt）')
parser.add_argument('--speaker', type=str, required=True,
                    help='目標說話人名稱')
parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                    help='CosyVoice 預訓練模型路徑（含 campplus.onnx）')
parser.add_argument('--threshold', type=float, default=0.65,
                    help='cosine similarity 閾值（0~1，預設 0.65）')
parser.add_argument('--min_sec', type=float, default=2.0,
                    help='片段最短秒數')
parser.add_argument('--max_sec', type=float, default=20.0,
                    help='片段最長秒數')
parser.add_argument('--target_sr', type=int, default=24000,
                    help='輸出音檔取樣率（CosyVoice3 預設 24000）')
parser.add_argument('--whisper_model', type=str, default='base',
                    help='Whisper 模型大小')
parser.add_argument('--preview', type=int, default=0,
                    help='預覽模式：只處理前 N 段（0=全部處理）')
parser.add_argument('--no_transcribe', action='store_true',
                    help='不執行 Whisper 轉寫（只分離音檔）')
parser.add_argument('--vad_threshold', type=float, default=0.4,
                    help='Silero-VAD 語音偵測閾值（0~1，預設 0.4）')
parser.add_argument('--merge_gap', type=float, default=0.5,
                    help='VAD 片段間隔小於此秒數則合併（預設 0.5）')

args = parser.parse_args()

# ============================================================
#  全域路徑
# ============================================================
REF_DIR    = Path(args.ref_dir).resolve()
AUDIO_DIR  = Path(args.audio_dir).resolve()
OUTPUT_DIR = Path(args.output_dir).resolve()
MODEL_DIR  = Path(args.model_dir).resolve()
CAMPPLUS   = MODEL_DIR / 'campplus.onnx'
TEMP_DIR   = OUTPUT_DIR / '_temp'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
              '.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts'}


# ============================================================
#  工具函數
# ============================================================
def ffmpeg_to_wav(src: Path, dst: Path, sr: int = 16000) -> bool:
    """用 ffmpeg 把任意音檔/影片轉成單聲道 WAV"""
    if dst.exists():
        return True
    cmd = [
        'ffmpeg', '-y', '-i', str(src),
        '-ac', '1', '-ar', str(sr),
        '-acodec', 'pcm_s16le',
        '-loglevel', 'error',
        str(dst)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except FileNotFoundError:
        print("[ERROR] 找不到 ffmpeg，請先安裝 ffmpeg 並加入 PATH")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg 轉換失敗: {src}\n  {e.stderr.decode(errors='replace')}")
        return False


def load_audio(path, target_sr: int = 16000) -> torch.Tensor:
    """讀取音檔並重取樣（支援 WAV/MP3 等格式）"""
    path = str(path)
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio


def extract_embedding(audio_16k: torch.Tensor, session: onnxruntime.InferenceSession) -> np.ndarray:
    """用 campplus 提取說話人嵌入向量"""
    if audio_16k.dim() == 1:
        audio_16k = audio_16k.unsqueeze(0)
    feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = session.run(
        None,
        {session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
    )[0].flatten()
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """計算 cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================
#  主流程
# ============================================================
def main():
    print("=" * 60)
    print(f"  說話人分離腳本")
    print(f"  目標說話人: {args.speaker}")
    print(f"  相似度閾值: {args.threshold}")
    print(f"  片段長度:   {args.min_sec}s ~ {args.max_sec}s")
    print("=" * 60)

    # ------ Step 1: 載入 campplus 模型 ------
    print("\n[1/6] 載入 campplus 說話人嵌入模型...")
    if not CAMPPLUS.exists():
        print(f"[ERROR] 找不到 {CAMPPLUS}")
        sys.exit(1)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 4
    ort_session = onnxruntime.InferenceSession(
        str(CAMPPLUS), sess_options=option, providers=["CPUExecutionProvider"]
    )
    print("  campplus.onnx 載入完成")

    # ------ Step 2: 提取參考音檔的嵌入向量 ------
    print("\n[2/6] 提取參考音檔嵌入向量...")
    ref_files = sorted([f for f in REF_DIR.iterdir()
                        if f.suffix.lower() in AUDIO_EXTS and f.is_file()])
    if len(ref_files) == 0:
        print(f"[ERROR] 在 {REF_DIR} 找不到任何音檔！")
        print("  請放入 2~5 段只有目標說話人在講話的短音檔（3~10 秒）")
        sys.exit(1)
    print(f"  找到 {len(ref_files)} 個參考音檔")

    ref_embeddings = []
    for rf in ref_files:
        try:
            audio_16k = load_audio(rf, 16000)
            dur = audio_16k.shape[-1] / 16000
            emb = extract_embedding(audio_16k, ort_session)
            ref_embeddings.append(emb)
            print(f"    {rf.name} ({dur:.1f}s) -> embedding OK")
        except Exception as e:
            print(f"    {rf.name} -> FAILED: {e}")

    if len(ref_embeddings) == 0:
        print("[ERROR] 沒有成功提取任何參考嵌入！")
        sys.exit(1)

    # 取平均作為目標說話人的代表嵌入
    ref_mean = np.mean(ref_embeddings, axis=0)
    ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-8)
    print(f"  參考嵌入維度: {ref_mean.shape[0]}，已計算平均向量")

    # ------ Step 3: 載入 Silero-VAD ------
    print("\n[3/6] 載入 Silero-VAD 語音偵測模型...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    get_speech_timestamps = vad_utils[0]
    print("  Silero-VAD 載入完成")

    # ------ Step 4: 處理所有音檔 ------
    print(f"\n[4/6] 掃描音檔資料夾: {AUDIO_DIR}")
    audio_files = sorted([f for f in AUDIO_DIR.iterdir()
                          if f.suffix.lower() in AUDIO_EXTS and f.is_file()])
    if len(audio_files) == 0:
        print(f"[ERROR] 在 {AUDIO_DIR} 找不到任何音檔/影片！")
        sys.exit(1)
    print(f"  找到 {len(audio_files)} 個檔案")

    all_segments = []  # [(audio_tensor_24k, similarity, src_file, start_sec, end_sec)]
    total_kept = 0
    total_rejected = 0

    for fi, af in enumerate(audio_files):
        print(f"\n  --- [{fi+1}/{len(audio_files)}] {af.name} ---")

        # 讀取音檔並轉成 16kHz mono
        try:
            audio_16k = load_audio(af, 16000).squeeze(0)  # (samples,)
        except Exception as e:
            # 如果 torchaudio 讀不了，嘗試用 ffmpeg 轉 WAV
            wav_16k_path = TEMP_DIR / f"{af.stem}_16k.wav"
            if ffmpeg_to_wav(af, wav_16k_path, 16000):
                try:
                    audio_16k = load_audio(wav_16k_path, 16000).squeeze(0)
                except Exception as e2:
                    print(f"    讀取失敗: {e2}")
                    continue
            else:
                print(f"    讀取失敗: {e}")
                continue

        total_dur = audio_16k.shape[0] / 16000
        print(f"    總長度: {total_dur:.1f}s")

        # VAD 偵測語音段落（分段處理長音檔，每段 5 分鐘）
        CHUNK_SEC = 300  # 5 分鐘一段
        chunk_samples = CHUNK_SEC * 16000
        speech_timestamps = []
        n_chunks = max(1, int(np.ceil(audio_16k.shape[0] / chunk_samples)))
        print(f"    分 {n_chunks} 段處理 VAD...")
        for ci in range(n_chunks):
            c_start = ci * chunk_samples
            c_end = min((ci + 1) * chunk_samples, audio_16k.shape[0])
            chunk = audio_16k[c_start:c_end]
            try:
                # 重置 VAD 狀態
                vad_model.reset_states()
                ts = get_speech_timestamps(
                    chunk,
                    vad_model,
                    threshold=args.vad_threshold,
                    sampling_rate=16000,
                    min_speech_duration_ms=int(args.min_sec * 1000),
                    max_speech_duration_s=args.max_sec * 2,
                    min_silence_duration_ms=300,
                    speech_pad_ms=100,
                )
                # 偏移到全域位置
                for t in ts:
                    t['start'] += c_start
                    t['end'] += c_start
                speech_timestamps.extend(ts)
            except Exception as e:
                print(f"    VAD chunk {ci+1}/{n_chunks} 失敗: {e}")
                continue

        if len(speech_timestamps) == 0:
            print(f"    未偵測到語音片段")
            continue

        # 合併太近的片段
        merged = []
        for ts in speech_timestamps:
            if merged and (ts['start'] - merged[-1]['end']) / 16000 < args.merge_gap:
                merged[-1]['end'] = ts['end']
            else:
                merged.append(dict(ts))

        # 再切割太長的片段
        final_segments = []
        max_samples = int(args.max_sec * 16000)
        for seg in merged:
            seg_len = seg['end'] - seg['start']
            if seg_len <= max_samples:
                final_segments.append(seg)
            else:
                # 按 max_sec 切割
                pos = seg['start']
                while pos < seg['end']:
                    chunk_end = min(pos + max_samples, seg['end'])
                    if (chunk_end - pos) / 16000 >= args.min_sec:
                        final_segments.append({'start': pos, 'end': chunk_end})
                    pos = chunk_end

        print(f"    VAD 偵測到 {len(speech_timestamps)} 段語音，合併/切割後 {len(final_segments)} 段")

        # 從 16kHz resample 到目標取樣率（24kHz）用於最終輸出
        audio_24k = torchaudio.transforms.Resample(16000, args.target_sr)(audio_16k.unsqueeze(0)).squeeze(0)

        # 對每個片段提取嵌入並比對
        kept_this_file = 0
        rejected_this_file = 0

        for seg in final_segments:
            start_s = seg['start'] / 16000
            end_s = seg['end'] / 16000
            dur = end_s - start_s

            if dur < args.min_sec:
                continue

            # 提取 16kHz 片段的嵌入
            seg_audio = audio_16k[seg['start']:seg['end']]
            try:
                emb = extract_embedding(seg_audio.unsqueeze(0), ort_session)
            except Exception:
                rejected_this_file += 1
                continue

            sim = cosine_sim(emb, ref_mean)

            if sim >= args.threshold:
                # 提取 24kHz 片段用於輸出
                start_24k = int(start_s * args.target_sr)
                end_24k = int(end_s * args.target_sr)
                seg_24k = audio_24k[start_24k:end_24k]

                all_segments.append((seg_24k, sim, af.name, start_s, end_s))
                kept_this_file += 1
            else:
                rejected_this_file += 1

        total_kept += kept_this_file
        total_rejected += rejected_this_file
        print(f"    保留 {kept_this_file} 段（{args.speaker}），丟棄 {rejected_this_file} 段（其他人）")

        if args.preview > 0 and total_kept >= args.preview:
            print(f"\n  [預覽模式] 已收集 {total_kept} 段，停止處理")
            break

    # ------ Step 5: 儲存分離結果 ------
    print(f"\n[5/6] 儲存分離結果...")
    print(f"  總共保留: {total_kept} 段")
    print(f"  總共丟棄: {total_rejected} 段")

    if total_kept == 0:
        print("[ERROR] 沒有找到任何符合目標說話人的片段！")
        print("  建議：")
        print("  1. 降低 --threshold（目前 {:.2f}），試試 0.5 或 0.55".format(args.threshold))
        print("  2. 確認參考音檔確實是目標說話人的聲音")
        print("  3. 確認音檔中確實有目標說話人在講話")
        sys.exit(1)

    # 按相似度排序（高到低）
    all_segments.sort(key=lambda x: -x[1])

    # 限制預覽模式
    if args.preview > 0:
        all_segments = all_segments[:args.preview]

    # 儲存
    total_duration = 0.0
    manifest = []

    for i, (seg_audio, sim, src_name, start_s, end_s) in enumerate(all_segments):
        clip_name = f"clip_{i:04d}"
        wav_path = OUTPUT_DIR / f"{clip_name}.wav"
        dur = seg_audio.shape[0] / args.target_sr
        total_duration += dur

        # 儲存 WAV
        torchaudio.save(str(wav_path), seg_audio.unsqueeze(0), args.target_sr)
        manifest.append({
            'clip': clip_name,
            'src': src_name,
            'start': f"{start_s:.2f}",
            'end': f"{end_s:.2f}",
            'dur': f"{dur:.2f}",
            'sim': f"{sim:.4f}",
        })

    # 寫入來源清單（方便查閱）
    manifest_path = OUTPUT_DIR / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f"說話人: {args.speaker}\n")
        f.write(f"相似度閾值: {args.threshold}\n")
        f.write(f"片段數: {len(manifest)}\n")
        f.write(f"總時長: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分鐘)\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'clip':<14} {'src':<25} {'start':>8} {'end':>8} {'dur':>6} {'sim':>8}\n")
        f.write(f"{'-'*70}\n")
        for m in manifest:
            f.write(f"{m['clip']:<14} {m['src']:<25} {m['start']:>8} {m['end']:>8} {m['dur']:>6} {m['sim']:>8}\n")

    print(f"  已儲存 {len(manifest)} 個音檔到 {OUTPUT_DIR}")
    print(f"  總時長: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分鐘)")
    print(f"  來源清單: {manifest_path}")

    # ------ Step 6: Whisper 轉寫 ------
    if args.no_transcribe or args.preview > 0:
        if args.preview > 0:
            print(f"\n[6/6] 預覽模式 - 跳過轉寫")
            print(f"  請到 {OUTPUT_DIR} 聽聽看這些片段是不是目標說話人")
            print(f"  如果太多雜音：提高 --threshold（目前 {args.threshold}）")
            print(f"  如果片段太少：降低 --threshold")
        else:
            print(f"\n[6/6] 已跳過轉寫（--no_transcribe）")
    else:
        print(f"\n[6/6] 用 Whisper ({args.whisper_model}) 轉寫 {len(manifest)} 個片段...")
        try:
            import whisper
            wmodel = whisper.load_model(args.whisper_model)
            print(f"  Whisper {args.whisper_model} 載入完成")

            success_count = 0
            for i, m in enumerate(tqdm(manifest, desc="  轉寫進度")):
                wav_path = OUTPUT_DIR / f"{m['clip']}.wav"
                txt_path = OUTPUT_DIR / f"{m['clip']}.txt"
                try:
                    result = wmodel.transcribe(
                        str(wav_path),
                        language='zh',
                        initial_prompt='以下是正體中文語音轉寫。'
                    )
                    text = result['text'].strip()
                    if text:
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        success_count += 1
                except Exception as e:
                    print(f"    {m['clip']} 轉寫失敗: {e}")

            print(f"  轉寫完成: {success_count}/{len(manifest)} 個片段")
        except ImportError:
            print("  [WARN] 未安裝 whisper，跳過轉寫")
            print("  安裝方式: pip install openai-whisper")

    # ------ 清理暫存 ------
    print(f"\n清理暫存資料夾...")
    try:
        shutil.rmtree(str(TEMP_DIR), ignore_errors=True)
    except Exception:
        pass

    # ------ 完成 ------
    print("\n" + "=" * 60)
    print(f"  完成！")
    print(f"  輸出目錄: {OUTPUT_DIR}")
    print(f"  片段數量: {len(manifest)}")
    print(f"  總時長:   {total_duration:.1f} 秒 ({total_duration/60:.1f} 分鐘)")
    if not args.no_transcribe and args.preview == 0:
        print(f"\n  下一步：用 finetune_voice.py 訓練")
        print(f"    python finetune_voice.py --voice_dir {OUTPUT_DIR} --speaker {args.speaker}")
    print("=" * 60)


if __name__ == '__main__':
    main()
