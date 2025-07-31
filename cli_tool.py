import argparse
import json
import sys
import os
import traceback
import torch
import numpy as np
from translation_module import generate_bilingual_srt_content, clear_caches
from faster_whisper import WhisperModel
import subprocess
import pysrt
import tempfile
import time
from moviepy.editor import AudioFileClip

def load_config(path="config.json"):
    """載入設定檔"""
    if not os.path.exists(path):
        print(f"[錯誤] 設定檔 '{path}' 不存在。", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[錯誤] 讀取設定檔 '{path}' 失敗: {e}", file=sys.stderr)
        sys.exit(1)

def filter_segments_by_energy(segments, audio_path, energy_threshold=0.005):
    """
    [v3.0 修正] 透過更穩健的方式讀取音訊，避免 moviepy 產生非預期資料結構。
    """
    try:
        print(f"[資訊] 正在進行音訊能量分析以過濾幻聽...")
        
        with AudioFileClip(audio_path) as clip:
            # 確保音訊以 16000Hz 讀取
            audio_array = clip.to_soundarray(fps=16000, nbytes=2)
            if not isinstance(audio_array, np.ndarray):
                 raise TypeError("MoviePy did not return a numpy array.")

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1) # 轉為單聲道
            
        sample_rate = 16000
        filtered_segments = []
        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            
            # 確保切片範圍有效
            if start_sample >= end_sample or start_sample >= len(audio_array):
                continue
            end_sample = min(end_sample, len(audio_array))

            segment_audio = audio_array[start_sample:end_sample]
            
            if segment_audio.size == 0:
                continue

            rms_energy = np.sqrt(np.mean(np.square(segment_audio.astype(np.float32))))
            
            if rms_energy >= energy_threshold:
                filtered_segments.append(seg)
            else:
                print(f"[能量過濾] 捨棄低能量片段: \"{seg['text']}\" (能量: {rms_energy:.4f})")
        
        print(f"[資訊] 能量分析完成，原始 {len(segments)} 段，過濾後剩下 {len(filtered_segments)} 段。")
        return filtered_segments
    except Exception as e:
        print(f"[警告] 音訊能量分析失敗: {e}。將跳過此過濾步驟。")
        return segments


def run_faster_whisper(args, config, device, compute_type):
    """
    執行 faster-whisper 流程
    """
    print(f"[1/3] 載入精準模式模型並開始處理: {args.input}")
    model_name = config["whisperx_model"]["model_name"]
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments_iterator, info = model.transcribe(args.input, 
                                             language=args.lang,
                                             beam_size=5,
                                             vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=500, threshold=0.6))
    
    detected_lang = info.language
    print(f"[資訊] 偵測到語言: {detected_lang}")

    initial_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_iterator]
    if not initial_segments:
        raise ValueError("模型未能辨識出任何語音片段。")

    final_segments = filter_segments_by_energy(initial_segments, args.input)
    if not final_segments:
        raise ValueError("所有語音片段因能量過低被過濾，可能為靜音或純噪音。")

    return final_segments, detected_lang


def run_whisper_cpp(args, config):
    """
    執行 whisper.cpp 流程
    """
    print(f"[1/3] 執行快速模式 (whisper.cpp)...")
    cli_path = config["whisper_cpp"]["cli_path"]
    model_path = config["whisper_cpp"]["model_path"]
    
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"cli_temp_audio_{int(time.time())}.wav")
    print(f"建立臨時音訊檔於: {temp_audio_path}")
    
    with AudioFileClip(args.input) as audio_clip:
        audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
    
    raw_srt_path = temp_audio_path + ".srt"
    
    command = [
        cli_path,
        "-m", model_path,
        "-f", temp_audio_path,
        "-osrt",
        "-l", args.lang or "auto",
        "-t", "8", # 線程數
        "--vad-filter", "true",
        "--vad-threshold", "0.5",  # 降低 VAD 閾值以捕捉更多語音
        "--max-len", "20",  # 增加最大長度以容納更長的句子
        "--word-threshold", "0.4",  # 降低詞彙閾值以包含更多詞彙
        "--entropy-threshold", "2.4",  # 熵閾值
        "--logprob-threshold", "-1.0",  # 對數概率閾值
        "--no-timestamps", "false",  # 確保生成時間戳
        "--beam-size", "5",  # 使用 beam search
        "--best-of", "5",  # 最佳候選數
        "--temperature", "0.0"  # 降低溫度以獲得更穩定的輸出
    ]
    
    print(f"[資訊] 執行 Whisper.cpp 指令: {' '.join(command)}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', errors='ignore', bufsize=1, env=os.environ
    )
    
    while True:
        output_line = process.stdout.readline()
        if not output_line and process.poll() is not None:
            break
        if output_line:
            print(f"[whisper.cpp]: {output_line.strip()}")
            
    return_code = process.wait()
    if return_code != 0:
        stderr_output = process.stderr.read()
        raise subprocess.CalledProcessError(return_code, command, stderr=stderr_output)

    if not os.path.exists(raw_srt_path):
        raise FileNotFoundError("Whisper.cpp 未能成功生成 SRT 檔案。")

    subs = pysrt.open(raw_srt_path, encoding='utf-8')
    final_segments = [{"start": s.start.seconds + s.start.milliseconds / 1000.0, 
                       "end": s.end.seconds + s.end.milliseconds / 1000.0, 
                       "text": s.text} for s in subs]
    
    os.remove(raw_srt_path)
    os.remove(temp_audio_path)
    
    return final_segments, args.lang

def main():
    parser = argparse.ArgumentParser(
        description="雙核心字幕生成翻譯工具 (命令列版)。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="輸入音訊/影片檔案路徑。")
    parser.add_argument("--output", help="輸出檔案名稱前綴 (不含副檔名)。\n預設使用輸入檔名。")
    parser.add_argument("--mode", default="accurate", choices=["fast", "accurate"],
                        help="選擇處理模式 (預設: accurate)。\n"
                             "'fast': 使用 whisper.cpp。\n"
                             "'accurate': 使用 faster-whisper。")
    parser.add_argument("--lang", help="指定音訊語言 (例如 'ja', 'en')。")
    parser.add_argument("--model", help="指定翻譯模型名稱 (必須與 config.json 中相符)。")
    parser.add_argument("--cpu_compute_type", default="float32", choices=["int8", "float32"], 
                        help="[僅限 accurate 模式] 在 CPU 上執行時的計算類型 (預設: float32)。")
    args = parser.parse_args()

    try:
        config = load_config()
        prompt_config = load_config("prompt_config.json")
        if not prompt_config:
             print("[警告] 未找到 prompt_config.json，將使用簡易翻譯指令。")

        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_dir = os.path.dirname(args.input)
            args.output = os.path.join(output_dir, base_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if args.mode == "accurate":
            compute_type = args.cpu_compute_type if device == "cpu" else config["whisperx_model"]["compute_type"]
            print(f"[資訊] 執行精準模式 (faster-whisper) on {device} with {compute_type}")
            final_segments, detected_lang = run_faster_whisper(args, config, device, compute_type)
        else: # fast mode
            print("[資訊] 執行快速模式 (whisper.cpp)")
            final_segments, detected_lang = run_whisper_cpp(args, config)

        print("[2/3] 開始翻譯與輸出...")
        translation_model_name = args.model
        if not translation_model_name:
            lang_key = detected_lang if detected_lang and detected_lang in config["translation_models"] else "mixed"
            model_list = config["translation_models"][lang_key]
            translation_model_name = model_list[0]["name"]
        print(f"[資訊] 使用翻譯模型: {translation_model_name}")

        bilingual_content = generate_bilingual_srt_content(final_segments, config, translation_model_name, detected_lang, prompt_config)
        
        output_path = f"{args.output}_{args.mode}.srt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(bilingual_content)

        print(f"\n[3/3] [完成] 字幕已儲存至: {output_path}")

    except Exception as e:
        print(f"\n[嚴重錯誤] 處理過程中發生錯誤: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        clear_caches()

if __name__ == "__main__":
    main()
