import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import time
import subprocess
import logging
import json
import queue
import gc
import tempfile
import numpy as np
import re

# --- 依賴套件檢查函式 ---
def check_dependencies():
    """檢查所有必要的套件。"""
    errors = []
    try:
        import torch
    except ImportError as e:
        errors.append(f"- PyTorch 無法載入: {e}")
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
    except ImportError as e:
        errors.append(f"- MoviePy 無法載入: {e}")
    try:
        from translation_module import generate_bilingual_srt_content
    except ImportError as e:
        errors.append(f"- Translation Module 無法載入: {e}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        errors.append(f"- Faster-Whisper 無法載入: {e}")
    try:
        import pysrt
    except ImportError as e:
        errors.append(f"- PySRT 無法載入: {e}")
    return errors

# --- 全域設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 輔助函式 ---
def load_config(path="config.json"):
    if not os.path.exists(path):
        return None, f"設定檔 '{path}' 不存在。"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"讀取設定檔 '{path}' 失敗: {e}"

def time_to_seconds(time_str):
    try:
        h, m, s_float = time_str.split(':')
        s, ms = map(int, s_float.replace(',', '.').split('.'))
        return int(h) * 3600 + int(m) * 60 + s + ms / 1000.0
    except:
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except ValueError:
            return 0

def seconds_to_time(seconds):
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02}"

def filter_segments_by_energy(segments, audio_path, energy_threshold=0.005):
    """
    [v3.0 修正] 透過更穩健的方式讀取音訊，避免 moviepy 產生非預期資料結構。
    """
    try:
        from moviepy.editor import AudioFileClip
        logging.info(f"[資訊] 正在進行音訊能量分析以過濾幻聽...")
        
        with AudioFileClip(audio_path) as clip:
            # 確保音訊以 16000Hz 單聲道讀取
            audio_array = clip.to_soundarray(fps=16000, nbytes=2)
            if not isinstance(audio_array, np.ndarray):
                 raise TypeError("MoviePy did not return a numpy array.")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1) # 轉為單聲道
            
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
                logging.info(f"[能量過濾] 捨棄低能量片段: \"{seg['text']}\" (能量: {rms_energy:.4f})")
        
        logging.info(f"[資訊] 能量分析完成，原始 {len(segments)} 段，過濾後剩下 {len(filtered_segments)} 段。")
        return filtered_segments
    except Exception as e:
        logging.warning(f"[警告] 音訊能量分析失敗: {e}。將跳過此過濾步驟。")
        return segments

# --- 主應用程式類別 ---
class SubtitleApp:
    def __init__(self, root, dependency_errors):
        self.root = root
        self.root.title("智慧字幕工具 v3.0 (穩定版)")
        self.root.geometry("520x650")

        self.config, config_error = load_config()
        if config_error:
            messagebox.showerror("設定錯誤", config_error)
            self.root.destroy(); return
        
        self.prompt_config, prompt_error = load_config("prompt_config.json")
        if prompt_error:
            messagebox.showwarning("Prompt 設定警告", f"{prompt_error}\n將使用預設的簡易翻譯指令。")
            
        self.video_path = None
        self.is_processing = False
        self.gui_queue = queue.Queue()

        self.mode_var = tk.StringVar(value="accurate")
        self.use_time_range = tk.BooleanVar(value=False)
        self.start_time_var = tk.StringVar(value="00:00:00")
        self.end_time_var = tk.StringVar(value="00:00:00")
        self.translation_model_var = tk.StringVar()
        self.cpu_precision_var = tk.StringVar(value="float32")
        self.detected_lang = None

        self.create_widgets()
        self.process_queue()
        
        self.dependencies_ok = not dependency_errors
        if not self.dependencies_ok:
            self.mode_radio_accurate.config(state=tk.DISABLED)
            self.mode_radio_fast.config(state=tk.DISABLED)
            messagebox.showwarning("缺少套件", f"一個或多個核心套件無法載入，功能將受限。\n\n詳細資訊:\n" + "\n".join(dependency_errors))

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=5, fill="x")
        self.status_label = tk.Label(main_frame, text="請選擇媒體檔案...", font=("Arial", 12))
        self.status_label.pack(pady=5)
        self.btn_select_video = tk.Button(main_frame, text="選擇影片或音訊", command=self.select_video)
        self.btn_select_video.pack(pady=5)

        mode_frame = tk.LabelFrame(main_frame, text="處理模式", padx=10, pady=5)
        mode_frame.pack(pady=10, fill="x")
        self.mode_radio_fast = tk.Radiobutton(mode_frame, text="🚀 快速模式 (whisper.cpp)", variable=self.mode_var, value="fast")
        self.mode_radio_fast.pack(side="left", padx=5)
        self.mode_radio_accurate = tk.Radiobutton(mode_frame, text="🎯 精準模式 (faster-whisper)", variable=self.mode_var, value="accurate")
        self.mode_radio_accurate.pack(side="left", padx=5)

        time_frame = tk.LabelFrame(main_frame, text="時間範圍", padx=10, pady=10)
        time_frame.pack(pady=10, fill="x")
        time_frame.columnconfigure(1, weight=1)
        time_frame.columnconfigure(3, weight=1)
        
        self.use_time_range_cb = tk.Checkbutton(time_frame, text="使用時間範圍", variable=self.use_time_range)
        self.use_time_range_cb.grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(time_frame, text="開始:").grid(row=1, column=0, sticky="w", padx=5)
        self.start_time_entry = tk.Entry(time_frame, textvariable=self.start_time_var, width=10)
        self.start_time_entry.grid(row=1, column=1, sticky="w")
        tk.Label(time_frame, text="結束:").grid(row=1, column=2, sticky="w", padx=5)
        self.end_time_entry = tk.Entry(time_frame, textvariable=self.end_time_var, width=10)
        self.end_time_entry.grid(row=1, column=3, sticky="w")
        self.video_duration_label = tk.Label(time_frame, text="總時長: N/A")
        self.video_duration_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(5,0))

        options_frame = tk.LabelFrame(main_frame, text="進階選項 (僅限精準模式)", padx=10, pady=10)
        options_frame.pack(pady=10, fill="x")
        
        cpu_precision_frame = tk.Frame(options_frame)
        cpu_precision_frame.pack(anchor="w", pady=2)
        tk.Label(cpu_precision_frame, text="CPU 運算精度:").pack(side="left")
        self.precision_radio_fast = tk.Radiobutton(cpu_precision_frame, text="高效能 (int8)", variable=self.cpu_precision_var, value="int8")
        self.precision_radio_fast.pack(side="left", padx=5)
        self.precision_radio_accurate = tk.Radiobutton(cpu_precision_frame, text="最高精度 (float32)", variable=self.cpu_precision_var, value="float32")
        self.precision_radio_accurate.pack(side="left", padx=5)

        translation_frame = tk.LabelFrame(main_frame, text="翻譯模型選擇", padx=10, pady=10)
        translation_frame.pack(pady=10, fill="x")
        tk.Label(translation_frame, text="使用模型:").pack(side="left", padx=5)
        self.translation_model_menu = ttk.Combobox(translation_frame, textvariable=self.translation_model_var, state="readonly")
        self.translation_model_menu.pack(fill="x", expand=True)
        self.update_translation_model_menu()

        self.btn_transcribe = tk.Button(main_frame, text="開始生成字幕", command=self.start_processing_thread, font=("Arial", 12, "bold"), height=2, width=15, state=tk.DISABLED)
        self.btn_transcribe.pack(pady=10)
        
    def update_translation_model_menu(self, lang_code=None):
        self.detected_lang = lang_code
        lang_key = lang_code if lang_code and lang_code in self.config.get("translation_models", {}) else "mixed"
        models = self.config.get("translation_models", {}).get(lang_key, [])
        model_names = [m.get("name") for m in models if m.get("name")]
        self.translation_model_menu['values'] = model_names
        self.translation_model_var.set(model_names[0] if model_names else "")

    def process_queue(self):
        try:
            while True:
                msg_type, value = self.gui_queue.get_nowait()
                if msg_type == "status": self.status_label.config(text=value)
                elif msg_type == "progress": self.progress_var.set(value)
                elif msg_type == "processing_done": self.toggle_controls(True)
                elif msg_type == "set_duration":
                    self.video_duration_label.config(text=f"總時長: {value}")
                    self.end_time_var.set(value)
                elif msg_type == "language_detected":
                    self.update_translation_model_menu(value)
                    status_msg = f"偵測到語言: {value}。請選擇翻譯模型。" if value else "語言偵測失敗，將使用預設模型。"
                    self.update_status(status_msg)
                    self.btn_transcribe.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def toggle_controls(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.is_processing = not enabled
        for widget in [self.btn_select_video, self.mode_radio_fast, self.mode_radio_accurate,
                       self.use_time_range_cb, self.start_time_entry, self.end_time_entry,
                       self.translation_model_menu, self.precision_radio_fast, self.precision_radio_accurate]:
            widget.config(state=state)
        self.btn_transcribe.config(state=tk.NORMAL if enabled and self.video_path else tk.DISABLED)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("媒體檔案", "*.mp4;*.mkv;*.mov;*.avi;*.mp3;*.wav"), ("所有檔案", "*.*")])
        if file_path:
            self.video_path = file_path
            self.update_status(f"已選擇: {os.path.basename(self.video_path)}")
            self.btn_transcribe.config(state=tk.DISABLED)
            threading.Thread(target=self.get_video_info_worker, args=(file_path,), daemon=True).start()

    def get_video_info_worker(self, file_path):
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(file_path) as clip:
                duration = clip.duration
            duration_str = seconds_to_time(duration)
            self.gui_queue.put(("set_duration", duration_str))
            self.update_status("正在偵測語言，請稍候...")
            self.detect_language_worker(file_path)
        except Exception as e:
            logging.error(f"讀取影片資訊失敗: {e}")
            self.update_status("無法讀取影片資訊")
            self.gui_queue.put(("set_duration", "N/A"))
            self.gui_queue.put(("language_detected", None))

    def detect_language_worker(self, file_path):
        temp_audio_path = None
        logging.info("--- 開始語言偵測 ---")
        try:
            from moviepy.editor import VideoFileClip
            from faster_whisper import WhisperModel
            temp_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_dir, f"lang_detect_temp_{int(time.time())}.wav")
            with VideoFileClip(file_path) as clip:
                clip.subclip(0, min(clip.duration, 30)).audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
            model = WhisperModel("base", device="cpu", compute_type="int8")
            _, info = model.transcribe(temp_audio_path, beam_size=5)
            self.gui_queue.put(("language_detected", info.language))
        except Exception as e:
            logging.error(f"語言偵測失敗: {e}")
            self.gui_queue.put(("language_detected", None))
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            logging.info("--- 語言偵測結束 ---")
    
    def update_status(self, text):
        self.gui_queue.put(("status", text))

    def update_progress(self, value):
        self.gui_queue.put(("progress", value))

    def start_processing_thread(self):
        if not self.video_path:
            messagebox.showerror("錯誤", "請先選擇一個媒體檔案。")
            return
        self.toggle_controls(False)
        worker = self.fast_mode_worker if self.mode_var.get() == "fast" else self.accurate_mode_worker
        threading.Thread(target=worker, daemon=True).start()
    
    def build_output_path(self):
        video_dir = os.path.dirname(self.video_path)
        base_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        mode_tag = self.mode_var.get()
        # 取模型名稱作為檔名標籤，移除不合法字元並截短
        model_tag_raw = self.translation_model_var.get() or "model"
        model_tag = re.sub(r"[^A-Za-z0-9\-_]", "", model_tag_raw)[:40]
        return os.path.join(video_dir, f"{base_filename}_{mode_tag}_{model_tag}.srt")

    def fast_mode_worker(self):
        logging.info("--- 開始快速模式處理 (幻聽修正版) ---")
        temp_audio_path = None
        try:
            cli_path = self.config["whisper_cpp"]["cli_path"]
            model_path = self.config["whisper_cpp"]["model_path"]
            if not all(map(os.path.exists, [cli_path, model_path])):
                raise FileNotFoundError(f"whisper-cli 或模型路徑不存在，請檢查 config.json。\nCLI: {cli_path}\nModel: {model_path}")
            
            self.update_status("快速模式：正在提取音訊...")
            from moviepy.editor import AudioFileClip
            import pysrt
            
            temp_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")
            logging.info(f"建立臨時音訊檔於: {temp_audio_path}")
            
            start_s, duration = None, None
            if self.use_time_range.get():
                start_s = time_to_seconds(self.start_time_var.get())
                end_s = time_to_seconds(self.end_time_var.get())
                duration = end_s - start_s
            with AudioFileClip(self.video_path) as clip:
                audio_clip = clip.subclip(start_s, start_s + duration) if start_s is not None else clip.audio
                audio_clip.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
            
            self.update_status("快速模式：正在進行語音辨識...")
            raw_srt_path = temp_audio_path + ".srt" 
            
            command = [
                cli_path, "-m", model_path, "-f", temp_audio_path, "-osrt",
                "-l", self.detected_lang or "auto", "-t", "8",
                "--vad-filter", "true", "--vad-threshold", "0.5",
                "--max-len", "20", "--word-threshold", "0.4",
                "--entropy-threshold", "2.4", "--logprob-threshold", "-1.0",
                "--no-timestamps", "false", "--beam-size", "5",
                "--best-of", "5", "--temperature", "0.0"
            ]
            logging.info(f"執行命令: {' '.join(command)}")

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, 
                encoding='utf-8', errors='ignore', bufsize=1, env=os.environ
            )
            
            while True:
                output_line = process.stdout.readline()
                if not output_line and process.poll() is not None: break
                if output_line: logging.info(f"[whisper.cpp]: {output_line.strip()}")
            
            return_code = process.wait()
            if return_code != 0:
                stderr_output = process.stderr.read()
                raise subprocess.CalledProcessError(return_code, command, stderr=stderr_output)

            if not os.path.exists(raw_srt_path):
                raise FileNotFoundError(f"Whisper.cpp 未能成功生成 SRT 檔案 ({raw_srt_path})。")

            subs = pysrt.open(raw_srt_path, encoding='utf-8')
            segments = [{"start": s.start.seconds + s.start.milliseconds / 1000.0, 
                         "end": s.end.seconds + s.end.milliseconds / 1000.0, 
                         "text": s.text} for s in subs]
            
            segments = filter_segments_by_energy(segments, temp_audio_path)
            self.update_status("快速模式：正在翻譯字幕...")
            from translation_module import generate_bilingual_srt_content, clear_caches
            bilingual_content = generate_bilingual_srt_content(segments, self.config, self.translation_model_var.get(), self.detected_lang, self.prompt_config)
            
            output_path = self.build_output_path()
            with open(output_path, "w", encoding="utf-8") as f: f.write(bilingual_content)
            
            self.update_progress(100)
            self.update_status("快速模式處理完成！")
            messagebox.showinfo("完成", f"字幕已成功生成！\n路徑:\n{output_path}")
        except subprocess.CalledProcessError as e:
            logging.exception("快速模式處理失敗:")
            error_message = f"快速模式處理失敗，whisper-cli.exe 傳回錯誤。\n\n錯誤碼: {e.returncode}\n\n詳細資訊:\n{e.stderr}"
            self.update_status(f"錯誤: whisper-cli.exe 執行失敗")
            messagebox.showerror("錯誤", error_message)
        except Exception as e:
            logging.exception("快速模式處理失敗:")
            self.update_status(f"錯誤: {e}")
            messagebox.showerror("錯誤", f"快速模式處理失敗:\n{e}")
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            raw_srt_path = (temp_audio_path or "") + ".srt"
            if os.path.exists(raw_srt_path): os.remove(raw_srt_path)
            if 'clear_caches' in locals(): clear_caches()
            self.gui_queue.put(("processing_done", None))

    def accurate_mode_worker(self):
        logging.info("--- 開始精準模式處理 (幻聽修正版) ---")
        temp_audio_path = None
        model = None
        try:
            import torch
            from moviepy.editor import VideoFileClip, AudioFileClip
            from faster_whisper import WhisperModel
            from translation_module import generate_bilingual_srt_content, clear_caches

            self.update_status("準備環境...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            compute_type = self.cpu_precision_var.get() if device == "cpu" else self.config["whisperx_model"]["compute_type"]
            model_size = self.config["whisperx_model"]["model_name"]
            logging.info(f"環境準備完成 (模型: {model_size}, 裝置: {device}, 計算類型: {compute_type})。")

            self.update_status(f"載入音訊...")
            audio_path_for_model = self.video_path
            if self.use_time_range.get():
                start_s = time_to_seconds(self.start_time_var.get())
                end_s = time_to_seconds(self.end_time_var.get())
                temp_dir = tempfile.gettempdir()
                temp_audio_path = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.wav")
                with AudioFileClip(self.video_path) as clip:
                    clip.subclip(start_s, end_s).audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
                audio_path_for_model = temp_audio_path

            self.update_status(f"辨識與分段中 (模型: {model_size})...")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            segments_iterator, _ = model.transcribe(audio_path_for_model, 
                                                    language=self.detected_lang,
                                                    beam_size=5,
                                                    vad_filter=True,
                                                    vad_parameters=dict(min_silence_duration_ms=500, threshold=0.6))

            initial_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_iterator]
            if not initial_segments: raise ValueError("模型未能辨識出任何語音片段。")
            
            self.update_status("正在進行能量分析以過濾幻聽...")
            final_segments = filter_segments_by_energy(initial_segments, audio_path_for_model)
            if not final_segments:
                raise ValueError("所有語音片段因能量過低被過濾，可能為靜音或純噪音。")

            self.update_status("正在翻譯字幕...")
            bilingual_content = generate_bilingual_srt_content(final_segments, self.config, self.translation_model_var.get(), self.detected_lang, self.prompt_config)
            
            output_path = self.build_output_path()
            with open(output_path, "w", encoding="utf-8") as f: f.write(bilingual_content)
            
            self.update_progress(100)
            self.update_status("精準模式處理完成！")
            messagebox.showinfo("完成", f"字幕已成功生成！\n路徑:\n{output_path}")
        except Exception as e:
            logging.exception("精準模式處理失敗:")
            self.update_status(f"錯誤: {e}")
            messagebox.showerror("錯誤", f"精準模式處理失敗:\n{e}")
        finally:
            logging.info("正在清理暫存檔案與記憶體...")
            if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            if 'clear_caches' in locals(): clear_caches()
            if model is not None:
                del model
                gc.collect()
                if 'torch' in locals() and torch.cuda.is_available(): torch.cuda.empty_cache()
            self.gui_queue.put(("processing_done", None))

if __name__ == "__main__":
    print("--- 正在檢查依賴套件 ---")
    dependency_errors = check_dependencies()
    print("--- 檢查報告 ---")
    if not dependency_errors:
        print("所有核心套件皆已成功載入。")
    else:
        for error in dependency_errors:
            print(error)
    print("--------------------")
    root = tk.Tk()
    app = SubtitleApp(root, dependency_errors)
    root.mainloop()
