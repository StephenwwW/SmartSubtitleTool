import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file_info):
    """
    處理單一檔案的函式，呼叫 cli_tool.py。
    """
    i, total_files, fname, folder, no_diarize_flag = file_info
    input_path = os.path.join(folder, fname)
    
    print(f"[{i}/{total_files}] 開始處理: {fname}")

    cmd = ["python", "cli_tool.py", "--input", input_path]
    if no_diarize_flag:
        cmd.append("--no_diarize")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=7200)
        return (True, fname, None)
    except subprocess.CalledProcessError as e:
        error_message = f"處理失敗，錯誤碼 {e.returncode}。\n錯誤輸出:\n{e.stderr}"
        return (False, fname, error_message)
    except Exception as e:
        error_message = f"執行時發生未知錯誤: {e}"
        return (False, fname, error_message)

def main():
    parser = argparse.ArgumentParser(
        description="批次處理資料夾內所有媒體檔案。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--folder", required=True, help="要處理的資料夾路徑。")
    parser.add_argument("--no_diarize", action="store_true", help="關閉語者分離功能。")
    parser.add_argument("--parallel", type=int, default=1, help="平行處理的任務數量 (預設: 1)。")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[錯誤] 資料夾不存在: {args.folder}", file=sys.stderr)
        return

    SUPPORTED_EXTS = [".mp3", ".wav", ".mp4", ".m4a", ".webm", ".mkv", ".avi", ".mov"]
    files_to_process = sorted([f for f in os.listdir(args.folder) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])

    if not files_to_process:
        print("[資訊] 資料夾中沒有找到可處理的媒體檔案。")
        return

    total_files = len(files_to_process)
    print(f"[資訊] 發現 {total_files} 個檔案，準備開始處理...")
    
    start_time = time.time()
    successful_files = []
    failed_files = []

    tasks = [(i + 1, total_files, fname, args.folder, args.no_diarize) for i, fname in enumerate(files_to_process)]

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            success, fname, error_msg = future.result()
            if success:
                print(f"[成功] 已完成處理: {fname}")
                successful_files.append(fname)
            else:
                print(f"[失敗] 處理檔案時發生錯誤: {fname}", file=sys.stderr)
                print(f"  └─ 原因: {error_msg}", file=sys.stderr)
                failed_files.append((fname, error_msg))

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("批次處理完成報告")
    print(f"總耗時: {total_time:.2f} 秒")
    print(f"成功: {len(successful_files)} / 失敗: {len(failed_files)}")
    if failed_files:
        print("\n--- 失敗檔案列表 ---")
        for fname, error_msg in failed_files:
            print(f"- {fname}")
    print("="*50)

if __name__ == "__main__":
    main()
