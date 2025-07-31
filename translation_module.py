import os
import gc
import logging
from llama_cpp import Llama

# --- 全域快取 ---
MODEL_CACHE = {}
TRANSLATION_CACHE = {}

def load_llm_model(model_path, model_name):
    """安全地載入或從快取讀取 LLM 模型。"""
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    try:
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
        MODEL_CACHE[model_path] = llm
        return llm
    except Exception as e:
        raise RuntimeError(f"載入模型 '{model_name}' ({model_path}) 失敗: {e}")

def get_translation_model_info(config, selected_model_name, lang_code):
    """根據語言和使用者選擇，獲取正確的模型資訊。"""
    lang_key = lang_code if lang_code and lang_code in config.get("translation_models", {}) else "mixed"
    model_choices = config.get("translation_models", {}).get(lang_key, [])
    if not model_choices:
        raise ValueError(f"設定檔 config.json 中缺少 '{lang_key}' 語言的模型設定。")

    for model_info in model_choices:
        if model_info.get("name") == selected_model_name:
            return model_info
    logging.warning(f"在 config.json 中找不到名為 '{selected_model_name}' 的模型，將使用 '{lang_key}' 列表中的第一個模型。")
    return model_choices[0]

def build_prompt_for_others(text_to_translate, prompt_id, api_style, prompt_config):
    """為非 Qwen 模型建立 prompt。"""
    templates = prompt_config.get("prompt_templates", {})
    template = templates.get(prompt_id, templates.get('default_fallback'))
    if not template:
        logging.error(f"在 prompt_config.json 中找不到 ID 為 '{prompt_id}' 或 'default_fallback' 的樣板。")
        return [{"role": "user", "content": f"Translate to Traditional Chinese (Taiwan):\n{text_to_translate}"}]

    if api_style == "legacy":
        return template.get("template", "{text}").format(text=text_to_translate)
    else:
        system_prompt = template.get("system", "")
        user_prompt = template.get("user", "{text}").format(text=text_to_translate)
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def _extract_final_translation(raw_output: str) -> str:
    """
    從 LLM 原始輸出中提取最終翻譯句子，過濾掉 Qwen 可能生成的解釋或思考內容。
    策略：
    1. 移除空行並逆序掃描，尋找第一個不含「原文」「翻譯」「答案」「分析」「範例」「請」等關鍵字的行。
    2. 若找不到，退而取最後一行。
    """
    if not isinstance(raw_output, str):
        return raw_output
    lines = [l.strip() for l in raw_output.splitlines() if l.strip()]
    if not lines:
        return raw_output.strip()
    forbidden_keywords = [
        "原文", "翻譯", "答案", "分析", "範例", "請", "需要", "確保", "首先", "总结", 
        "最後", "检查", "希望", "用户", "你", "我", "再想想", "再考虑", "要注意", 
        "另外", "接下来", "例如", "这样", "这里", "符合", "习惯", "表达", "语气", 
        "避免", "问题", "处理", "可能", "显得", "组合", "检查", "考虑", "注意", "今後", 
        "とても", "本当に", "思考", "理解", "根據", "建議", "因此", "所以", "然而",
        "不過", "但是", "然後", "接著", "如下", "如上", "以下", "以上", "結果",
        "輸出", "回答", "提供", "給出", "生成", "產生", "進行", "執行", "完成",
        "開始", "結束", "正在", "已經", "將會", "應該", "可以", "必須", "需要"
    ]
    for ln in reversed(lines):
        if not ln or all(ch in "。.!?,，,、" for ch in ln):
            continue
        if any(k in ln for k in forbidden_keywords):
            continue
        return ln
    candidate = lines[-1]
    return candidate

def _clean_translation(text: str) -> str:
    """移除重複詞語與雜訊符號，讓輸出更乾淨"""
    import re
    if not isinstance(text, str):
        return text
    # 1) 移除開頭/結尾多餘空白與符號
    cleaned = text.strip(" \t\n\r-—:：")
    # 2) 移除重複 token (>3 次)
    tokens = cleaned.split()
    result = []
    i = 0
    while i < len(tokens):
        result.append(tokens[i])
        repeat_count = 1
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            repeat_count += 1; j += 1
        if repeat_count >= 3:
            # 跳過重複
            i = j
        else:
            # 保留原數量
            result.extend(tokens[i+1:j])
            i = j
    cleaned = " ".join(result)
    # 3) 假如還有連續相同句子以逗號或頓號分隔，再次合併
    cleaned = re.sub(r'(\S.{0,20}?)(?:[、,，]?\s*\1){2,}', r'\1', cleaned)
    return cleaned

def translate_text(text, config, selected_model_name, main_lang_code, prompt_config):
    """
    [v3.9 最終修正版]
    - 根據使用者提供的精準分析，為 Qwen 模型建立專用處理路徑。
    - 強制將 Qwen 的 prompt 展平成單一純文字，並僅使用 llm(prompt=...) 模式呼叫。
    - 此方法可完全避免 GGUF 的 chat 模式相容性問題。
    """
    cache_key = (selected_model_name, text)
    if cache_key in TRANSLATION_CACHE:
        return TRANSLATION_CACHE[cache_key]

    try:
        model_info = get_translation_model_info(config, selected_model_name, main_lang_code)
        model_path = model_info.get("path")
        model_name = model_info.get("name")
        api_style = model_info.get("api_style", "chat")
        prompt_id = model_info.get("prompt_id", "default_fallback")
        
        llm = load_llm_model(model_path, model_name)
        translated_text = ""

        # --- [核心修正] Qwen 模型專用處理路徑 ---
        if "qwen" in model_name.lower():
            # logging.info(f"偵測到 Qwen 模型，強制啟用純文字 Prompt 模式。")
            
            # 根據語言動態選擇 Prompt 樣板
            prompt_key = "pro_qwen_v2_ja" if main_lang_code == "ja" else "pro_qwen_v2_en"
            if main_lang_code not in ["ja", "en"]:
                prompt_key = "pro_universal_v1" # 備用
            
            template = prompt_config["prompt_templates"].get(prompt_key, {})
            system_prompt = template.get("system_template", "")
            user_prompt = template.get("user_template", "{text}").format(text=text)
            
            # 將 system 和 user prompt 展平成單一、乾淨的純文字 prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            
            # logging.info(f">>> Qwen 實際輸出 prompt (純文字) >>>\n{full_prompt}")
            
            stop_tokens = ["<|im_end|>", "翻譯：", "翻譯:", "原文：", "原文:"]
            
            # 直接使用 llm(prompt=...) 模式，不經過 chat completion
            result = llm(prompt=full_prompt, max_tokens=256, temperature=0.1, top_p=0.9, repeat_penalty=1.3, stop=stop_tokens)
            raw_output = result["choices"][0]["text"].strip()
            translated_text = _extract_final_translation(raw_output)

        # --- 其他模型維持原有處理邏輯 ---
        else:
            # logging.info(f"使用標準模式處理模型: {model_name}")
            prompt = build_prompt_for_others(text, prompt_id, api_style, prompt_config)
            
            if api_style == "legacy":
                result = llm(prompt, max_tokens=256, temperature=0.1, top_p=0.9, repeat_penalty=1.3, stop=["\n", "Original:", "Traditional Chinese (Taiwan):", "原文：", "システム:"])
                raw_output = result["choices"][0]["text"].strip()
                translated_text = _extract_final_translation(raw_output)
            else: # 標準 Chat 模型
                try:
                    result = llm.create_chat_completion(messages=prompt, max_tokens=1024, temperature=0.1)
                    raw_output = result['choices'][0]['message']['content'].strip()
                    translated_text = _extract_final_translation(raw_output)
                except Exception as e:
                    logging.warning(f"標準 Chat 模式失敗: {e}，其他模型可能也需要純文字模式。")
                    translated_text = "..." # 避免錯誤

        if not translated_text:
            logging.warning(f"模型 '{model_name}' 為文字 '{text[:20]}...' 返回了空的翻譯結果。")
            translated_text = "..."

        # 最後做清理與快取
        translated_text = _clean_translation(translated_text)
        TRANSLATION_CACHE[cache_key] = translated_text
        return translated_text
    except Exception as e:
        logging.error(f"翻譯文字 '{text[:20]}...' 時發生嚴重錯誤: {e}", exc_info=True)
        return f"{text} (<翻譯錯誤>)"

def clear_caches():
    global MODEL_CACHE, TRANSLATION_CACHE
    logging.info("翻譯模組: 正在清除模型與翻譯快取...")
    MODEL_CACHE.clear(); TRANSLATION_CACHE.clear()
    gc.collect()

def _deduplicate_segments(segments, similarity_threshold=0.9):
    """移除連續重複或高度相似的字幕片段以減少重複翻譯。"""
    from difflib import SequenceMatcher
    deduped = []
    prev_text = ""
    for seg in segments:
        curr_text = seg.get("text", "").strip()
        if not curr_text:
            continue
        if prev_text:
            ratio = SequenceMatcher(None, prev_text, curr_text).ratio()
            if ratio >= similarity_threshold:
                # 視為重複，跳過
                continue
        deduped.append(seg)
        prev_text = curr_text
    return deduped


def generate_bilingual_srt_content(segments, config, selected_model_name, main_lang_code, prompt_config):
    srt_content = []
    segments = _deduplicate_segments(segments)
    total_segments = len(segments)
    for idx, segment in enumerate(segments):
        if (idx + 1) % 50 == 0:  # 改為每 50 句才顯示一次進度
            logging.info(f"翻譯進度: {idx + 1}/{total_segments}")
        # 清理原文內部重複與換行
        raw_original = segment.get("text", "").strip()
        original_lines = [l for l in raw_original.splitlines() if l.strip()]
        # 去除連續相同行
        cleaned_lines = []
        for l in original_lines:
            if not cleaned_lines or cleaned_lines[-1] != l:
                cleaned_lines.append(l)
        original_text = " ".join(cleaned_lines)
        if not original_text: continue
        translated_text = translate_text(original_text, config, selected_model_name, main_lang_code, prompt_config)
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        srt_content.append(f"{idx + 1}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(original_text)
        srt_content.append(f"{translated_text}\n")
    # logging.info("翻譯模組: 所有句子翻譯完成。")
    return "\n".join(srt_content)

def format_time(seconds):
    if not isinstance(seconds, (int, float)): return "00:00:00,000"
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
