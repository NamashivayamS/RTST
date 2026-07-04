import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import re
import sentencepiece as spm
from huggingface_hub import snapshot_download
import ctranslate2
from IndicTransToolkit.processor import IndicProcessor

MODEL_REGISTRY = {
    "eng_Latn→tam_Taml": "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
    "eng_Latn→hin_Deva": "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
    "eng_Latn→tel_Telu": "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
    "eng_Latn→kan_Knda": "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
    "eng_Latn→mal_Mlym": "adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M",
    "tam_Taml→eng_Latn": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
    "hin_Deva→eng_Latn": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
    "tel_Telu→eng_Latn": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
    "kan_Knda→eng_Latn": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
    "mal_Mlym→eng_Latn": "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M",
}

LANG_CODE_MAP = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
}

DEFAULT_TARGET_MAP = {
    "eng_Latn": "tam_Taml",
    "tam_Taml": "eng_Latn",
    "hin_Deva": "eng_Latn",
    "tel_Telu": "eng_Latn",
    "kan_Knda": "eng_Latn",
    "mal_Mlym": "eng_Latn",
}

def _sanitize_translation(text: str) -> str:
    """
    Strips IndicProcessor boundary artifacts that leak into translation output.
    
    Known artifacts:
      - Leading 'ൾ' (Malayalam closing bracket, U+0D3E vicinity, used as sentence marker)
      - Leading/trailing periods from sentence boundary tokens
      - Multiple spaces
    """
    # Strip leading Malayalam bracket artifact (U+0D3E and nearby range)
    text = text.lstrip('ൾ').lstrip()
    
    # Strip leading punctuation artifacts (rogue period at start)
    text = re.sub(r'^\s*[\.।]\s*', '', text)
    
    # Strip Tamil boundary artifacts ('க்கு' and 'வங்கம்')
    text = re.sub(r'^க்கு\s*', '', text)
    text = re.sub(r'^வங்கம்\s*,?\s*', '', text)
    
    # Strip leading hyphen-word fragments (e.g. "-டைம்" from "real-time" boundary cuts)
    text = re.sub(r'^-\S+\s*', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()

def _is_devanagari_script(text: str) -> bool:
    devanagari = sum(1 for ch in text if '\u0900' <= ch <= '\u097F')
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if alpha_chars == 0:
        return False
    return (devanagari / alpha_chars) > 0.3

def _is_malayalam_script(text: str) -> bool:
    malayalam = sum(1 for ch in text if '\u0D00' <= ch <= '\u0D7F')
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if alpha_chars == 0:
        return False
    return (malayalam / alpha_chars) > 0.3

def _is_non_tamil(text: str) -> bool:
    return _is_devanagari_script(text) or _is_malayalam_script(text)



class TranslationService:
    """
    Bidirectional IndicTrans2 translation service.
    Lazily loads each direction's model on first use to avoid holding
    both 200M models in VRAM simultaneously.
    """

    def __init__(self):
        self._models: dict = {}
        self._sp_src: dict = {}
        self._sp_tgt: dict = {}
        self.ip = IndicProcessor(inference=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("TranslationService initialized. Models will load on first use.")

    def _pair_key(self, src_lang: str, tgt_lang: str) -> str:
        return f"{src_lang}→{tgt_lang}"

    def _load_model_if_needed(self, src_lang: str, tgt_lang: str):
        key = self._pair_key(src_lang, tgt_lang)
        model_name = MODEL_REGISTRY.get(key)
        
        if model_name is None:
            raise ValueError(
                f"No model registered for '{key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
            
        if model_name in self._models:
            return

        print(f"TranslationService: Downloading/Loading model '{model_name}'...")
        
        # 1. Download/Locate the model snapshot from HuggingFace
        try:
            model_path = snapshot_download(model_name, local_files_only=True)
        except Exception:
            model_path = snapshot_download(model_name)
        
        # 2. Locate the specific directory containing 'model.bin'
        if "en-indic" in model_name:
            ct2_model_dir = os.path.join(model_path, "en-indic-200m-ct2", "ctranslate2_model")
        else:
            ct2_model_dir = os.path.join(model_path, "indic-en-200m-ct2", "ctranslate2_model")
            
        # 3. Load the SentencePiece tokenizers from the original AI4Bharat repo
        original_model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if "en-indic" in model_name else "ai4bharat/indictrans2-indic-en-dist-200M"
        try:
            ai4_path = snapshot_download(original_model_name, allow_patterns=["*model.SRC", "*model.TGT"], local_files_only=True)
        except Exception:
            ai4_path = snapshot_download(original_model_name, allow_patterns=["*model.SRC", "*model.TGT"])
        sp_src = spm.SentencePieceProcessor(model_file=os.path.join(ai4_path, "model.SRC"))
        sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(ai4_path, "model.TGT"))
        
        # 4. Load the C++ CTranslate2 Engine with int8 quantization
        compute_type = "int8_float16" if self.device == "cuda" else "int8"
        model = ctranslate2.Translator(
            ct2_model_dir, 
            device=self.device, 
            compute_type=compute_type,
            inter_threads=1,    # Single request at a time (serialized by lock)
            intra_threads=4
        )

        self._sp_src[model_name] = sp_src
        self._sp_tgt[model_name] = sp_tgt
        self._models[model_name] = model
        print(f"TranslationService: CTranslate2 Engine '{model_name}' loaded successfully.")

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str | None = None,
        max_new_tokens: int = 96
    ) -> str:
        if src_lang in LANG_CODE_MAP:
            src_lang = LANG_CODE_MAP[src_lang]
        if tgt_lang is None:
            tgt_lang = DEFAULT_TARGET_MAP.get(src_lang)
            if tgt_lang is None:
                raise ValueError(f"Cannot determine target for src_lang='{src_lang}'.")
        elif tgt_lang in LANG_CODE_MAP:
            tgt_lang = LANG_CODE_MAP[tgt_lang]

        return self.translate_batch([text], src_lang, tgt_lang, max_new_tokens)[0]

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 96
    ) -> list[str]:
        if not texts:
            return []

        texts = [t for t in texts if t.strip()]
        if not texts:
            return []

        original_texts = texts[:]  # keep originals before IndicProcessor modifies them

        self._load_model_if_needed(src_lang, tgt_lang)

        key = self._pair_key(src_lang, tgt_lang)
        model_name = MODEL_REGISTRY.get(key)
        sp_src = self._sp_src[model_name]
        sp_tgt = self._sp_tgt[model_name]
        model = self._models[model_name]

        processed = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)

        import time
        _t0 = time.perf_counter()
        
        # Convert text to SentencePiece string tokens for CTranslate2
        # IndicProcessor wraps text like: 'eng_Latn tam_Taml Hello'
        # We must split the tags, encode the text, and prepend the tags as list elements
        source_tokens_batch = []
        for p_text in processed:
            parts = p_text.split(" ", 2)
            if len(parts) >= 3:
                s_lang, t_lang, core_text = parts[0], parts[1], parts[2]
            else:
                s_lang, t_lang, core_text = src_lang, tgt_lang, p_text
            
            sp_tokens = sp_src.encode(core_text, out_type=str)
            source_tokens_batch.append([s_lang, t_lang] + sp_tokens)
            
        # Generate using CTranslate2 C++ Engine
        results = model.translate_batch(
            source_tokens_batch,
            batch_type="tokens",
            max_decoding_length=max_new_tokens,
            beam_size=1, 
            no_repeat_ngram_size=3,
            repetition_penalty=1.15
        )
        
        _t1 = time.perf_counter()
        
        decoded = []
        for res in results:
            target_tokens = res.hypotheses[0]
            # Decode using target SentencePiece model
            decoded_text = sp_tgt.decode(target_tokens)
            decoded.append(decoded_text)
            
        # Telemetry for token generation
        input_tokens = len(source_tokens_batch[0]) if source_tokens_batch else 0
        print(
            f"[TRANSLATE-CT2] {src_lang}->{tgt_lang} "
            f"input_tokens={input_tokens} "
            f"time={(_t1-_t0):.3f}s"
        )

        results = self.ip.postprocess_batch(decoded, lang=tgt_lang)

        for i, (src, res) in enumerate(zip(texts, results)):
            # Warn if output is suspiciously short relative to input
            if len(res.split()) < len(src.split()) * 0.3 and len(src.split()) > 4:
                print(
                    f"[Translation] WARNING: Short output ratio "
                    f"({len(res.split())} words from {len(src.split())} word input) "
                    f"— possible truncation. Input: '{src[:60]}'"
                )

        # Sanitize IndicProcessor boundary artifacts from all results
        results = [_sanitize_translation(r) for r in results]

        # ── Script drift guard ────────────────────────────────────────────────
        if tgt_lang == "tam_Taml":
            validated = []
            for src_text, result in zip(original_texts, results):
                if _is_non_tamil(result):
                    print(
                        f"[Translation] Script drift: expected Tamil script.\n"
                        f"  Input : '{src_text}'\n"
                        f"  Output: '{result}'"
                    )
                    validated.append(src_text)  # fallback to English source
                else:
                    validated.append(result)
            return validated

        return results

    def translate_auto(self, text: str, detected_language: str, target_language: str | None = None) -> dict:
        """
        Main pipeline entry point. Accepts Whisper ISO 639-1 code,
        routes to correct direction automatically.
        """
        src_lang = LANG_CODE_MAP.get(detected_language)
        if src_lang is None:
            raise ValueError(
                f"Unsupported language='{detected_language}'. "
                f"Supported: {list(LANG_CODE_MAP.keys())}"
            )
            
        if detected_language == target_language:
            tgt_lang = src_lang
        elif detected_language == "en":
            # If English is spoken, translate TO the target language from the UI dropdown
            tgt_lang = LANG_CODE_MAP.get(target_language, "tam_Taml")
        else:
            # If an Indic language is spoken, always translate TO English
            tgt_lang = "eng_Latn"
            
        if src_lang == tgt_lang:
            return {
                "translated_text": text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            
        translated = self.translate(text, src_lang, tgt_lang)
        return {
            "translated_text": translated,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }


if __name__ == "__main__":
    service = TranslationService()

    en_text = "Hello friends, the meeting will start tomorrow morning."
    result = service.translate_auto(en_text, detected_language="en")
    print(f"EN→TA: '{en_text}'")
    print(f"     → '{result['translated_text']}'\n")

    ta_text = "காலை வணக்கம் அனைவருக்கும்."
    result = service.translate_auto(ta_text, detected_language="ta")
    print(f"TA→EN: '{ta_text}'")
    print(f"     → '{result['translated_text']}'")
