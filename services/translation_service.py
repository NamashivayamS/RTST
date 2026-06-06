import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

MODEL_REGISTRY = {
    "eng_Latn→tam_Taml": "ai4bharat/indictrans2-en-indic-dist-200M",
    "eng_Latn→hin_Deva": "ai4bharat/indictrans2-en-indic-dist-200M",
    "eng_Latn→tel_Telu": "ai4bharat/indictrans2-en-indic-dist-200M",
    "eng_Latn→kan_Knda": "ai4bharat/indictrans2-en-indic-dist-200M",
    "eng_Latn→mal_Mlym": "ai4bharat/indictrans2-en-indic-dist-200M",
    "tam_Taml→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
    "hin_Deva→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
    "tel_Telu→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
    "kan_Knda→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
    "mal_Mlym→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
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
}


class TranslationService:
    """
    Bidirectional IndicTrans2 translation service.
    Lazily loads each direction's model on first use to avoid holding
    both 200M models in VRAM simultaneously.
    """

    def __init__(self):
        self._models: dict = {}
        self._tokenizers: dict = {}
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

        print(f"TranslationService: Loading model '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16   # ~400MB VRAM saving vs float32
        ).to(self.device)

        self._tokenizers[model_name] = tokenizer
        self._models[model_name] = model
        print(f"TranslationService: Model '{model_name}' loaded successfully.")

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str | None = None,
        max_new_tokens: int = 256     # FIXED: was max_length (counted input+output)
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
        max_new_tokens: int = 256     # FIXED: was max_length
    ) -> list[str]:
        if not texts:
            return []

        # Guard: empty strings reach the model and cause undefined behaviour
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []

        self._load_model_if_needed(src_lang, tgt_lang)

        key = self._pair_key(src_lang, tgt_lang)
        model_name = MODEL_REGISTRY.get(key)
        tokenizer = self._tokenizers[model_name]
        model = self._models[model_name]

        processed = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            processed,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,   # FIXED: only counts output tokens
                num_beams=2                      # 2x faster than default, preserves Tamil accuracy
            )

        decoded = tokenizer.batch_decode(
            generated_tokens.cpu().tolist(),
            skip_special_tokens=True
        )

        return self.ip.postprocess_batch(decoded, lang=tgt_lang)

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
            
        if detected_language == "en":
            # If English is spoken, translate TO the target language from the UI dropdown
            tgt_lang = LANG_CODE_MAP.get(target_language, "tam_Taml")
        else:
            # If an Indic language is spoken, always translate TO English
            tgt_lang = "eng_Latn"
            
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
