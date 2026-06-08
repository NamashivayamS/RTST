import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
from models.whisper_model import whisper_model


# Maps Whisper ISO 639-1 codes → IndicTrans2 language tokens
WHISPER_LANG_TO_INDICTRANS = {
    "ta": "tam_Taml",
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
}

# Whisper sometimes misidentifies Tamil as these languages.
# Since we now support them, we no longer force reclassification.
TAMIL_CONFUSED_AS = set()

TAMIL_SCRIPT_INDICATORS = {"ml", "kn", "te", "hi"}  # these use different scripts

# Hallucination phrases Whisper emits for near-silence in Tamil context.
# If the full transcription matches one of these, treat it as empty.
HALLUCINATION_PATTERNS = [
    r"^\s*\.\s*$",                    # just a period
    r"^\s*நன்றி\s*\.?\s*$",           # "thank you" on silence
    r"^\s*சரி\s*\.?\s*$",             # "ok" on silence
    r"^\s*[\u0B80-\u0BFF]{1,3}\s*$",  # 1-3 Tamil chars only (hallucinated syllable)
    r"^\s*[a-zA-Z]{1,4}\s*$",         # 1-4 Latin chars only
    r"^\s*\.\.\.\s*$",                # ellipsis
]

_HALLUCINATION_RE = [re.compile(p) for p in HALLUCINATION_PATTERNS]


def _is_hallucination(text: str) -> bool:
    """Returns True if text looks like a Whisper hallucination rather than real speech."""
    t = text.strip()
    if not t:
        return True
    return any(r.match(t) for r in _HALLUCINATION_RE)


from config import STT_NO_SPEECH_THRESHOLD, STT_LANG_CONFIDENCE_FLOOR, STT_BEAM_SIZE

class STTService:
    """
    Speech-to-Text service using Faster-Whisper (medium model).

    Key improvements over the basic wrapper:
    - no_speech_prob threshold: rejects frames Whisper itself is uncertain about
    - Hallucination filter: catches common Tamil silence-hallucinations
    - Tamil/Malayalam reclassification: fixes Whisper's most common dialect error
    - Tanglish detection: flags code-switched input for special handling downstream
    - beam_size=5 for accuracy (can be reduced to 1 for speed if needed)
    """

    # Reject frames where Whisper's own silence probability exceeds this.
    NO_SPEECH_THRESHOLD = STT_NO_SPEECH_THRESHOLD

    # If Tamil is detected with less than this confidence AND the language
    # could be confused with a South Indian neighbour, reclassify as Tamil.
    LANG_CONFIDENCE_THRESHOLD = STT_LANG_CONFIDENCE_FLOOR

    # Tanglish detection: fraction of words that appear to be English
    # (Latin-script, non-punctuation) in a predominantly Tamil utterance.
    TANGLISH_ENGLISH_RATIO_THRESHOLD = 0.25

    def __init__(self, beam_size: int = STT_BEAM_SIZE):
        self.model     = whisper_model
        self.beam_size = beam_size
        print("STTService initialized and ready.")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_input,            # str path or 1-D float32 numpy array at 16kHz
        language: str | None = None,
        no_speech_threshold: float | None = None
    ) -> dict:
        """
        Transcribes audio and returns a rich result dict.

        Returns:
            {
                "text":            str,   # clean transcription (empty if silence)
                "language":        str,   # detected ISO 639-1 code
                "language_prob":   float,
                "is_silence":      bool,  # True → skip downstream processing
                "is_tanglish":     bool,  # True → code-switched Tamil+English
                "indictrans_lang": str | None,
                "segments":        list,
            }
        """
        segments_gen, info = self.model.transcribe(
            audio_input,
            beam_size=self.beam_size,
            language=language,
            # Provide context to bias the model toward Tamil and English evenly.
            # This helps language detection on very short sentences for both languages.
            initial_prompt=(
                "Coimbatore, Tirupur, Chennai, Trichy, Madurai, Ramraj, Namashivayam. "
                "Hello. Good morning. வணக்கம். நன்றி."
            ),
            # Whisper's own silence probability — returned in segment.no_speech_prob
            vad_filter=False,             # Disabled to avoid double-VAD penalty
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=False,  # prevents context bleed between chunks
            temperature=0.0,                   # force greedy decoding to prevent 15-20s latency spikes
        )

        segments = list(segments_gen)

        # ── Silence / no-speech guard ──────────────────────────────────────
        # Whisper reports no_speech_prob per segment. If ALL segments are
        # low-confidence or the overall frame is silent, skip processing.
        if not segments:
            return self._empty("", info, silence=True)

        avg_no_speech = sum(
            getattr(s, "no_speech_prob", 0.0) for s in segments
        ) / len(segments)

        threshold = no_speech_threshold if no_speech_threshold is not None else self.NO_SPEECH_THRESHOLD
        if avg_no_speech > threshold:
            print(f"[STT] Silence detected (no_speech_prob={avg_no_speech:.2f} > {threshold:.2f}) — skipping.")
            return self._empty("", info, silence=True)

        # ── Assemble full text ─────────────────────────────────────────────
        full_text = " ".join(seg.text.strip() for seg in segments).strip()

        # Repetition hallucination check
        words = full_text.split()
        if len(words) >= 6:
            unique_ratio = len(set(w.lower().strip('.,!?') for w in words)) / len(words)
            if unique_ratio < 0.4:
                print(f"[STT] Repetition hallucination: '{full_text[:50]}...'")
                return self._empty(full_text, info, silence=True)

        # ── Hallucination filter ───────────────────────────────────────────
        if _is_hallucination(full_text):
            print(f"[STT] Hallucination filtered: '{full_text}'")
            return self._empty(full_text, info, silence=True)

        # ── Language reclassification ──────────────────────────────────────
        detected_lang = info.language
        lang_prob     = info.language_probability
        
        print(
            f"[STT] Language={detected_lang} "
            f"Prob={lang_prob:.2%}"
        )

        # Whisper often confuses Tamil with Malayalam/Kannada/Telugu.
        # Check if output text is actually Tamil script regardless of detected language
        if detected_lang != "ta" and detected_lang != "en":
            tamil_chars = sum(1 for ch in full_text if '\u0B80' <= ch <= '\u0BFF')
            tamil_ratio = tamil_chars / max(len(full_text), 1)
            
            if tamil_ratio > 0.3:
                print(f"[STT] Tamil script detected in '{detected_lang}' output — reclassifying to 'ta'")
                detected_lang = "ta"

        # If language is still unsupported, check confidence before falling back
        if detected_lang not in WHISPER_LANG_TO_INDICTRANS:
            if lang_prob < 0.70:
                # Low confidence wrong language — retry with forced English
                print(f"[STT] Low-confidence '{detected_lang}' ({lang_prob:.0%}) — retrying as English")
                segments_gen2, info2 = self.model.transcribe(
                    audio_input,
                    beam_size=self.beam_size,
                    language="en",
                    vad_filter=False,
                    condition_on_previous_text=False,
                    temperature=0.0,
                )
                segments      = list(segments_gen2)
                info          = info2
                detected_lang = "en"
                lang_prob     = info2.language_probability

                # Re-run hallucination check on new transcription
                full_text = " ".join(seg.text.strip() for seg in segments).strip()
                if _is_hallucination(full_text):
                    return self._empty(full_text, info, silence=True)
            else:
                # High confidence wrong language — just fall back silently
                print(f"[STT] Unsupported language '{detected_lang}' — falling back to 'en'")
                detected_lang = "en"

        # ── Tanglish / False-Tamil detection ───────────────────────────────
        is_tanglish = False
        if detected_lang == "ta":
            latin_ratio = self._get_latin_ratio(full_text)
            
            # If the text is almost entirely English (e.g. >90%), Whisper's LID 
            # made a mistake (biased by our prompt). Force it to English.
            if latin_ratio > 0.90:
                print(f"[STT] Reclassifying 'ta' → 'en' (Transcription is 100% English)")
                detected_lang = "en"
            elif latin_ratio >= self.TANGLISH_ENGLISH_RATIO_THRESHOLD:
                is_tanglish = True
                print(f"[STT] Tanglish detected in: '{full_text}'")
        elif detected_lang == "en":
            # NEW: check if the text is actually Tamil script despite English detection
            tamil_chars = sum(1 for ch in full_text if '\u0B80' <= ch <= '\u0BFF')
            tamil_ratio = tamil_chars / max(len(full_text), 1)
            if tamil_ratio > 0.5:   # majority Tamil characters
                print(f"[STT] Tamil script detected in 'en' output — reclassifying to 'ta'")
                detected_lang = "ta"

        indictrans_lang = WHISPER_LANG_TO_INDICTRANS.get(detected_lang)

        return {
            "text":            full_text,
            "language":        detected_lang,
            "language_prob":   lang_prob,
            "is_silence":      False,
            "is_tanglish":     is_tanglish,
            "indictrans_lang": indictrans_lang,
            "segments":        segments,
        }

    def transcribe_to_text(self, audio_input, language: str | None = None) -> str:
        """Convenience — returns only the text string."""
        return self.transcribe(audio_input, language)["text"]

    # ──────────────────────────────────────────────────────────────────────
    # Tanglish detection
    # ──────────────────────────────────────────────────────────────────────

    def _get_latin_ratio(self, text: str) -> float:
        """
        Returns the ratio of Latin-script words to total words.
        """
        words = text.split()
        if not words:
            return 0.0

        latin_words = sum(
            1 for w in words
            if re.match(r"^[a-zA-Z]+$", w.strip(".,!?;:"))
        )
        return latin_words / len(words)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(text: str, info, silence: bool) -> dict:
        return {
            "text":            text,
            "language":        info.language,
            "language_prob":   info.language_probability,
            "is_silence":      silence,
            "is_tanglish":     False,
            "indictrans_lang": None,
            "segments":        [],
        }


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    service = STTService()

    audio_path = "tests/audio_samples/ref_cropped.wav"
    result = service.transcribe(audio_path, language="ta")

    print(f"\nDetected  : {result['language']} ({result['language_prob']:.2%})")
    print(f"Silence   : {result['is_silence']}")
    print(f"Tanglish  : {result['is_tanglish']}")
    print(f"Token     : {result['indictrans_lang']}")
    print(f"Text      : {result['text']}")
