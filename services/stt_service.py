import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
from models.whisper_model import whisper_model, tamil_whisper_model


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

# ── Confidence-based retry thresholds ─────────────────────────────────────────
RETRY_AVG_LOGPROB_THRESHOLD  = -1.5   # below this = poor transcription quality (tightened for latency)
RETRY_COMPRESSION_RATIO_MAX  = 3.2    # above this = hallucination, reject not retry
RETRY_NO_SPEECH_THRESHOLD    = 0.80   # above this = bad audio, worth retrying
RETRY_BEAM_SIZE              = 3      # beam size for retry pass (reduced for latency)

# Hallucination phrases Whisper emits for near-silence in Tamil context.
# If the full transcription matches one of these, treat it as empty.
HALLUCINATION_PATTERNS = [
    r"^\s*\.\s*$",                    # just a period
    r"^\s*நன்றி\s*\.?\s*$",           # "thank you" on silence
    r"^\s*சரி\s*\.?\s*$",             # "ok" on silence
    r"^\s*[\u0B80-\u0BFF]{1,3}\s*$",  # 1-3 Tamil chars only (hallucinated syllable)
    r"^\s*[a-zA-Z]{1,4}\s*$",         # 1-4 Latin chars only
    r"^\s*\.+\s*$",                # ellipsis or multiple periods
    r"(?i).*namashivayam.*ramraj.*tirupur.*", # Initial prompt leakage
    r"(?i).*hello.*how are you.*",
    r"(?i).*வணக்கம்.*எப்படி இருக்கிறீர்கள்.*",
]

_HALLUCINATION_RE = [re.compile(p) for p in HALLUCINATION_PATTERNS]


def _is_hallucination(text: str) -> bool:
    """Returns True if text looks like a Whisper hallucination rather than real speech."""
    t = text.strip()
    if not t:
        return True
    return any(r.match(t) for r in _HALLUCINATION_RE)


from config import STT_NO_SPEECH_THRESHOLD, STT_LANG_CONFIDENCE_FLOOR, STT_BEAM_SIZE, STT_BEAM_SIZE_TAMIL

class STTService:
    """
    Speech-to-Text service using Faster-Whisper with dual-model routing.

    Architecture:
    - Primary model: large-v3-turbo (English + language detection + multilingual fallback)
    - Tamil model: vasista22/whisper-tamil-medium (specialized Tamil fine-tune)

    Flow:
    1. Primary model transcribes the audio chunk (fast, good at English)
    2. If Tamil is detected, re-transcribe with the Tamil-specific model
    3. If Tamil model is unavailable, use primary model's output as fallback

    Key features:
    - no_speech_prob threshold: rejects frames Whisper itself is uncertain about
    - Hallucination filter: catches common Tamil silence-hallucinations
    - Tamil/Malayalam reclassification: fixes Whisper's most common dialect error
    - Tanglish detection: flags code-switched input for special handling downstream
    """

    # Reject frames where Whisper's own silence probability exceeds this.
    NO_SPEECH_THRESHOLD = STT_NO_SPEECH_THRESHOLD

    # If Tamil is detected with less than this confidence AND the language
    # could be confused with a South Indian neighbour, reclassify as Tamil.
    LANG_CONFIDENCE_THRESHOLD = STT_LANG_CONFIDENCE_FLOOR

    # Tanglish detection: fraction of words that appear to be English
    # (Latin-script, non-punctuation) in a predominantly Tamil utterance.
    TANGLISH_ENGLISH_RATIO_THRESHOLD = 0.40

    def __init__(self, beam_size: int = STT_BEAM_SIZE):
        self.model       = whisper_model
        self.tamil_model = tamil_whisper_model  # None if not available
        self.beam_size       = beam_size
        self.beam_size_tamil = STT_BEAM_SIZE_TAMIL

        if self.tamil_model is not None:
            print("STTService initialized with DUAL-MODEL routing (Turbo + Tamil).")
        else:
            print("STTService initialized (single model — Tamil model not loaded).")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_input,            # str path or 1-D float32 numpy array at 16kHz
        language: str | None = None,
        no_speech_threshold: float | None = None,
        force_language: str = "",
        initial_prompt: str | None = None
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
        # ── Tamil-First Bypass (Priority 2) ──
        if force_language == "ta" and self.tamil_model is not None:
            print("[STT] Tamil-First Mode active: Bypassing primary model directly to Tamil model.")
            segments_gen, info = self.tamil_model.transcribe(
                audio_input,
                beam_size=self.beam_size_tamil,
                language="ta",
                initial_prompt=initial_prompt,
                vad_filter=False,            # Already validated by streaming VAD
                condition_on_previous_text=False,
                temperature=0.0,
            )
            segments = list(segments_gen)
            detected_lang = "ta"
        else:
            segments_gen, info = self.model.transcribe(
                audio_input,
                beam_size=self.beam_size,
                language=language,
                initial_prompt=initial_prompt,
                vad_filter=False,            # Already validated by streaming VAD
                condition_on_previous_text=False,
                temperature=0.0,
            )

            segments = list(segments_gen)
            detected_lang = info.language

            # ── Dual-Model Routing ────────────────────────────────────────────────
            # If the primary model detects Tamil, re-run with the specialized Tamil model.
            if detected_lang == "ta" and self.tamil_model is not None:
                print("[STT] Primary model detected Tamil. Re-routing through specialized Tamil model...")
                segments_gen_ta, info_ta = self.tamil_model.transcribe(
                    audio_input,
                    beam_size=self.beam_size_tamil,
                    language="ta",
                    initial_prompt=initial_prompt,
                    vad_filter=False,            # Already validated by streaming VAD
                    condition_on_previous_text=False,
                    temperature=0.0,
                )
                segments = list(segments_gen_ta)
                info = info_ta
                detected_lang = "ta"

        # ── Quality assessment ─────────────────────────────────────────────────
        quality = self._assess_segment_quality(segments)

        if quality["action"] == "reject":
            # Hallucination detected — treat as silence, don't retry
            print(f"[STT] Rejected: {quality['reason']}")
            return self._empty("", info, silence=True)

        if quality["action"] == "retry":
            print(
                f"[STT] Running retry pass "
                f"(beam_size={RETRY_BEAM_SIZE}, temperature=0.2)"
            )
            
            retry_model = self.tamil_model if (detected_lang == "ta" and self.tamil_model is not None) else self.model
            retry_lang_arg = "ta" if retry_model == self.tamil_model else language
            
            retry_gen, retry_info = retry_model.transcribe(
                audio_input,
                beam_size=RETRY_BEAM_SIZE,
                language=retry_lang_arg,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,
                    speech_pad_ms=100,
                    min_speech_duration_ms=200,
                ),
                condition_on_previous_text=False,
                temperature=0.2,   # add slight randomness to escape local optima
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
            retry_segments = list(retry_gen)
            retry_quality  = self._assess_segment_quality(retry_segments)

            if retry_quality["action"] == "reject":
                # Retry also hallucinated — discard
                print("[STT] Retry also hallucinated — discarding")
                return self._empty("", info, silence=True)

            if retry_quality["avg_logprob"] > quality["avg_logprob"]:
                print(
                    f"[STT] Retry improved: "
                    f"logprob {quality['avg_logprob']:.2f} "
                    f"→ {retry_quality['avg_logprob']:.2f}"
                )
                segments = retry_segments
                info = retry_info
            else:
                print(
                    f"[STT] Retry did not improve "
                    f"({retry_quality['avg_logprob']:.2f} ≤ "
                    f"{quality['avg_logprob']:.2f}) — keeping pass 1"
                )

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

        # Clean up any residual prompt leakage or common repetitive hallucinations
        full_text = re.sub(r'(?i)^(?:hello[.,\s]*|வணக்கம்[.,\s]*)+', '', full_text).strip()

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
                
        # If detected as Hindi but text has no Devanagari — Whisper echoed English prompt
        if detected_lang == "hi":
            deva_chars = sum(1 for ch in full_text if '\u0900' <= ch <= '\u097F')
            if deva_chars == 0:
                print(f"[STT] Hindi detected but no Devanagari script — reclassifying to 'en'")
                detected_lang = "en"

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
            
            # Check for Devanagari (Hindi)
            deva_chars = sum(1 for ch in full_text if '\u0900' <= ch <= '\u097F')
            deva_ratio = deva_chars / max(len(full_text), 1)
            
            if tamil_ratio > 0.5:   # majority Tamil characters
                print(f"[STT] Tamil script detected in 'en' output — reclassifying to 'ta'")
                detected_lang = "ta"
            elif deva_ratio > 0.5:
                print(f"[STT] Devanagari script detected in 'en' output — reclassifying to 'hi'")
                detected_lang = "hi"

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

    def _assess_segment_quality(self, segments: list) -> dict:
        """
        Analyses Whisper segment quality metrics and returns one of three actions:
          'accept'  — quality is good, continue normally
          'retry'   — quality is poor but worth a second pass
          'reject'  — hallucination detected, discard immediately

        These three metrics are Whisper's own internal quality signals:
          avg_logprob:       log probability of output tokens (lower = less confident)
          compression_ratio: output/input length ratio (higher = repetition/hallucination)
          no_speech_prob:    Whisper's own silence estimate per segment
        """
        if not segments:
            return {
                "action": "reject", "reason": "no segments",
                "avg_logprob": 0.0, "max_compression": 0.0, "avg_no_speech": 1.0
            }

        avg_logprob = sum(
            getattr(s, "avg_logprob", 0.0) for s in segments
        ) / len(segments)

        max_compression = max(
            getattr(s, "compression_ratio", 1.0) for s in segments
        )

        avg_no_speech = sum(
            getattr(s, "no_speech_prob", 0.0) for s in segments
        ) / len(segments)

        # High compression = repetition/hallucination.
        # Retrying will produce the same garbage — reject outright.
        if max_compression > RETRY_COMPRESSION_RATIO_MAX:
            print(
                f"[STT] Quality: REJECT "
                f"(compression={max_compression:.2f} > {RETRY_COMPRESSION_RATIO_MAX})"
            )
            return {
                "action": "reject",
                "reason": f"hallucination (compression={max_compression:.2f})",
                "avg_logprob": avg_logprob,
                "max_compression": max_compression,
                "avg_no_speech": avg_no_speech,
            }

        # Low logprob or high no_speech = poor quality but worth retrying
        if (avg_logprob < RETRY_AVG_LOGPROB_THRESHOLD or
                avg_no_speech > RETRY_NO_SPEECH_THRESHOLD):
            print(
                f"[STT] Quality: RETRY "
                f"(logprob={avg_logprob:.2f}, no_speech={avg_no_speech:.2f})"
            )
            return {
                "action": "retry",
                "reason": f"low quality (logprob={avg_logprob:.2f}, no_speech={avg_no_speech:.2f})",
                "avg_logprob": avg_logprob,
                "max_compression": max_compression,
                "avg_no_speech": avg_no_speech,
            }

        print(
            f"[STT] Quality: ACCEPT "
            f"(logprob={avg_logprob:.2f}, compression={max_compression:.2f})"
        )
        return {
            "action": "accept",
            "reason": "ok",
            "avg_logprob": avg_logprob,
            "max_compression": max_compression,
            "avg_no_speech": avg_no_speech,
        }

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
