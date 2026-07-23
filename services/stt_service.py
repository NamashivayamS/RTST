import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
import time
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


TAMIL_CONFUSED_AS = set()

TAMIL_SCRIPT_INDICATORS = {"ml", "kn", "te", "hi"}  # these use different scripts

# ── Confidence-based retry thresholds ─────────────────────────────────────────
RETRY_AVG_LOGPROB_THRESHOLD  = -1.5   # below this = poor transcription quality (tightened for latency)
RETRY_COMPRESSION_RATIO_MAX  = 3.2    # above this = hallucination, reject not retry (English)
RETRY_COMPRESSION_RATIO_MAX_INDIC = 5.0  # Tamil/Indic scripts have naturally higher compression ratios
RETRY_NO_SPEECH_THRESHOLD    = 0.80   # above this = bad audio, worth retrying
RETRY_BEAM_SIZE              = 3      # beam size for retry pass (reduced for latency)


HALLUCINATION_PATTERNS = [
    r"^\s*\.\s*$",                    # just a period
    r"^\s*நன்றி\s*\.?\s*$",           # "thank you" on silence
    r"^\s*சரி\s*\.?\s*$",             # "ok" on silence
    r"^\s*[\u0B80-\u0BFF]{1,3}\s*$",  # 1-3 Tamil chars only (hallucinated syllable)
    r"^\s*[a-zA-Z]{1,4}\s*$",         # 1-4 Latin chars only
    r"^\s*\.+\s*$",                
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


from config import STT_NO_SPEECH_THRESHOLD, STT_LANG_CONFIDENCE_FLOOR, LANGUAGE_REGISTRY


def _get_script_ratio(text: str, script_range: tuple) -> float:
    """Returns the fraction of characters in text within the given Unicode range."""
    if not script_range or not text:
        return 0.0
    lo, hi = script_range
    script_chars = sum(1 for ch in text if lo <= ord(ch) <= hi)
    return script_chars / max(len(text), 1)

# Default beam sizes for transcription
STT_BEAM_SIZE = 2        # primary model (English detection)
STT_BEAM_SIZE_TAMIL = 5 # Tamil fine-tune (beam search for accuracy)

class STTService:
    """
    Speech-to-Text service using Faster-Whisper with dual-model routing.

    Architecture:
    - Primary model: medium (English + language detection + multilingual fallback)
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
        # ── Per-stage timing instrumentation ────────────────────────────────
        _primary_start = time.perf_counter()
        tamil_rerouted = False
        retry_fired = False
        stage_timings = {"primary_pass_ms": 0, "tamil_reroute_ms": 0, "retry_pass_ms": 0}

        # Determine target language code for Whisper.
        # If force_language is explicitly provided (from UI selection), use it.
        # Otherwise, fall back to language parameter, which defaults to None (auto-detection).
        lang_arg = force_language if force_language else language
        if lang_arg == "":
            lang_arg = None

        # ── Tamil-First Bypass (Priority 2) ──
        if force_language == "ta" and self.tamil_model is not None:
            print("[STT] Tamil-First Mode active: Bypassing primary model directly to Tamil model.")
            segments_gen, info = self.tamil_model.transcribe(
                audio_input,
                beam_size=self.beam_size_tamil,
                language="ta",
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=1500,
                    speech_pad_ms=300,
                ),
                condition_on_previous_text=False,
                temperature=0.0,
                repetition_penalty=1.2,
            )
            segments = list(segments_gen)
            detected_lang = "ta"
            stage_timings["primary_pass_ms"] = int((time.perf_counter() - _primary_start) * 1000)
        else:
            segments_gen, info = self.model.transcribe(
                audio_input,
                beam_size=self.beam_size,
                language=lang_arg,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=1500,
                    speech_pad_ms=300,
                ),
                condition_on_previous_text=False,
                temperature=0.0,
                repetition_penalty=1.2,
            )

            segments = list(segments_gen)
            detected_lang = info.language
            stage_timings["primary_pass_ms"] = int((time.perf_counter() - _primary_start) * 1000)

            # ── Dual-Model Routing ────────────────────────────────────────────────
            # If the primary model detects Tamil, re-run with the specialized Tamil model.
            if detected_lang == "ta" and self.tamil_model is not None:
                print("[STT] Primary model detected Tamil. Re-routing through specialized Tamil model...")
                _reroute_start = time.perf_counter()
                tamil_rerouted = True
                segments_gen_ta, info_ta = self.tamil_model.transcribe(
                    audio_input,
                    beam_size=self.beam_size_tamil,
                    language="ta",
                    initial_prompt=initial_prompt,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=150,
                        min_silence_duration_ms=1500,
                        speech_pad_ms=300,
                    ),
                    condition_on_previous_text=False,
                    temperature=0.0,
                    repetition_penalty=1.2,
                )
                segments = list(segments_gen_ta)
                info = info_ta
                detected_lang = "ta"
                stage_timings["tamil_reroute_ms"] = int((time.perf_counter() - _reroute_start) * 1000)

        # ── Quality assessment ─────────────────────────────────────────────────
        quality = self._assess_segment_quality(segments, detected_lang=detected_lang)

        if quality["action"] == "reject":
            # If the first pass has a high compression ratio (hallucination loop),
            # try to recover it with the retry pass (which uses temperature & repetition penalty)
            # instead of rejecting/discarding outright (preventing utterance loss).
            if "hallucination" in quality.get("reason", ""):
                print(f"[STT] First-pass hallucination (compression={quality['max_compression']:.2f}) — attempting recovery via retry pass...")
                quality["action"] = "retry"
            else:
                print(f"[STT] Rejected: {quality['reason']}")
                return self._empty("", info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

        if quality["action"] == "retry":
            print(
                f"[STT] Running retry pass "
                f"(beam_size={RETRY_BEAM_SIZE}, temperature=0.2)"
            )
            _retry_start = time.perf_counter()
            retry_fired = True
            
            retry_model = self.tamil_model if (detected_lang == "ta" and self.tamil_model is not None) else self.model
            retry_lang_arg = "ta" if retry_model == self.tamil_model else lang_arg
            
            retry_gen, retry_info = retry_model.transcribe(
                audio_input,
                beam_size=RETRY_BEAM_SIZE,
                language=retry_lang_arg,
                initial_prompt=initial_prompt,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=150,
                    min_silence_duration_ms=1500,
                    speech_pad_ms=300,
                ),
                condition_on_previous_text=False,
                temperature=0.2,   # add slight randomness to escape local optima
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
            retry_segments = list(retry_gen)
            retry_quality  = self._assess_segment_quality(retry_segments, detected_lang=detected_lang)
            stage_timings["retry_pass_ms"] = int((time.perf_counter() - _retry_start) * 1000)

            if retry_quality["action"] == "reject":
                # Retry also hallucinated — discard
                print("[STT] Retry also hallucinated — discarding")
                return self._empty("", info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

            if retry_quality["avg_logprob"] > quality["avg_logprob"]:
                print(
                    f"[STT] Retry improved: "
                    f"logprob {quality['avg_logprob']:.2f} "
                    f"→ {retry_quality['avg_logprob']:.2f}"
                )
                segments = retry_segments
                info = retry_info
                quality = retry_quality
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
            return self._empty("", info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

        avg_no_speech = sum(
            getattr(s, "no_speech_prob", 0.0) for s in segments
        ) / len(segments)

        threshold = no_speech_threshold if no_speech_threshold is not None else self.NO_SPEECH_THRESHOLD
        if avg_no_speech > threshold:
            print(f"[STT] Silence detected (no_speech_prob={avg_no_speech:.2f} > {threshold:.2f}) — skipping.")
            return self._empty("", info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

        # ── Assemble full text ─────────────────────────────────────────────
        full_text = " ".join(seg.text.strip() for seg in segments).strip()

        # Clean up any residual prompt leakage or common repetitive hallucinations
        full_text = re.sub(r'(?i)^(?:hello[.,\s]*|வணக்கம்[.,\s]*)+', '', full_text).strip()

        # ── Phrase-level repetition deduplication ──────────────────────────
        # Whisper sometimes repeats a multi-word phrase many times (e.g.,
        # "கலியிஸ் செய்தார் கலியிஸ் செய்தார் கலியிஸ் செய்தார்").
        # Collapse any phrase of 2-6 words that repeats 3+ times consecutively.
        full_text = re.sub(r'((?:\S+\s+){1,5}\S+?)(?:\s+\1){2,}', r'\1', full_text).strip()

        # Repetition hallucination check
        words = full_text.split()
        if len(words) >= 6:
            unique_ratio = len(set(w.lower().strip('.,!?') for w in words)) / len(words)
            if unique_ratio < 0.4:
                print(f"[STT] Repetition hallucination: '{full_text[:50]}...'")
                return self._empty(full_text, info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

        # ── Hallucination filter ───────────────────────────────────────────
        if _is_hallucination(full_text):
            print(f"[STT] Hallucination filtered: '{full_text}'")
            return self._empty(full_text, info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)

        # Don't overwrite detected_lang if Tamil-first bypass was used —
        # the fine-tuned model's info.language is unreliable (can return 'ml', 'kn').
        if force_language != "ta":
            detected_lang = info.language
        # else: keep detected_lang = "ta" as set by the bypass
        
        lang_prob     = info.language_probability
        
        print(
            f"[STT] Language={detected_lang} "
            f"Prob={lang_prob:.2%}"
        )

        # ── Universal Script Validation (forced language) ─────────────────
        # If user explicitly selected a non-English Indic language, verify
        # that Whisper's output actually contains the expected native script.
        # If not, it means Whisper transliterated into Latin → reclassify.
        if force_language and force_language != "en" and force_language in LANGUAGE_REGISTRY:
            expected = LANGUAGE_REGISTRY[force_language]
            sr = expected.get("script_range")
            if sr:
                native_ratio = _get_script_ratio(full_text, sr)
                if native_ratio < 0.15:
                    print(
                        f"[STT] Script mismatch: forced '{force_language}' "
                        f"({expected['name']}) but only {native_ratio:.0%} "
                        f"native script — reclassifying to 'en'"
                    )
                    detected_lang = "en"
                else:
                    # Script matches → trust the user's explicit selection
                    detected_lang = force_language
        else:
            # Auto-detect mode OR forced English:
            # Rescue misidentified scripts (e.g., Whisper says "ml" but text is Tamil)
            if detected_lang not in ("en",) and detected_lang in LANGUAGE_REGISTRY:
                expected = LANGUAGE_REGISTRY[detected_lang]
                sr = expected.get("script_range")
                if sr and _get_script_ratio(full_text, sr) < 0.15:
                    # Whisper's detected language doesn't match the script.
                    # Scan all other languages for a match.
                    rescued = False
                    for lc, li in LANGUAGE_REGISTRY.items():
                        if lc in ("en", detected_lang):
                            continue
                        sr2 = li.get("script_range")
                        if sr2 and _get_script_ratio(full_text, sr2) > 0.3:
                            print(f"[STT] {li['name']} script detected in "
                                  f"'{detected_lang}' output — reclassifying to '{lc}'")
                            detected_lang = lc
                            rescued = True
                            break
                    if not rescued:
                        print(f"[STT] No native script in '{detected_lang}' output "
                              f"— reclassifying to 'en'")
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
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=150,
                        min_silence_duration_ms=1500,
                        speech_pad_ms=300,
                    ),
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
                    return self._empty(full_text, info, silence=True, tamil_rerouted=tamil_rerouted, retry_fired=retry_fired, stage_timings=stage_timings)
            else:
                # High confidence wrong language — just fall back silently
                print(f"[STT] Unsupported language '{detected_lang}' — falling back to 'en'")
                detected_lang = "en"


        # ── Tanglish / False-Tamil detection ───────────────────────────────
        is_tanglish = False
        if detected_lang != "en":
            latin_ratio = self._get_latin_ratio(full_text)
            
            # If the text is almost entirely English (e.g. >90%), Whisper's LID 
            # made a mistake (biased by our prompt). Force it to English.
            if latin_ratio > 0.90:
                print(f"[STT] Reclassifying '{detected_lang}' → 'en' (Transcription is 100% English)")
                detected_lang = "en"
            elif detected_lang == "ta" and latin_ratio >= self.TANGLISH_ENGLISH_RATIO_THRESHOLD:
                is_tanglish = True
                print(f"[STT] Tanglish detected in: '{full_text}'")
        elif detected_lang == "en":
            # Rescue: Whisper labeled output as English, but the text
            # may actually be in a native Indic script. Scan all languages.
            for lc, li in LANGUAGE_REGISTRY.items():
                if lc == "en":
                    continue
                sr = li.get("script_range")
                if sr and _get_script_ratio(full_text, sr) > 0.3:
                    print(f"[STT] {li['name']} script detected in 'en' "
                          f"output — reclassifying to '{lc}'")
                    detected_lang = lc
                    break

        indictrans_lang = WHISPER_LANG_TO_INDICTRANS.get(detected_lang)

        return {
            "text":            full_text,
            "language":        detected_lang,
            "language_prob":   lang_prob,
            "is_silence":      False,
            "is_tanglish":     is_tanglish,
            "indictrans_lang": indictrans_lang,
            "segments":        segments,
            "avg_logprob":     quality.get("avg_logprob", -1.0),
            "tamil_rerouted":  tamil_rerouted,
            "retry_fired":     retry_fired,
            "stage_timings":   stage_timings,
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

    def _assess_segment_quality(self, segments: list, detected_lang: str = "en") -> dict:
        """
        Analyses Whisper segment quality metrics and returns one of three actions:
          'accept'  — quality is good, continue normally
          'retry'   — quality is poor but worth a second pass
          'reject'  — hallucination detected, discard immediately

        These three metrics are Whisper's own internal quality signals:
          avg_logprob:       log probability of output tokens (lower = less confident)
          compression_ratio: output/input length ratio (higher = repetition/hallucination)
          no_speech_prob:    Whisper's own silence estimate per segment

        Note: Tamil/Indic scripts produce naturally higher compression ratios
        in Whisper's tokenizer, so we use a relaxed threshold for non-English.
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
        # Use a higher threshold for Indic scripts (Tamil, Hindi, etc.) because
        # their complex scripts naturally produce higher compression ratios.
        compression_limit = (
            RETRY_COMPRESSION_RATIO_MAX if detected_lang == "en"
            else RETRY_COMPRESSION_RATIO_MAX_INDIC
        )
        if max_compression > compression_limit:
            print(
                f"[STT] Quality: REJECT "
                f"(compression={max_compression:.2f} > {compression_limit}, lang={detected_lang})"
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
    def _empty(text: str, info, silence: bool, tamil_rerouted: bool = False, retry_fired: bool = False, stage_timings: dict | None = None) -> dict:
        if stage_timings is None:
            stage_timings = {"primary_pass_ms": 0, "tamil_reroute_ms": 0, "retry_pass_ms": 0}
        return {
            "text":            text,
            "language":        info.language,
            "language_prob":   info.language_probability,
            "is_silence":      silence,
            "is_tanglish":     False,
            "indictrans_lang": None,
            "segments":        [],
            "avg_logprob":     -1.0,
            "tamil_rerouted":  tamil_rerouted,
            "retry_fired":     retry_fired,
            "stage_timings":   stage_timings,
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
    print(f"Rerouted  : {result.get('tamil_rerouted')}")
    print(f"Retry     : {result.get('retry_fired')}")
    print(f"Timings   : {result.get('stage_timings')}")
