
============================================================
FILE: services\chunking_service.py
============================================================

class ChunkingService:
    def __init__(self, min_words: int = 3, max_words: int = 5):
        """
        Initializes the chunking service with safe word limits for the TTS engine.
        IndicF5 performs best on 3-5 word chunks.

        min_words: minimum words before a punctuation boundary triggers a split.
                   Prevents single-word chunks like "வணக்கம்." which degrade TTS quality.
        max_words: hard upper limit — always split here regardless of punctuation.
        """
        self.min_words = min_words
        self.max_words = max_words

    def split_text_for_tts(self, text: str) -> list[str]:
        """
        Intelligently splits Tamil/English text into safe TTS chunks.

        Split priority:
          1. Punctuation boundary — only if current chunk has >= min_words.
             This prevents tiny 1-word chunks like "வணக்கம்." from being sent alone.
          2. max_words hard limit — always split here to prevent TTS degradation.
          3. Remaining words — flushed as the final chunk regardless of size.
        """
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []

        punctuation_marks = {'.', ',', '!', '?', ';', ':', '।'}

        for word in words:
            current_chunk.append(word)

            has_punctuation = any(word.endswith(p) for p in punctuation_marks)

            # Condition 1: Natural pause AND chunk is large enough to stand alone
            if has_punctuation and len(current_chunk) >= self.min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                continue

            # Condition 2: Hard word limit reached — must split
            if len(current_chunk) >= self.max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Flush remaining words as the final chunk
        if current_chunk:
            # If remaining words are very short AND there are previous chunks,
            # merge into the last chunk rather than creating a tiny orphan chunk.
            if chunks and len(current_chunk) < self.min_words:
                chunks[-1] = chunks[-1] + " " + " ".join(current_chunk)
            else:
                chunks.append(" ".join(current_chunk))

        return chunks


# Quick test when run directly
if __name__ == "__main__":
    service = ChunkingService(min_words=3, max_words=5)

    tests = [
        # Standard multi-sentence Tamil
        "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்.",
        # Single short word with punctuation — should NOT become a 1-word chunk
        "வணக்கம். நான் நலமாக இருக்கிறேன்.",
        # English sentence
        "Hello friends, the meeting will start tomorrow morning at eight o'clock.",
        # No punctuation at all
        "இது ஒரு நீண்ட வாக்கியம் எந்த நிறுத்தற்குறியும் இல்லாமல் தொடர்கிறது",
    ]

    for test in tests:
        print(f"\nInput : {test}")
        chunks = service.split_text_for_tts(test)
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i} ({len(chunk.split())} words): '{chunk}'")


============================================================
FILE: services\correction_service.py
============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.corrections.correction_engine import apply_corrections
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS


class CorrectionService:
    """
    Post-STT text correction service.

    Language-aware: Tamil/Tanglish corrections only applied when source
    language is Tamil ('ta'). English input returned as-is.
    """

    def __init__(self):
        print("CorrectionService initialized and ready.")

    def correct(self, text: str, language: str = "ta") -> str:
        if not text.strip():
            return text
        if language == "ta":
            return apply_corrections(text)
        if language == "en":
            return text.strip()
        return text.strip()

    def correct_tamil_only(self, text: str) -> str:
        text = text.strip()
        corrected = text
        for wrong, correct in TAMIL_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected

    def correct_tanglish_only(self, text: str) -> str:
        text = text.strip()
        corrected = text
        for wrong, correct in TANGLISH_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected


if __name__ == "__main__":
    service = CorrectionService()
    tamil_raw = "நேத்து மீட்டிங்ல என்னடா நடந்துச்சு"
    english_raw = "  hello friends how are you  "
    print("Tamil correction:")
    print(f"  IN : {tamil_raw}")
    print(f"  OUT: {service.correct(tamil_raw, language='ta')}")
    print("\nEnglish (no-op):")
    print(f"  IN : {english_raw!r}")
    print(f"  OUT: {service.correct(english_raw, language='en')!r}")


============================================================
FILE: services\punctuation_service.py
============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.punctuation_model import punctuation_model


class PunctuationService:
    """
    Punctuation restoration using DeepMultilingualPunctuation.
    CPU-bound intentionally. Only applied to English input in the pipeline —
    skipped for Tamil (Latin-script model, adds latency without benefit).
    """

    def __init__(self):
        self.model = punctuation_model
        print("PunctuationService initialized and ready.")

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        return self.model.restore_punctuation(text)

    def restore_if_needed(self, text: str) -> str:
        """Skip if text already has punctuation (e.g. numbers like 3.5 count)."""
        punctuation_chars = set(".!?,;:")
        # Only skip if there's an actual sentence-ending punctuation
        # not just any dot (avoids the 3.5 false positive)
        sentence_enders = set(".!?")
        already_punctuated = any(ch in text for ch in sentence_enders)
        if already_punctuated:
            return text
        return self.restore(text)


if __name__ == "__main__":
    service = PunctuationService()
    samples = [
        "the meeting will start tomorrow morning at eight",
        "hello friends welcome to our translation system",
        "bro inniku namma project demo panrom"
    ]
    for sample in samples:
        result = service.restore(sample)
        print(f"  IN : {sample}")
        print(f"  OUT: {result}\n")


============================================================
FILE: services\refinement_service.py
============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.translation_refinement.translation_refiner import refine_translation


class RefinementService:
    """
    Post-translation refinement.
    Only refines Tamil output (en→ta). English output from IndicTrans2
    is clean enough to not need rule-based refinement.
    """

    def __init__(self):
        print("RefinementService initialized and ready.")

    def refine(self, text: str, tgt_lang: str = "tam_Taml") -> str:
        if not text.strip():
            return text
        if tgt_lang == "tam_Taml":
            return refine_translation(text)
        return text

    def refine_auto(self, translation_result: dict) -> str:
        return self.refine(
            translation_result["translated_text"],
            tgt_lang=translation_result["tgt_lang"]
        )


if __name__ == "__main__":
    service = RefinementService()
    raw_tamil = "குட் மார்னிங் ஹலோ எல்லாருக்கும் தாங்க் யூ"
    refined = service.refine(raw_tamil, tgt_lang="tam_Taml")
    print(f"IN : {raw_tamil}")
    print(f"OUT: {refined}")


============================================================
FILE: services\router_service.py
============================================================

import threading
import queue
import time
import sys
import os
import itertools

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── IMPORTANT: Load CPU-bound services BEFORE any CUDA model ──────────────────
# PunctuationModel silently crashes if Whisper has already claimed the CUDA
# context on Windows. Maintain this import order strictly.
from services.correction_service  import CorrectionService
from services.punctuation_service import PunctuationService
from services.refinement_service  import RefinementService

# ── GPU models ────────────────────────────────────────────────────────────────
from services.vad_service         import VADService
from services.stt_service         import STTService          # Whisper loads here
from services.translation_service import TranslationService

# ── Downstream ────────────────────────────────────────────────────────────────
from services.chunking_service    import ChunkingService
from services.tts_service         import TTSService          # IndicF5 loads here

_SHUTDOWN_SENTINEL = object()


class RouterService:
    """
    Orchestrates the full Real-Time Speech Translation pipeline.

    Pipeline:
        Audio → VAD gate → STT → Correction → (Punctuation for EN only)
              → Translation → Refinement → Chunking → TTS → Audio out

    Key design decisions:
    - CPU models initialised before GPU models (Windows CUDA conflict fix).
    - Stale audio queue is flushed before each new utterance so old chunks
      never bleed into the next translation's audio stream.
    - VRAM is cleared after each TTS generation cycle to prevent fragmentation.
    - Tanglish input is flagged by STT and handled with combined corrections.
    - Punctuation restoration is SKIPPED for Tamil input (Latin-script model,
      adds latency without benefit for Tamil text).
    - is_silence gate: empty/hallucinated STT output never reaches translation.
    """

    def __init__(self):
        print("RouterService: Initialising all services in main thread...")

        # NOTE:
        # Global cancellation token.
        # Valid only for single-user architecture.
        self.cancel_event = threading.Event()

        # Single-user pipeline protection
        self.processing_lock = threading.Lock()        

        self.request_counter = itertools.count(1)

        # CPU-bound first — punctuation model must load before Whisper
        self.correction_service  = CorrectionService()
        self.punctuation_service = PunctuationService()
        self.refinement_service  = RefinementService()

        # GPU models
        self.vad_service         = VADService()
        self.stt_service         = STTService()
        self.translation_service = TranslationService()

        # Downstream
        self.chunking_service = ChunkingService(min_words=3, max_words=5)
        self.tts_service      = TTSService()

        # Queues
        self.tts_input_queue   = queue.Queue()
        self.audio_output_queue = queue.Queue()

        # Background TTS worker
        self.worker_thread = threading.Thread(
            target=self._tts_worker_loop,
            daemon=False,
            name="TTS-Worker"
        )
        self.worker_thread.start()

        print("RouterService: All services ready. Background TTS worker started.")

    # ──────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ──────────────────────────────────────────────────────────────────────────

    def shutdown(self):
        self.tts_input_queue.put(_SHUTDOWN_SENTINEL)
        self.worker_thread.join(timeout=30)
        print("RouterService: Shutdown complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Background TTS worker
    # ──────────────────────────────────────────────────────────────────────────

    def _tts_worker_loop(self):
        print("[TTS Worker] Ready and listening for chunks...")
        import torch

        while True:
            wait_start = time.perf_counter()
            payload = self.tts_input_queue.get()
            wait_time = time.perf_counter() - wait_start

            print(
                f"[QUEUE WAIT] {wait_time:.3f}s"
            )

            if payload is _SHUTDOWN_SENTINEL:
                self.tts_input_queue.task_done()
                break

            try:
                if self.cancel_event.is_set():
                    print("[TTS Worker] Client disconnected. Cancelling.")
                    continue

                chunk_text   = payload["text"]
                chunk_index  = payload["chunk_index"]
                total_chunks = payload["total_chunks"]

                print(f"[TTS Worker] Generating chunk {chunk_index}/{total_chunks}: '{chunk_text}'")

                tts_start = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                audio_arr, sr = self.tts_service.generate_audio(chunk_text)
                #audio_arr = np.zeros(24000, dtype=np.float32)
                #sr = 24000
                tts_time = time.perf_counter() - tts_start

                print(
                    f"[TTS Worker] "
                    f"Chunk={chunk_index} "
                    f"Time={tts_time:.3f}s"
                )

                self.audio_output_queue.put({
                    "text":         chunk_text,
                    "audio":        audio_arr,
                    "sample_rate":  sr,
                    "chunk_index":  chunk_index,
                    "total_chunks": total_chunks,
                })

            except Exception as e:
                print(f"[TTS Worker Error] chunk='{payload.get('text', '')}': {e}")
                self.audio_output_queue.put({
                    "text":         payload.get("text", ""),
                    "audio":        np.zeros(24000, dtype=np.float32),
                    "sample_rate":  24000,
                    "chunk_index":  payload.get("chunk_index", -1),
                    "total_chunks": payload.get("total_chunks", -1),
                    "error":        str(e),
                })

            finally:
                self.tts_input_queue.task_done()

                # Optional VRAM cleanup.
                # Benchmark before deciding whether this should remain enabled.
                try:
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        used = torch.cuda.memory_allocated() / 1024**2
                        reserved = torch.cuda.memory_reserved() / 1024**2
                        peak = torch.cuda.max_memory_allocated() / 1024**2
                        print(
                            f"[VRAM] "
                            f"Allocated={used:.0f}MB "
                            f"Reserved={reserved:.0f}MB "
                            f"Peak={peak:.0f}MB"
                        )
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    # Queue management
    # ──────────────────────────────────────────────────────────────────────────

    def _flush_stale_audio(self):
        """
        Drains any leftover audio chunks from the previous utterance.

        Without this, if the WebSocket client disconnected mid-utterance or
        missed frames, stale audio chunks accumulate and are returned to the
        NEXT caller — causing wrong audio to play under the wrong subtitle.
        """
        flushed = 0
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        if flushed:
            print(f"[Router] Flushed {flushed} stale audio chunk(s) from previous utterance.")

    # ──────────────────────────────────────────────────────────────────────────
    # Main pipeline entry point
    # ──────────────────────────────────────────────────────────────────────────

    def process_audio(self, audio_input, language: str | None = None) -> dict:
        """
        Full pipeline: raw audio array (float32, 16kHz) → subtitles + TTS queue.

        Returns immediately with the translation result dict.
        Audio chunks are generated asynchronously and retrieved via
        get_generated_audio().
        """
        request_id = next(self.request_counter)

        print(f"\n{'='*60}")
        print(f"[REQUEST {request_id}] START")
        print(f"{'='*60}")

        if not self.processing_lock.acquire(blocking=False):

            print(
                f"[REQUEST {request_id}] "
                f"SKIPPED - Previous request still running"
            )

            return self._empty_result()

        pipeline_start = time.perf_counter()

        try:
            if isinstance(audio_input, np.ndarray):

                duration_sec = len(audio_input) / 16000

                print(
                    f"[AUDIO] Duration={duration_sec:.2f}s"
                )        

            # ── Flush any stale audio from the previous utterance ────────────────
            self._flush_stale_audio()

            # ── 1. VAD gate ──────────────────────────────────────────────────────
            # Quick energy check before spending GPU time on Whisper.
            # If the audio array is all near-silence, skip immediately.
            if isinstance(audio_input, np.ndarray):
                rms = float(np.sqrt(np.mean(audio_input ** 2)))
                if rms < 0.005:
                    print(f"[Pipeline] VAD gate: RMS={rms:.4f} — silence, skipping.")
                    return self._empty_result()

            print(
                f"[QUEUE] TTS Input="
                f"{self.tts_input_queue.qsize()}"
            )

            print(
                f"[QUEUE] Audio Output="
                f"{self.audio_output_queue.qsize()}"
            )        

            # ── 2. STT ───────────────────────────────────────────────────────────
            stt_start = time.perf_counter()
            stt_result = self.stt_service.transcribe(audio_input, language=language)
            stt_time = time.perf_counter() - stt_start

            if stt_result["is_silence"]:
                print("[Pipeline] STT silence gate triggered — skipping.")
                return self._empty_result()

            raw_text    = stt_result["text"]
            src_lang    = stt_result["language"]
            is_tanglish = stt_result["is_tanglish"]

            print(f"\n[Pipeline] STT ({src_lang}, tanglish={is_tanglish}): {raw_text}")

            if not raw_text.strip():
                return self._empty_result()

            # ── 3. Correction ─────────────────────────────────────────────────────
            # For Tanglish: apply both Tamil corrections AND Tanglish transliterations.
            # For pure Tamil: apply only Tamil corrections.
            # For English: pass through unchanged (no Tamil-script substitutions).
            cleaned_text = self.correction_service.correct(raw_text, language=src_lang)
            print(f"[Pipeline] Corrected : {cleaned_text}")

            # ── 4. Punctuation (English only) ─────────────────────────────────────
            # deepmultilingualpunctuation is a Latin-script model — it adds 200-400ms
            # latency for Tamil input but produces no useful output. Skip for Tamil.
            if src_lang == "en":
                punctuated_text = self.punctuation_service.restore_if_needed(cleaned_text)
                print(f"[Pipeline] Punctuated: {punctuated_text}")
            else:
                punctuated_text = cleaned_text
                print(f"[Pipeline] Punctuation skipped (Tamil input)")

            # ── 5. Translation ────────────────────────────────────────────────────
            translation_start = time.perf_counter()
            translation_result = self.translation_service.translate_auto(
                punctuated_text,
                detected_language=src_lang
            )
            translation_time = time.perf_counter() - translation_start
            print(f"[Pipeline] Translated: {translation_result['translated_text']}")

            # ── 6. Refinement ─────────────────────────────────────────────────────
            refined_text = self.refinement_service.refine_auto(translation_result)
            print(f"[Pipeline] Refined   : {refined_text}")

            # ── 7. Chunk + queue for TTS ──────────────────────────────────────────
            chunks = self.chunking_service.split_text_for_tts(refined_text)
            total  = len(chunks)

            for idx, chunk in enumerate(chunks, start=1):
                print(f"[Pipeline] Queuing TTS chunk {idx}/{total}: '{chunk}'")
                self.tts_input_queue.put({
                    "text":         chunk,
                    "chunk_index":  idx,
                    "total_chunks": total,
                })
                print(
                    f"[QUEUE] TTS Input="
                    f"{self.tts_input_queue.qsize()}"
                )

            total_time = time.perf_counter() - pipeline_start
            print(
                f"[LATENCY] "
                f"STT={stt_time:.3f}s "
                f"TRANS={translation_time:.3f}s "
                f"TOTAL={total_time:.3f}s"
            )

            return {
                "raw_text":        raw_text,
                "cleaned_text":    cleaned_text,
                "punctuated_text": punctuated_text,
                "translated_text": refined_text,
                "src_lang":        src_lang,
                "tgt_lang":        translation_result["tgt_lang"],
                "chunks":          chunks,
                "total_chunks":    total,
            }
        
        finally:
            print(
                f"[REQUEST {request_id}] COMPLETE"
            )

            self.processing_lock.release()

            print(
                f"[REQUEST {request_id}] LOCK RELEASED"
            )

    def process_translation(self, translated_text: str) -> dict:
        """
        Lightweight shortcut: skip STT/translation, go straight to TTS.
        Useful for testing and for when you already have translated text.
        """
        self._flush_stale_audio()
        print(f"\n[Pipeline] process_translation: {translated_text}")

        chunks = self.chunking_service.split_text_for_tts(translated_text)
        total  = len(chunks)

        for idx, chunk in enumerate(chunks, start=1):
            print(f"[Pipeline] Queuing TTS chunk {idx}/{total}: '{chunk}'")
            self.tts_input_queue.put({
                "text":         chunk,
                "chunk_index":  idx,
                "total_chunks": total,
            })
            print(
                f"[QUEUE] TTS Input="
                f"{self.tts_input_queue.qsize()}"
            )

        return {
            "translated_text": translated_text,
            "chunks":          chunks,
            "total_chunks":    total,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Audio retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get_generated_audio(self, block: bool = True, timeout: float = 60.0):
        try:
            return self.audio_output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def drain_audio(self, total_chunks: int, timeout: float = 60.0) -> list:
        results = []
        for i in range(total_chunks):
            payload = self.get_generated_audio(block=True, timeout=timeout)
            if payload is None:
                print(f"[Router] WARNING: Timeout waiting for chunk {i+1}/{total_chunks}")
            results.append(payload)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        return {
            "raw_text": "", "cleaned_text": "", "punctuated_text": "",
            "translated_text": "", "src_lang": "", "tgt_lang": "",
            "chunks": [], "total_chunks": 0,
        }


# ── Quick simulation ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import soundfile as sf
    import traceback

    try:
        router = RouterService()

        test_translation = (
            "காலை வணக்கம் அனைவருக்கும். "
            "இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."
        )

        start_t = time.time()
        result  = router.process_translation(test_translation)
        print(f"\nRouter returned in {time.time()-start_t:.4f}s")
        print(f"Total chunks queued: {result['total_chunks']}")

        payloads = router.drain_audio(result["total_chunks"], timeout=120)
        for p in payloads:
            if p is None:
                print("  ⚠️  Chunk timed out.")
                continue
            out = f"output_stream_chunk_{p['chunk_index']}.wav"
            sf.write(out, p["audio"], p["sample_rate"])
            print(f"  ✅ Chunk {p['chunk_index']}/{p['total_chunks']}: '{p['text']}' → {out}")

        print("\nSimulation complete.")
        router.shutdown()

    except Exception:
        traceback.print_exc()


============================================================
FILE: services\stt_service.py
============================================================

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
}

# Whisper sometimes misidentifies Tamil as these languages.
# When this happens AND confidence is low, we reclassify as Tamil.
TAMIL_CONFUSED_AS = {"ml", "kn", "te", "hi"}

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
    # 0.6 is a good balance — catches silence without rejecting quiet speech.
    NO_SPEECH_THRESHOLD = 0.6

    # If Tamil is detected with less than this confidence AND the language
    # could be confused with a South Indian neighbour, reclassify as Tamil.
    LANG_CONFIDENCE_THRESHOLD = 0.75

    # Tanglish detection: fraction of words that appear to be English
    # (Latin-script, non-punctuation) in a predominantly Tamil utterance.
    TANGLISH_ENGLISH_RATIO_THRESHOLD = 0.25

    def __init__(self, beam_size: int = 1):
        self.model     = whisper_model
        self.beam_size = beam_size
        print("STTService initialized and ready.")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_input,            # str path or 1-D float32 numpy array at 16kHz
        language: str | None = None
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
            # Whisper's own silence probability — returned in segment.no_speech_prob
            vad_filter=True,             # built-in VAD pre-filter (fast, CPU)
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
            condition_on_previous_text=False,  # prevents context bleed between chunks
            temperature=0.0,             # greedy — faster, less hallucination
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

        if avg_no_speech > self.NO_SPEECH_THRESHOLD:
            print(f"[STT] Silence detected (no_speech_prob={avg_no_speech:.2f}) — skipping.")
            return self._empty("", info, silence=True)

        # ── Assemble full text ─────────────────────────────────────────────
        full_text = " ".join(seg.text.strip() for seg in segments).strip()

        # ── Hallucination filter ───────────────────────────────────────────
        if _is_hallucination(full_text):
            print(f"[STT] Hallucination filtered: '{full_text}'")
            return self._empty(full_text, info, silence=True)

        # ── Language reclassification ──────────────────────────────────────
        detected_lang = info.language
        lang_prob     = info.language_probability

        # Whisper often confuses Tamil with Malayalam/Kannada/Telugu.
        # If confidence is low and the confused language is a known neighbour,
        # reclassify as Tamil. This is safe because our pipeline only handles
        # ta and en — anything else gets reclassified to the closer match.
        if (
            detected_lang in TAMIL_CONFUSED_AS
            and lang_prob < self.LANG_CONFIDENCE_THRESHOLD
        ):
            print(
                f"[STT] Reclassifying '{detected_lang}' ({lang_prob:.2%}) → 'ta' "
                f"(low-confidence South Indian language)"
            )
            detected_lang = "ta"

        # If language is still unsupported (e.g. 'fr', 'zh'), fall back to English
        if detected_lang not in WHISPER_LANG_TO_INDICTRANS:
            print(f"[STT] Unsupported language '{detected_lang}' — falling back to 'en'")
            detected_lang = "en"

        # ── Tanglish detection ─────────────────────────────────────────────
        is_tanglish = False
        if detected_lang == "ta":
            is_tanglish = self._detect_tanglish(full_text)
            if is_tanglish:
                print(f"[STT] Tanglish detected in: '{full_text}'")

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

    def _detect_tanglish(self, text: str) -> bool:
        """
        Returns True if the text appears to be code-switched Tamil+English.

        Strategy: count the ratio of Latin-script words to total words.
        A ratio above TANGLISH_ENGLISH_RATIO_THRESHOLD in a Tamil-detected
        utterance signals Tanglish.
        """
        words = text.split()
        if not words:
            return False

        latin_words = sum(
            1 for w in words
            if re.match(r"^[a-zA-Z]+$", w.strip(".,!?;:"))
        )
        ratio = latin_words / len(words)
        return ratio >= self.TANGLISH_ENGLISH_RATIO_THRESHOLD

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


============================================================
FILE: services\translation_service.py
============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

MODEL_REGISTRY = {
    "eng_Latn→tam_Taml": "ai4bharat/indictrans2-en-indic-dist-200M",
    "tam_Taml→eng_Latn": "ai4bharat/indictrans2-indic-en-dist-200M",
}

LANG_CODE_MAP = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
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
        if key in self._models:
            return

        model_name = MODEL_REGISTRY.get(key)
        if model_name is None:
            raise ValueError(
                f"No model registered for '{key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        print(f"TranslationService: Loading model for {key} ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16   # ~400MB VRAM saving vs float32
        ).to(self.device)

        self._tokenizers[key] = tokenizer
        self._models[key] = model
        print(f"TranslationService: Model for {key} loaded successfully.")

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
        tokenizer = self._tokenizers[key]
        model = self._models[key]

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
                max_new_tokens=max_new_tokens   # FIXED: only counts output tokens
            )

        decoded = tokenizer.batch_decode(
            generated_tokens.cpu().tolist(),
            skip_special_tokens=True
        )

        return self.ip.postprocess_batch(decoded, lang=tgt_lang)

    def translate_auto(self, text: str, detected_language: str) -> dict:
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
        tgt_lang = DEFAULT_TARGET_MAP[src_lang]
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


============================================================
FILE: services\tts_service.py
============================================================

import numpy as np
from models.indic_f5_model import generate_tamil_speech

class TTSService:
    """
    Standardized wrapper service for the IndicF5 Text-To-Speech engine.
    This service is designed to be called asynchronously by the background worker thread.
    """
    def __init__(self):
        # Model is already pre-loaded into memory by indic_f5_model import
        print("TTSService initialized and ready.")

    def generate_audio(self, text: str) -> tuple[np.ndarray, int]:
        """
        Generates audio for a given text chunk.
        Returns:
            audio_data (np.ndarray): The raw PCM audio array.
            sample_rate (int): The sample rate (24000 for IndicF5).
        """
        if not text.strip():
            # Return empty audio for empty chunks to prevent model crash
            return np.array([]), 24000
            
        try:
            # Call our heavily optimized and tuned F5 generation function
            audio_arr, sr = generate_tamil_speech(text)
            return audio_arr, sr
        except Exception as e:
            print(f"[TTSService Error] Failed to generate audio for chunk '{text}': {e}")
            # Fallback to silence on failure to keep pipeline alive
            return np.zeros(24000, dtype=np.float32), 24000

# Quick test when run directly
if __name__ == "__main__":
    service = TTSService()
    arr, sr = service.generate_audio("சோதனை.")
    print(f"Generated {len(arr)} samples at {sr}Hz.")


============================================================
FILE: services\vad_service.py
============================================================

import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps


class VADService:
    """
    Voice Activity Detection service using Silero VAD.
    Detects speech segments in raw audio, filtering out silence.
    Runs on CPU intentionally to preserve GPU VRAM for Whisper and TTS.
    """

    def __init__(self, sampling_rate: int = 16000):
        print("VADService: Loading Silero VAD model...")
        self.model = load_silero_vad()
        self.sampling_rate = sampling_rate
        print("VADService: Silero VAD loaded successfully.")

    def get_speech_segments(
        self,
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> list[dict]:
        """
        Detects speech segments in a numpy audio array.

        Args:
            audio:          1-D float32 numpy array at self.sampling_rate.
            return_seconds: If True, timestamps are in seconds (float).
                            If False, timestamps are in samples (int).

        Returns:
            List of dicts with 'start' and 'end' keys, e.g.:
            [{'start': 0.32, 'end': 2.88}, ...]
        """
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1-D audio array, got shape {audio.shape}."
            )

        wav_tensor = torch.tensor(audio, dtype=torch.float32)

        segments = get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=return_seconds
        )

        return segments

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Returns True if any speech is detected in the audio chunk.
        Useful as a quick gate before sending audio to Whisper.
        """
        segments = self.get_speech_segments(audio)
        return len(segments) > 0

    def extract_speech_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Returns a new array containing only the speech portions,
        with silence stripped out. Useful for reducing Whisper input length.
        """
        # Get segments in samples (not seconds) for slicing
        segments = self.get_speech_segments(audio, return_seconds=False)

        if not segments:
            return np.array([], dtype=np.float32)

        parts = [audio[seg["start"]: seg["end"]] for seg in segments]
        return np.concatenate(parts)


# Quick test when run directly
if __name__ == "__main__":
    import soundfile as sf

    service = VADService()

    # Replace with any available .wav for a quick sanity check
    audio_path = "tests/audio_samples/ref_cropped.wav"
    audio, sr = sf.read(audio_path)

    print(f"Audio shape: {audio.shape}, sample rate: {sr}")
    segments = service.get_speech_segments(audio.astype(np.float32))

    print(f"\nDetected {len(segments)} speech segment(s):")
    for seg in segments:
        print(f"  {seg}")

    print(f"\nHas speech: {service.has_speech(audio.astype(np.float32))}")


============================================================
FILE: services\__init__.py
============================================================


