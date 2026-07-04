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

from config import ENABLE_TTS

# ── CPU models (Must load FIRST to prevent Windows CUDA crash) ────────────────
from services.punctuation_service import PunctuationService
from services.correction_service  import CorrectionService
from services.refinement_service  import RefinementService
from services.speaker_id_service  import SpeakerIDService

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

        # NOTE: There is no shared cancel_event on RouterService.
        # Cancellation is handled per-connection via state.cancel_event in main.py.
        # The TTS worker does not need a cancel check here because main.py drains
        # and discards the audio queue in the WebSocket finally block.

        self.request_counter = itertools.count(1)

        self.correction_service  = CorrectionService()
        self.punctuation_service = PunctuationService()
        self.refinement_service  = RefinementService()
        self.speaker_id_service  = SpeakerIDService()

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

        # Split locks for Pipeline Concurrency
        self.stt_lock = threading.Lock()
        self.translation_lock = threading.Lock()

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

                # Removed: torch.cuda.empty_cache() — causes GPU stall, no benefit
                # with 1.86GB VRAM headroom
                try:
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

    def process_audio(
        self,
        audio_input,
        source_lang: str = "",
        target_lang: str = "ta",
        stt_context: str = "",
        language: str | None = None,
        skip_vad: bool = False,
        no_speech_threshold: float = None,
        rms_gate: float = 0.005,
        cancel_event: threading.Event = None,
    ) -> dict:
        """
        Full pipeline: raw audio array (float32, 16kHz) → subtitles + TTS queue.

        Returns immediately with the translation result dict.
        Audio chunks are generated asynchronously and retrieved via
        get_generated_audio().
        """
        request_id = next(self.request_counter)

        print(f"\n{'='*60}")

        print(
            f"[THREAD={threading.get_ident()}] "
            f"[REQUEST {request_id}] START"
        )

        print(f"{'='*60}")

        pipeline_start = time.perf_counter()
        _stt_lock_held = False

        if cancel_event and cancel_event.is_set():
            print(f"[REQUEST {request_id}] CANCELLED - Client disconnected")
            return self._empty_result()

        try:
            if isinstance(audio_input, np.ndarray):

                duration_sec = len(audio_input) / 16000

                print(
                    f"[AUDIO] Duration={duration_sec:.2f}s"
                )        


            # ── 1. VAD gate ──────────────────────────────────────────────────────
            vad_start = time.perf_counter()
            # Quick energy check before spending GPU time on Whisper.
            # If the audio array is all near-silence, skip immediately.
            """if isinstance(audio_input, np.ndarray):
                rms = float(np.sqrt(np.mean(audio_input ** 2)))
                if rms < 0.005:
                    print(f"[Pipeline] VAD gate: RMS={rms:.4f} — silence, skipping.")
                    return self._empty_result()"""
            if isinstance(audio_input, np.ndarray):

                rms = float(np.sqrt(np.mean(audio_input ** 2)))

                if rms < rms_gate:

                    print(
                        f"[Pipeline] RMS Gate: "
                        f"RMS={rms:.4f} — silence, skipping."
                    )

                    return self._empty_result()

                print(
                    f"[Pipeline] RMS Gate Passed "
                    f"(RMS={rms:.4f})"
                )

                if skip_vad:
                    print("[Pipeline] VAD skipped (pre-validated by streaming VAD)")
                    speech_audio = audio_input
                    vad_time = time.perf_counter() - vad_start
                    print(f"[Pipeline] VAD Processing Time: {vad_time:.3f}s")
                else:
                    # Silero VAD verification
                    segments = self.vad_service.get_speech_segments(
                        audio_input,
                        return_seconds=False
                    )

                    if not segments:
                        print("[Pipeline] Silero VAD: No speech detected.")
                        return self._empty_result()

                    print(f"[VAD] Segments={len(segments)}")

                    for i, seg in enumerate(segments, 1):
                        print(f"[VAD] {i}: {seg['start']} -> {seg['end']}")

                    start_idx = segments[0]["start"]
                    end_idx = segments[-1]["end"]
                    speech_audio = audio_input[start_idx:end_idx]

                    speech_rms = float(np.sqrt(np.mean(speech_audio ** 2)))
                    print(f"[VAD] Extracted RMS={speech_rms:.4f}")

                    if len(speech_audio) == 0:
                        print("[Pipeline] Speech extraction returned empty audio.")
                        return self._empty_result()

                    original_duration = len(audio_input) / 16000
                    speech_duration = len(speech_audio) / 16000

                    print(f"[Pipeline] Speech Extraction: {original_duration:.2f}s → {speech_duration:.2f}s")
                    audio_input = speech_audio
                    print("[Pipeline] Silero VAD: Speech detected.")

                    vad_time = time.perf_counter() - vad_start
                    print(f"[Pipeline] VAD Processing Time: {vad_time:.3f}s")
            

            print(
                f"[QUEUE] TTS Input="
                f"{self.tts_input_queue.qsize()}"
            )

            print(
                f"[QUEUE] Audio Output="
                f"{self.audio_output_queue.qsize()}"
            )        

            # ── 2. STT ───────────────────────────────────────────────────────────
            if cancel_event and cancel_event.is_set():
                print(f"[REQUEST {request_id}] CANCELLED BEFORE STT - Client disconnected")
                return self._empty_result()
                
            if not self.stt_lock.acquire(timeout=10.0):
                print(f"[REQUEST {request_id}] SKIPPED - STT lock timeout")
                return self._empty_result()
            _stt_lock_held = True

            # ── Flush any stale audio from the previous utterance ────────────────
            self._flush_stale_audio()

            stt_start = time.perf_counter()
            stt_result = self.stt_service.transcribe(
                audio_input, 
                language=language, 
                no_speech_threshold=no_speech_threshold,
                force_language=source_lang,
                initial_prompt=stt_context if stt_context else None
            )
            stt_time = time.perf_counter() - stt_start
            self.stt_lock.release()   # Release STT lock early for the next request
            _stt_lock_held = False

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
            cleaned_text = self.correction_service.correct(raw_text, language=src_lang, is_tanglish=is_tanglish)
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

            # ── Minimum word gate ─────────────────────────────────────────
            # Short fragments (≤3 words) produce garbage translations.
            # Skip them rather than wasting GPU time on nonsense output.
            word_count = len(punctuated_text.split())
            if word_count < 4:
                print(
                    f"[Pipeline] Fragment gate: {word_count} words — "
                    f"skipping translation: '{punctuated_text}'"
                )
                return {
                    **self._empty_result(),
                    "raw_text": raw_text,
                    "src_lang": src_lang,
                    "tgt_lang": "",
                    "translated_text": "",
                }

            # ── 5. Return STT result for window translation in main.py ─────────
            # process_audio() intentionally stops here. Translation is handled
            # by translate_with_window() which main.py calls separately.
            # This prevents triple-translation and ensures both draft + accurate
            # passes run under a single translation_lock acquisition.

            total_time = time.perf_counter() - pipeline_start
            print(
                f"[LATENCY] "
                f"STT={stt_time:.3f}s "
                f"TOTAL={total_time:.3f}s"
            )

            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "punctuated_text": punctuated_text,
                "src_lang": src_lang,
                "is_tanglish": is_tanglish,
                "stt_time": stt_time,
                "total_time": total_time,
                "language_prob": stt_result.get("language_prob", 0.0),
                "avg_logprob": stt_result.get("avg_logprob", -1.0),
            }

        
        finally:
            print(f"[REQUEST {request_id}] COMPLETE")
            if _stt_lock_held:
                self.stt_lock.release()
                _stt_lock_held = False
            print(f"[REQUEST {request_id}] LOCKS RELEASED")

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

    def translate_with_window(
        self,
        current_text: str,
        window: list[str],
        src_lang: str,
        target_lang: str,
    ) -> dict:
        """
        Sliding-window re-translation for improved contextual accuracy.

        Runs TWO translations under ONE translation_lock acquisition:

          Pass 1 — current chunk alone   → fast draft   (~150ms)
          Pass 2 — window + current      → accurate     (~200-250ms)

        Both passes share the GPU lock so no other request can interleave.
        Total added latency vs single-pass: ~50-100ms (only the 2nd call).

        Args:
            current_text : corrected/punctuated source text for this chunk
            window       : list of previous source texts (max TRANSLATION_WINDOW_SIZE-1)
            src_lang     : ISO 639-1 code ("ta", "en", etc.)
            target_lang  : user-selected target ("ta", "en", etc.)

        Returns:
            {
                "draft_translation":    str,
                "accurate_translation": str,
                "window_was_used":      bool,
                "src_indic":            str,   # IndicTrans2 source code
                "tgt_indic":            str,   # IndicTrans2 target code
                "translation_time":     float, # seconds for both passes
            }
        """
        from services.translation_service import LANG_CODE_MAP, DEFAULT_TARGET_MAP

        # ── Resolve ISO codes to IndicTrans2 codes ────────────────────────
        src_indic = LANG_CODE_MAP.get(src_lang)
        if src_indic is None:
            print(f"[Window] Unsupported src_lang '{src_lang}' — skipping translation.")
            return {
                "draft_translation": current_text,
                "accurate_translation": current_text,
                "window_was_used": False,
                "src_indic": "", "tgt_indic": "",
                "translation_time": 0.0,
            }

        if src_lang == target_lang:
            tgt_indic = src_indic
        elif src_lang == "en":
            tgt_indic = LANG_CODE_MAP.get(target_lang, "tam_Taml")
        else:
            tgt_indic = "eng_Latn"

        if src_indic == tgt_indic:
            return {
                "draft_translation": current_text,
                "accurate_translation": current_text,
                "window_was_used": False,
                "src_indic": src_indic, "tgt_indic": tgt_indic,
                "translation_time": 0.0,
            }

        # ── Acquire lock (or skip gracefully) ─────────────────────────────
        if not self.translation_lock.acquire(timeout=10.0):
            print("[Window] Translation lock timeout — skipping translation.")
            return {
                "draft_translation": current_text,
                "accurate_translation": current_text,
                "window_was_used": False,
                "src_indic": src_indic, "tgt_indic": tgt_indic,
                "translation_time": 0.0,
            }

        try:
            t_start = time.perf_counter()

            # ── Pass 1: translate current chunk alone (draft) ─────────────
            draft = self.translation_service.translate(
                current_text, src_indic, tgt_indic
            )
            t_pass1 = time.perf_counter()
            print(f"[Window] Pass-1 (draft) : {t_pass1-t_start:.3f}s → '{draft[:80]}'")

            # ── Pass 2: translate window + current combined (accurate) ────
            # Only run if there is at least one previous chunk in the window.
            # On the first utterance, window is empty → skip Pass 2.
            #
            # Key: after translating the combined text, we must EXTRACT only
            # the current chunk's translation. The model outputs the full
            # combined translation (prev + current), so we split by sentence
            # boundaries and skip the sentences belonging to the window.
            window_was_used = len(window) > 0
            if window_was_used:
                # ── Normalize window text with sentence-ending punctuation ──
                # so the model produces a clean sentence boundary between
                # the window's translation and the current chunk's translation.
                window_parts = []
                n_window_sentences = 0
                for w in window:
                    w = w.strip()
                    if w and w[-1] not in '.!?।':
                        w += '.'
                    window_parts.append(w)
                    n_window_sentences += self._count_sentences(w)

                window_text = " ".join(window_parts)
                combined = window_text + " " + current_text

                # Use 2× max_new_tokens for the combined pass to prevent
                # the model from truncating before it reaches the current chunk.
                # Cap at 256: IndicTrans2's decoder produces repetition artifacts
                # beyond ~256 output tokens. With TRANSLATION_WINDOW_SIZE=2 this
                # is 192 (fine). At size=3 it would be 288 (unsafe) → cap to 256.
                combined_max_tokens = min(256, 96 * (1 + len(window)))
                accurate_combined = self.translation_service.translate(
                    combined, src_indic, tgt_indic,
                    max_new_tokens=combined_max_tokens,
                )
                t_pass2 = time.perf_counter()
                print(
                    f"[Window] Pass-2 (raw combined): {t_pass2-t_pass1:.3f}s "
                    f"→ '{accurate_combined[:100]}'"
                )

                # ── Extract only the current chunk's translation ───────────
                accurate = self._extract_current_translation(
                    accurate_combined, n_window_sentences
                )
                print(
                    f"[Window] Pass-2 (extracted): "
                    f"skipped {n_window_sentences} window sentence(s) "
                    f"→ '{accurate[:80]}'"
                )

                # If extraction returned nothing, fall back to draft
                if not accurate.strip():
                    print("[Window] Extraction returned empty — falling back to draft")
                    accurate = draft
            else:
                accurate = draft
                print("[Window] Pass-2 skipped (first utterance — no previous window)")

            translation_time = time.perf_counter() - t_start

            return {
                "draft_translation": draft,
                "accurate_translation": accurate,
                "window_was_used": window_was_used,
                "src_indic": src_indic,
                "tgt_indic": tgt_indic,
                "translation_time": translation_time,
            }

        finally:
            self.translation_lock.release()

    # ── Sentence-boundary helpers for sliding-window extraction ────────────

    @staticmethod
    def _count_sentences(text: str) -> int:
        """
        Count sentences by splitting on sentence-ending punctuation.
        Handles English (. ! ?), Hindi/Sanskrit (।), and Tamil text.
        """
        import re
        text = text.strip()
        if not text:
            return 0
        # Split on sentence-ending punctuation followed by space or end-of-string
        parts = re.split(r'(?<=[.!?।])\s+', text)
        return len([p for p in parts if p.strip()])

    @staticmethod
    def _extract_current_translation(
        combined_translation: str,
        n_window_sentences: int,
    ) -> str:
        """
        Given the full translation of [window_text + current_text],
        skip the first `n_window_sentences` sentences (which belong to
        the window / context) and return only the remainder — the
        accurate translation of the current chunk.

        If the output has fewer sentences than expected, returns the
        full combined text as a safe fallback (caller checks for empty).
        """
        import re
        text = combined_translation.strip()
        if n_window_sentences <= 0:
            return text

        # Split on sentence-ending punctuation followed by whitespace.
        # re.split with a capturing group keeps the delimiters.
        parts = re.split(r'((?<=[.!?।])\s+)', text)

        # Rebuild into sentence segments (text + delimiter pairs)
        sentences = []
        current = ""
        for part in parts:
            current += part
            # Check if this part ends with sentence-ending punctuation
            if re.search(r'[.!?।]\s*$', current):
                sentences.append(current)
                current = ""
        if current.strip():
            sentences.append(current)  # last segment without trailing punct

        if len(sentences) <= n_window_sentences:
            # Couldn't split enough (model truncated early or merged everything).
            # Return empty string to force the caller to fall back to the draft translation.
            return ""

        # Return everything after the window sentences
        extracted = "".join(sentences[n_window_sentences:]).strip()
        return extracted

    @staticmethod
    def _empty_result() -> dict:
        return {
            "raw_text": "", "cleaned_text": "", "punctuated_text": "",
            "src_lang": "", "language_prob": 0.0, "avg_logprob": -1.0,
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
