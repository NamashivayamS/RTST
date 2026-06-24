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

from services.correction_service  import CorrectionService
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

        self.request_counter = itertools.count(1)

        # CPU-bound first — punctuation model must load before Whisper
        # IMPORTANT: We import PunctuationService inside __init__ rather than globally.
        # This prevents auto-formatters from alphabetically sorting the import below Whisper,
        # which would cause a silent CUDA crash on Windows.
        from services.punctuation_service import PunctuationService
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
        _translation_lock_held = False

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

            # ── 5. Translation ────────────────────────────────────────────────────
            if cancel_event and cancel_event.is_set():
                print(f"[REQUEST {request_id}] CANCELLED BEFORE TRANSLATION - Client disconnected")
                return self._empty_result()
                
            if not self.translation_lock.acquire(timeout=10.0):
                print(f"[REQUEST {request_id}] SKIPPED - Translation lock timeout")
                return self._empty_result()
            _translation_lock_held = True

            translation_start = time.perf_counter()
            translation_result = self.translation_service.translate_auto(
                punctuated_text,
                detected_language=src_lang,
                target_language=target_lang
            )
            translation_time = time.perf_counter() - translation_start
            self.translation_lock.release()   # Release Translation lock
            _translation_lock_held = False

            print(f"[Pipeline] Translated: {translation_result['translated_text']}")

            # ── 6. Refinement ─────────────────────────────────────────────────────
            refined_text = self.refinement_service.refine_auto(translation_result)
            print(f"[Pipeline] Refined   : {refined_text}")

            # ── 7. Chunk + queue for TTS ──────────────────────────────────────────
            if ENABLE_TTS:
                chunks = self.chunking_service.split_text_for_tts(
                    refined_text
                )
                total = len(chunks)
            else:
                chunks = []
                total = 0

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
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "punctuated_text": punctuated_text,
                "translated_text": refined_text,
                "src_lang": src_lang,
                "tgt_lang": translation_result["tgt_lang"],
                "chunks": chunks,
                "total_chunks": total,
                "total_time": time.perf_counter() - pipeline_start,
                "language_prob": stt_result.get("language_prob", 0.0),
            }

        
        finally:
            print(f"[REQUEST {request_id}] COMPLETE")
            if _stt_lock_held:
                self.stt_lock.release()
                _stt_lock_held = False
            if _translation_lock_held:
                self.translation_lock.release()
                _translation_lock_held = False
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
