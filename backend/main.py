"""
RealTimeSpeechTranslator — FastAPI Backend (Streaming Architecture)
===================================================================
Protocol unchanged. What changes internally:
  - Frontend now sends 250ms chunks continuously (no silence detection)
  - Backend maintains a per-connection ring buffer
  - Silero VAD on backend fires pipeline when speech boundary detected
  - asyncio.create_task() keeps receive loop running during pipeline execution
  - No global processing_lock — per-connection is_processing flag instead
"""

import asyncio
import json
import queue
import sys
import os
import traceback
import collections

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.connection_manager import ConnectionManager
from services.router_service import RouterService

app = FastAPI(
    title="RealTimeSpeechTranslator",
    description="Tamil ↔ English real-time speech translation API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()
router: RouterService | None = None

# ── Streaming constants ────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
RING_BUFFER_SEC    = 30                          # how much audio to keep
SILENCE_SAMPLES    = int(SAMPLE_RATE * 0.6)      # 600ms — fire sooner after pause
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * 0.5)      # 500ms — allow shorter phrases


# ── Per-connection state ───────────────────────────────────────────────────────
class ConnectionState:
    def __init__(self):
        maxlen = SAMPLE_RATE * RING_BUFFER_SEC
        self._ring   = collections.deque(maxlen=maxlen)
        self._total  = 0          # total samples ever pushed
        self._utt_start = 0       # sample index where current utterance began
        self.is_processing = False
        self.was_speaking  = False
        self.silence_samples = 0

    def push(self, chunk: np.ndarray):
        self._ring.extend(chunk.tolist())
        self._total += len(chunk)

    def get_utterance(self) -> np.ndarray:
        """Extract samples from utterance start to now."""
        buf = np.array(self._ring, dtype=np.float32)
        samples_in_buf = len(buf)
        # How far back is utt_start from current tail?
        offset = self._total - self._utt_start
        start  = max(0, samples_in_buf - offset)
        return buf[start:]

    def mark_utterance_start(self):
        self._utt_start = self._total

    @property
    def utterance_length(self) -> int:
        return self._total - self._utt_start


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global router
    if router is not None:
        return
    print("[Backend] Loading pipeline models...")
    loop = asyncio.get_event_loop()
    router = await loop.run_in_executor(None, RouterService)

    print("[Backend] Warming up models...")
    await loop.run_in_executor(
        None, lambda: router.translation_service.translate("Hello", "en", "tam_Taml")
    )
    await loop.run_in_executor(
        None, lambda: router.translation_service.translate("வணக்கம்", "ta", "eng_Latn")
    )
    await loop.run_in_executor(
        None, lambda: router.tts_service.generate_audio("வணக்கம்.")
    )
    print("[Backend] All models warmed up. Server ready.")


@app.on_event("shutdown")
async def shutdown_event():
    if router:
        router.shutdown()


# ── REST ───────────────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.get("/status")
async def pipeline_status():
    return {
        "pipeline_ready": router is not None,
        "active_connections": len(manager.active_connections),
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    await manager.connect(websocket)
    state = ConnectionState()
    loop  = asyncio.get_event_loop()

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── Ping / control ─────────────────────────────────────────────
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("action") == "ping":
                    await manager.send_json(websocket, {"type": "pong"})
                continue

            # ── 250ms audio chunk ──────────────────────────────────────────
            if "bytes" not in message or not message["bytes"]:
                continue

            chunk = np.frombuffer(message["bytes"], dtype=np.float32)
            state.push(chunk)

            # ── VAD: is there speech in this chunk? ────────────────────────
            try:
                segments = await loop.run_in_executor(
                    None,
                    lambda: router.vad_service.get_speech_segments(
                        chunk, return_seconds=False
                    )
                )
                has_speech = len(segments) > 0
            except Exception:
                has_speech = False

            if has_speech:
                if not state.was_speaking:
                    # Always mark the start of new speech,
                    # regardless of whether a pipeline is running
                    state.mark_utterance_start()
                    state.was_speaking = True
                state.silence_samples = 0
            else:
                state.silence_samples += len(chunk)

                if state.was_speaking and state.silence_samples >= SILENCE_SAMPLES:
                    # Speech just ended — fire pipeline if not already running
                    state.was_speaking = False

                    if (
                        not state.is_processing
                        and state.utterance_length >= MIN_SPEECH_SAMPLES
                        and state.silence_samples >= SILENCE_SAMPLES
                    ):
                        utterance = state.get_utterance()
                        state.mark_utterance_start()  # reset for next utterance
                        state.is_processing = True

                        # Fire and forget — receive loop keeps running immediately
                        asyncio.create_task(
                            _run_pipeline(websocket, utterance, state, loop)
                        )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        manager.disconnect(websocket)


async def _run_pipeline(
    websocket: WebSocket,
    audio: np.ndarray,
    state: ConnectionState,
    loop: asyncio.AbstractEventLoop,
):
    """
    Runs the full translation pipeline for one utterance.
    Executes as a background task — the WebSocket receive loop
    continues receiving new audio chunks while this runs.
    """
    try:
        print(f"[Pipeline Task] Utterance duration: {len(audio)/SAMPLE_RATE:.2f}s")
        state.mark_utterance_start()
        result = await loop.run_in_executor(
            None, router.process_audio, audio
        )

        if not result.get("translated_text"):
            return

        await manager.send_json(websocket, {
            "type":     "subtitle",
            "text":     result["translated_text"],
            "src_lang": result["src_lang"],
            "tgt_lang": result["tgt_lang"],
        })

        # TTS streaming (re-enable when ENABLE_TTS = True)
        total_chunks = result["total_chunks"]
        chunks_sent  = 0

        for _ in range(total_chunks):
            payload = await loop.run_in_executor(
                None,
                lambda: router.get_generated_audio(block=True, timeout=120)
            )
            if payload is None:
                break
            chunks_sent += 1
            await manager.send_json(websocket, {
                "type":         "audio",
                "chunk_index":  payload["chunk_index"],
                "total_chunks": payload["total_chunks"],
                "sample_rate":  payload["sample_rate"],
                "encoding":     "pcm_float32",
                "text":         payload["text"],
            })
            audio_bytes = (
                payload["audio"].tobytes()
                if payload["audio"].dtype == np.float32
                else payload["audio"].astype(np.float32).tobytes()
            )
            await manager.send_bytes(websocket, audio_bytes)

        await manager.send_json(websocket, {
            "type": "done", "total_chunks": chunks_sent
        })

    except Exception as e:
        print(f"[Pipeline Task] Error: {e}")
        traceback.print_exc()
        try:
            await manager.send_json(websocket, {
                "type": "error", "message": str(e)
            })
        except Exception:
            pass
    finally:
        state.is_processing = False
