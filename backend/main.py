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

import numpy as np
import torch
from silero_vad import load_silero_vad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    VAD_SILENCE_SEC, VAD_MIN_SPEECH_SEC,
    VAD_MAX_SPEECH_SEC, SAMPLE_RATE as CFG_SAMPLE_RATE,
    ENABLE_TTS, ENVIRONMENT_PRESETS,
    STT_NO_SPEECH_THRESHOLD, VAD_THRESHOLD
)

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

from pydantic import BaseModel
from datetime import datetime

class FeedbackReport(BaseModel):
    src_lang: str
    tgt_lang: str
    source_text: str
    translated_text: str
    issue_type: str
    correction: str | None = None

@app.post("/api/report")
def report_mistake(report: FeedbackReport):
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        **report.dict()
    }
    # Append asynchronously to avoid blocking
    with open(os.path.join(PROJECT_ROOT, "feedback_reports.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(report_data, ensure_ascii=False) + "\n")
    return {"status": "success", "message": "Report logged"}

app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "frontend")), name="static")

manager = ConnectionManager()
router: RouterService | None = None
gpu_lock: asyncio.Lock | None = None

# ── Streaming constants ────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
RING_BUFFER_SEC    = 30                          # how much audio to keep
SILENCE_SAMPLES    = int(CFG_SAMPLE_RATE * VAD_SILENCE_SEC)
MIN_SPEECH_SAMPLES = int(CFG_SAMPLE_RATE * VAD_MIN_SPEECH_SEC)
MAX_SPEECH_SAMPLES = int(CFG_SAMPLE_RATE * VAD_MAX_SPEECH_SEC)
MAX_PIPELINE_QUEUE = 2


def _vad_on_chunk(vad_model, audio: np.ndarray, sample_rate: int, threshold: float = 0.60) -> bool:
    WINDOW_SIZE = 512
    for i in range(0, len(audio) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = audio[i : i + WINDOW_SIZE]
        tensor = torch.tensor(window, dtype=torch.float32)
        if vad_model(tensor, sample_rate).item() > threshold:
            return True
    return False

# ── Per-connection state ───────────────────────────────────────────────────────
class ConnectionState:
    def __init__(self):
        self._maxlen    = SAMPLE_RATE * RING_BUFFER_SEC  # 480,000 samples
        self._ring      = np.zeros(self._maxlen, dtype=np.float32)  # pre-allocated
        self._write     = 0      # where to write next
        self._total     = 0      # total samples ever received
        self._utt_start = 0
        self.active_tasks    = 0
        self.pending_utterance = None
        self.was_speaking    = False
        self.silence_samples = 0
        self.target_lang = "ta"
        self.translation_context = []  # Stores last 5 English source sentences for context injection
        
        self.vad_threshold         = VAD_THRESHOLD
        self.silence_samples_limit = int(VAD_SILENCE_SEC * CFG_SAMPLE_RATE)
        self.min_speech_samples    = int(VAD_MIN_SPEECH_SEC * CFG_SAMPLE_RATE)
        self.max_speech_samples    = int(VAD_MAX_SPEECH_SEC * CFG_SAMPLE_RATE)
        self.no_speech_threshold   = STT_NO_SPEECH_THRESHOLD
        self.rms_gate              = 0.005
        
        # Load a separate lightweight Silero instance per connection
        self._vad_model = load_silero_vad()
        self._vad_model.reset_states()

    def push(self, chunk: np.ndarray):
        n   = len(chunk)
        end = self._write + n

        if end <= self._maxlen:
            # Simple case — fits without wrapping
            self._ring[self._write:end] = chunk
        else:
            # Wrap-around case — split across end and start of buffer
            first = self._maxlen - self._write
            self._ring[self._write:] = chunk[:first]   # fill to end
            self._ring[:n - first]   = chunk[first:]   # wrap to start
        
        self._write  = end % self._maxlen
        self._total += n

    def get_utterance(self) -> np.ndarray:
        offset = min(self._total - self._utt_start, self._maxlen)
        end    = self._write
        start  = (end - offset) % self._maxlen

        if start < end:
            return self._ring[start:end].copy()   # no wrap — simple slice
        else:
            return np.concatenate([              # wrap — two slices joined
                self._ring[start:],
                self._ring[:end]
            ])

    def get_utterance_smart_split(self) -> np.ndarray:
        """
        Extracts the utterance but scans backwards up to 1.5 seconds 
        to find a natural pause (lowest RMS) to avoid slicing a word in half.
        Adjusts _utt_start so the remaining audio rolls over to the next utterance.
        """
        utterance = self.get_utterance()
        
        # We only try to split if the utterance is reasonably long
        if len(utterance) < 2 * SAMPLE_RATE:
            self.mark_utterance_start()
            return utterance
            
        # Scan the last 1.0 seconds in 50ms windows
        scan_length = int(1.0 * SAMPLE_RATE)
        window_size = int(0.05 * SAMPLE_RATE)
        
        start_scan = max(0, len(utterance) - scan_length)
        
        min_rms = float('inf')
        best_split_idx = len(utterance)
        
        for i in range(start_scan, len(utterance) - window_size, window_size):
            window = utterance[i:i+window_size]
            rms = float(np.sqrt(np.mean(window**2)))
            if rms < min_rms:
                min_rms = rms
                best_split_idx = i + window_size // 2
                
        # Only split if we found a valid pause that doesn't truncate too much
        if best_split_idx > SAMPLE_RATE:
            split_utt = utterance[:best_split_idx]
            rollback_samples = len(utterance) - best_split_idx
            self._utt_start = self._total - rollback_samples
            print(f"[VAD] Force-fire: Rolled back {rollback_samples/SAMPLE_RATE:.2f}s to find clean word boundary.")
            return split_utt
            
        self.mark_utterance_start()
        return utterance

    def mark_utterance_start(self):
        self._utt_start = self._total

    @property
    def utterance_length(self) -> int:
        return self._total - self._utt_start


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global router
    global gpu_lock
    if router is not None:
        return
    gpu_lock = asyncio.Lock()
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
        None, lambda: router.translation_service.translate("नमस्ते", "hi", "eng_Latn")
    )
    
    if ENABLE_TTS:
        print("[Backend] Warming up TTS...")
        await loop.run_in_executor(
            None, lambda: router.tts_service.generate_audio("வணக்கம்.")
        )
        print("[Backend] TTS warmed up.")
    else:
        print("[Backend] TTS warmup skipped (ENABLE_TTS=False).")

    print("[Backend] All models warmed up. Server ready.")


@app.on_event("shutdown")
async def shutdown_event():
    if router:
        router.shutdown()


# ── REST ───────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ready" if router is not None else "loading",
        "models": {
            "whisper": "loaded",
            "translation": "loaded" if router else "pending",
            "tts": "enabled" if ENABLE_TTS else "disabled",
        },
        "active_connections": len(manager.active_connections),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Serve the NeuralTongue UI when someone visits the URL (like via ngrok)
    html_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

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

            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if ctrl.get("action") == "ping":
                    await manager.send_json(websocket, {"type": "pong"})
                elif ctrl.get("action") == "config":
                    # Handle target language
                    if "target_lang" in ctrl:
                        if state.target_lang != ctrl["target_lang"]:
                            state.target_lang = ctrl["target_lang"]
                            state.translation_context.clear()
                            print(f"[Config] Target language changed to: {state.target_lang} (Context wiped)")
                    # Handle environment preset
                    if "environment" in ctrl:
                        preset = ENVIRONMENT_PRESETS.get(ctrl["environment"])
                        if preset:
                            state.vad_threshold       = preset["vad_threshold"]
                            state.silence_samples_limit = int(CFG_SAMPLE_RATE * preset["silence_sec"])
                            state.min_speech_samples  = int(CFG_SAMPLE_RATE * preset["min_speech_sec"])
                            state.no_speech_threshold = preset["no_speech_threshold"]
                            state.rms_gate            = preset["rms_gate"]
                            print(f"[Config] Environment set to: {ctrl['environment']}")
                continue

            # ── 250ms audio chunk ──────────────────────────────────────────
            if "bytes" not in message or not message["bytes"]:
                continue

            chunk = np.frombuffer(message["bytes"], dtype=np.float32)
            state.push(chunk)

            # ── VAD: is there speech in this chunk? ────────────────────────
            try:
                has_speech = await loop.run_in_executor(
                    None,
                    lambda c=chunk: _vad_on_chunk(state._vad_model, c, SAMPLE_RATE, state.vad_threshold)
                )
            except Exception as e:
                print(f"[VAD Error] {e}")   # temporary — remove after confirming fix
                has_speech = False

            if has_speech:
                if not state.was_speaking:
                    # Always mark the start of new speech,
                    # regardless of whether a pipeline is running
                    state.mark_utterance_start()
                    state.was_speaking = True
                state.silence_samples = 0

                if state.utterance_length >= state.max_speech_samples:
                    utterance = state.get_utterance_smart_split()
                    if state.active_tasks >= MAX_PIPELINE_QUEUE:
                        print(f"[Pipeline] Replacing stale queued utterance with newer one")
                        state.pending_utterance = utterance
                    else:
                        state.active_tasks += 1
                        asyncio.create_task(
                            _run_pipeline(websocket, utterance, state, loop)
                        )
            else:
                state.silence_samples += len(chunk)

                if state.was_speaking and state.silence_samples >= state.silence_samples_limit:
                    # Speech just ended — fire pipeline if not already running
                    state.was_speaking = False

                    if (
                        state.utterance_length >= state.min_speech_samples
                        and state.silence_samples >= state.silence_samples_limit
                    ):
                        utterance = state.get_utterance()
                        state.mark_utterance_start()  # reset for next utterance
                        if state.active_tasks >= MAX_PIPELINE_QUEUE:
                            print(f"[Pipeline] Replacing stale queued utterance with newer one")
                            state.pending_utterance = utterance
                        else:
                            state.active_tasks += 1
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
        
        # Concurrency is now handled internally by RouterService (stt_lock & translation_lock)
        result = await loop.run_in_executor(
            None, lambda: router.process_audio(
                audio, 
                target_lang=state.target_lang, 
                skip_vad=True,
                no_speech_threshold=state.no_speech_threshold,
                rms_gate=state.rms_gate
            )
        )
        if not result.get("translated_text"):
            return

        await manager.send_json(websocket, {
            "type":     "subtitle",
            "text":     result["translated_text"],
            "source_text": result.get("punctuated_text") or result.get("raw_text") or "",
            "src_lang": result["src_lang"],
            "tgt_lang": result["tgt_lang"],
            "confidence": result.get("language_prob", None),
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
            "type": "done",
            "total_chunks": chunks_sent,
            "latency_ms": round(result.get("total_time", 0) * 1000),
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
        state.active_tasks = max(0, state.active_tasks - 1)
        if state.pending_utterance is not None:
            next_utt = state.pending_utterance
            state.pending_utterance = None
            state.active_tasks += 1
            asyncio.create_task(_run_pipeline(websocket, next_utt, state, loop))
