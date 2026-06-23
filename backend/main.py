# backend/main.py
"""
RealTimeSpeechTranslator — FastAPI Backend (Production-Grade)
=============================================================

Changes from demo version → production version:
  1. setup_logging() called first — all print() replaced with logger calls
  2. init_pool() / close_pool() in startup/shutdown — connection pooling
  3. start_report_writer() / stop_report_writer() — thread-safe JSONL writes
  4. Token auth on /ws/translate — rejects unknown connections with WS 4003
  5. meeting_id created at connect time (not lazily) — no race condition
  6. MAX_PIPELINE_QUEUE from config.py — not a magic number in main.py
  7. GC exceptions logged (not silently swallowed)
  8. Encryption key loaded once at startup (not re-read per utterance)
  9. _save() failures are logged at ERROR level with traceback
"""

# ── FIRST: Load environment variables ──────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # Python version doesn't support reconfigure (very old versions)

# ── SECOND: configure logging before any other import that touches logging ──────
from logger import setup_logging
setup_logging(
    log_level="INFO",          # set LOG_LEVEL=DEBUG env var to see all VAD/STT detail
    log_file="logs/ispeak.log"
)

import asyncio
import json
import logging
import os
import queue
import sys
import time
import threading
import traceback
import uuid

import numpy as np
import scipy.io.wavfile
import torch
import nacl.public
import nacl.utils

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from silero_vad import load_silero_vad

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Load config AFTER path setup ──────────────────────────────────────────────
from config import (
    VAD_SILENCE_SEC, VAD_MIN_SPEECH_SEC,
    VAD_MAX_SPEECH_SEC, SAMPLE_RATE as CFG_SAMPLE_RATE,
    ENABLE_TTS, ENVIRONMENT_PRESETS,
    STT_NO_SPEECH_THRESHOLD, VAD_THRESHOLD,
    DEFAULT_DEPARTMENT_ID, SERVER_PUBLIC_KEY_PATH,
    MAX_PIPELINE_QUEUE,                          # Fix 7: from config, not hardcoded
)

from backend.connection_manager import ConnectionManager
from services.router_service import RouterService
from database.connection import init_pool, close_pool   # Fix 1: pool lifecycle
from database.queries import create_meeting, save_utterance
from auth import validate_ws_token, reject_websocket    # Fix 4: auth
from report_writer import (                             # Fix 6: thread-safe writer
    start_report_writer, stop_report_writer, enqueue_report
)

logger = logging.getLogger("ispeak.ws")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RealTimeSpeechTranslator",
    description="Tamil ↔ English real-time speech translation API",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "frontend")), name="static")

# ── Streaming constants ────────────────────────────────────────────────────────
SAMPLE_RATE        = CFG_SAMPLE_RATE
RING_BUFFER_SEC    = 30
SILENCE_SAMPLES    = int(CFG_SAMPLE_RATE * VAD_SILENCE_SEC)
MIN_SPEECH_SAMPLES = int(CFG_SAMPLE_RATE * VAD_MIN_SPEECH_SEC)
MAX_SPEECH_SAMPLES = int(CFG_SAMPLE_RATE * VAD_MAX_SPEECH_SEC)

manager = ConnectionManager()
router: RouterService | None = None
gpu_lock: asyncio.Lock | None = None

# ── Fix 3: Encryption key loaded once at startup, not per-utterance ───────────
_server_public_key: nacl.public.PublicKey | None = None

os.makedirs(os.path.join(PROJECT_ROOT, "temp_audio"),   exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "secure_vault"), exist_ok=True)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global router, gpu_lock, _server_public_key

    if router is not None:
        return

    # Fix 6: thread-safe JSONL writer
    start_report_writer(
        path=os.path.join(PROJECT_ROOT, "feedback_reports.jsonl")
    )

    # Fix 1: initialise DB connection pool
    init_pool(minconn=2, maxconn=20)

    # Fix 3: load encryption key once
    try:
        with open(SERVER_PUBLIC_KEY_PATH, "rb") as kf:
            _server_public_key = nacl.public.PublicKey(kf.read())
        logger.info("[Startup] Encryption key loaded from %s", SERVER_PUBLIC_KEY_PATH)
    except FileNotFoundError:
        logger.warning(
            "[Startup] server_public.key not found at %s — "
            "audio encryption will be DISABLED. "
            "Generate a key pair before deploying to production.",
            SERVER_PUBLIC_KEY_PATH
        )

    # Fix 8: GC now runs inside the single startup handler
    asyncio.create_task(garbage_collector_loop())

    gpu_lock = asyncio.Lock()

    logger.info("[Startup] Loading pipeline models...")
    loop = asyncio.get_event_loop()
    router = await loop.run_in_executor(None, RouterService)

    logger.info("[Startup] Warming up translation models...")
    await loop.run_in_executor(
        None, lambda: router.translation_service.translate("Hello", "en", "tam_Taml")
    )
    await loop.run_in_executor(
        None, lambda: router.translation_service.translate("வணக்கம்", "ta", "eng_Latn")
    )

    if ENABLE_TTS:
        logger.info("[Startup] Warming up TTS...")
        await loop.run_in_executor(
            None, lambda: router.tts_service.generate_audio("வணக்கம்.")
        )
    else:
        logger.info("[Startup] TTS warmup skipped (ENABLE_TTS=False).")

    logger.info("[Startup] All models ready. Server accepting connections.")


# ── Shutdown ───────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    if router:
        router.shutdown()
    close_pool()            # Fix 1: drain pool cleanly
    stop_report_writer()    # Fix 6: flush pending reports before exit
    logger.info("[Shutdown] Clean shutdown complete.")


# ── Garbage collector (Fix 8: exceptions now logged, not swallowed) ────────────
gc_logger = logging.getLogger("ispeak.gc")

async def garbage_collector_loop():
    gc_logger.info("[GC] Garbage collector started (10-minute TTL on temp_audio).")
    consecutive_failures = 0

    while True:
        await asyncio.sleep(60)
        try:
            temp_dir = os.path.join(PROJECT_ROOT, "temp_audio")
            now      = time.time()
            deleted  = 0
            for filename in os.listdir(temp_dir):
                if not filename.endswith(".enc"):
                    continue
                filepath = os.path.join(temp_dir, filename)
                if os.path.getmtime(filepath) < now - 600:  # 10 minutes
                    os.remove(filepath)
                    deleted += 1

            if deleted:
                gc_logger.info("[GC] Deleted %d expired encrypted file(s).", deleted)

            consecutive_failures = 0  # reset on success

        except Exception:
            consecutive_failures += 1
            gc_logger.exception(
                "[GC] Error during cleanup (failure #%d).", consecutive_failures
            )
            if consecutive_failures >= 5:
                gc_logger.critical(
                    "[GC] %d consecutive GC failures — "
                    "temp_audio may be filling up. Check disk permissions.",
                    consecutive_failures
                )


# ── Feedback report endpoint ───────────────────────────────────────────────────
class FeedbackReport(BaseModel):
    utterance_id: str
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
        **report.dict(),
    }

    # Move encrypted audio to vault
    temp_enc_path = os.path.join(PROJECT_ROOT, "temp_audio", f"{report.utterance_id}.wav.enc")
    vault_path    = os.path.join(PROJECT_ROOT, "secure_vault", f"{report.utterance_id}.wav.enc")

    if os.path.exists(temp_enc_path):
        try:
            import shutil
            shutil.move(temp_enc_path, vault_path)
            report_data["audio_file"] = f"secure_vault/{report.utterance_id}.wav.enc"
            logging.getLogger("ispeak").info(
                "[Report] Audio moved to vault for utterance %s", report.utterance_id
            )
        except Exception:
            logging.getLogger("ispeak").exception(
                "[Report] Failed to move audio for %s", report.utterance_id
            )
    else:
        logging.getLogger("ispeak").warning(
            "[Report] Audio for %s not found in temp_audio "
            "(expired or never saved).", report.utterance_id
        )

    # Fix 6: non-blocking, thread-safe enqueue instead of open(file, "a")
    enqueue_report(report_data)

    return {"status": "success", "message": "Report logged"}


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ready" if router is not None else "loading",
        "models": {
            "whisper":     "loaded",
            "translation": "loaded" if router else "pending",
            "tts":         "enabled" if ENABLE_TTS else "disabled",
        },
        "active_connections": len(manager.active_connections),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/status")
async def pipeline_status():
    return {
        "pipeline_ready":    router is not None,
        "active_connections": len(manager.active_connections),
    }


# ── VAD helper ─────────────────────────────────────────────────────────────────
def _vad_on_chunk(vad_model, audio: np.ndarray, sample_rate: int, threshold: float) -> bool:
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
        self._maxlen    = SAMPLE_RATE * RING_BUFFER_SEC
        self._ring      = np.zeros(self._maxlen, dtype=np.float32)
        self._write     = 0
        self._total     = 0
        self._utt_start = 0
        self.active_tasks      = 0
        self.pending_utterance = None
        self.was_speaking      = False
        self.silence_samples   = 0
        self.target_lang       = "ta"
        self.meeting_id        = None   # set at connect time, not lazily
        self.cancel_event      = threading.Event()

        self.vad_threshold         = VAD_THRESHOLD
        self.silence_samples_limit = int(VAD_SILENCE_SEC   * CFG_SAMPLE_RATE)
        self.min_speech_samples    = int(VAD_MIN_SPEECH_SEC * CFG_SAMPLE_RATE)
        self.max_speech_samples    = int(VAD_MAX_SPEECH_SEC * CFG_SAMPLE_RATE)
        self.no_speech_threshold   = STT_NO_SPEECH_THRESHOLD
        self.rms_gate              = 0.005

        self._vad_model = load_silero_vad()
        self._vad_model.reset_states()

    def push(self, chunk: np.ndarray):
        n   = len(chunk)
        end = self._write + n
        if end <= self._maxlen:
            self._ring[self._write:end] = chunk
        else:
            first = self._maxlen - self._write
            self._ring[self._write:] = chunk[:first]
            self._ring[:n - first]   = chunk[first:]
        self._write  = end % self._maxlen
        self._total += n

    def get_utterance(self) -> np.ndarray:
        offset = min(self._total - self._utt_start, self._maxlen)
        end    = self._write
        start  = (end - offset) % self._maxlen
        if start < end:
            return self._ring[start:end].copy()
        return np.concatenate([self._ring[start:], self._ring[:end]])

    def get_utterance_smart_split(self) -> np.ndarray:
        utterance = self.get_utterance()
        if len(utterance) < 2 * SAMPLE_RATE:
            self.mark_utterance_start()
            return utterance
        scan_length = int(1.0 * SAMPLE_RATE)
        window_size = int(0.05 * SAMPLE_RATE)
        start_scan  = max(0, len(utterance) - scan_length)
        min_rms, best_split_idx = float("inf"), len(utterance)
        for i in range(start_scan, len(utterance) - window_size, window_size):
            w   = utterance[i : i + window_size]
            rms = float(np.sqrt(np.mean(w ** 2)))
            if rms < min_rms:
                min_rms, best_split_idx = rms, i + window_size // 2
        if best_split_idx > SAMPLE_RATE:
            split_utt      = utterance[:best_split_idx]
            rollback       = len(utterance) - best_split_idx
            self._utt_start = self._total - rollback
            return split_utt
        self.mark_utterance_start()
        return utterance

    def mark_utterance_start(self):
        self._utt_start = self._total

    @property
    def utterance_length(self) -> int:
        return self._total - self._utt_start


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    # ── Fix 4: authenticate before accepting ──────────────────────────────────
    token = websocket.query_params.get("token")
    if not validate_ws_token(token):
        await reject_websocket(websocket, reason="Invalid or missing token")
        return

    await manager.connect(websocket)
    state = ConnectionState()
    loop  = asyncio.get_event_loop()

    # ── Fix 5: create meeting at connect time (no race condition) ─────────────
    try:
        state.meeting_id = await loop.run_in_executor(
            None,
            lambda: create_meeting(
                title="Live Translation Session",
                department_id=DEFAULT_DEPARTMENT_ID,
            )
        )
        logger.info("[WS] Meeting created for new connection: %s", state.meeting_id)
    except Exception:
        logger.exception("[WS] Failed to create meeting — DB may be down.")
        # Allow connection to proceed; DB writes will fail silently per utterance
        # rather than blocking the entire demo.

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
                    if "target_lang" in ctrl:
                        state.target_lang = ctrl["target_lang"]
                        logger.info("[Config] target_lang → %s", state.target_lang)
                    if "environment" in ctrl:
                        preset = ENVIRONMENT_PRESETS.get(ctrl["environment"])
                        if preset:
                            state.vad_threshold         = preset["vad_threshold"]
                            state.silence_samples_limit = int(CFG_SAMPLE_RATE * preset["silence_sec"])
                            state.min_speech_samples    = int(CFG_SAMPLE_RATE * preset["min_speech_sec"])
                            state.no_speech_threshold   = preset["no_speech_threshold"]
                            state.rms_gate              = preset["rms_gate"]
                            logger.info("[Config] Environment → %s", ctrl["environment"])
                continue

            if "bytes" not in message or not message["bytes"]:
                continue

            chunk = np.frombuffer(message["bytes"], dtype=np.float32)
            state.push(chunk)

            try:
                has_speech = await loop.run_in_executor(
                    None,
                    lambda c=chunk: _vad_on_chunk(
                        state._vad_model, c, SAMPLE_RATE, state.vad_threshold
                    )
                )
            except Exception:
                logger.exception("[VAD] Error on chunk — treating as silence.")
                has_speech = False

            if has_speech:
                if not state.was_speaking:
                    state.mark_utterance_start()
                    state.was_speaking = True
                state.silence_samples = 0

                if state.utterance_length >= state.max_speech_samples:
                    utterance = state.get_utterance_smart_split()
                    if state.active_tasks >= MAX_PIPELINE_QUEUE:
                        logger.debug("[Pipeline] Replacing stale pending utterance.")
                        state.pending_utterance = utterance
                    else:
                        state.active_tasks += 1
                        asyncio.create_task(
                            _run_pipeline(websocket, utterance, state, loop)
                        )
            else:
                state.silence_samples += len(chunk)
                if state.was_speaking and state.silence_samples >= state.silence_samples_limit:
                    state.was_speaking = False
                    if (
                        state.utterance_length >= state.min_speech_samples
                        and state.silence_samples >= state.silence_samples_limit
                    ):
                        utterance = state.get_utterance()
                        state.mark_utterance_start()
                        if state.active_tasks >= MAX_PIPELINE_QUEUE:
                            logger.debug("[Pipeline] Replacing stale pending utterance.")
                            state.pending_utterance = utterance
                        else:
                            state.active_tasks += 1
                            asyncio.create_task(
                                _run_pipeline(websocket, utterance, state, loop)
                            )

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[WS] Unexpected error in receive loop.")
    finally:
        state.cancel_event.set()
        manager.disconnect(websocket)


# ── Pipeline task ──────────────────────────────────────────────────────────────
async def _run_pipeline(
    websocket: WebSocket,
    audio: np.ndarray,
    state: ConnectionState,
    loop: asyncio.AbstractEventLoop,
):
    pl_logger = logging.getLogger("ispeak.pipeline")
    pl_logger.info("[Pipeline] Utterance duration: %.2fs", len(audio) / SAMPLE_RATE)

    try:
        # ── Encrypt and save audio (Fix 3: uses pre-loaded key) ───────────────
        utt_id        = f"utt_{uuid.uuid4().hex[:8]}"
        enc_temp_path = os.path.join(PROJECT_ROOT, "temp_audio", f"{utt_id}.wav.enc")

        if _server_public_key is not None:
            def _encrypt_and_save():
                import io
                wav_io = io.BytesIO()
                scipy.io.wavfile.write(wav_io, SAMPLE_RATE, audio)
                box            = nacl.public.SealedBox(_server_public_key)  # key already in RAM
                encrypted_audio = box.encrypt(wav_io.getvalue())
                with open(enc_temp_path, "wb") as f:
                    f.write(encrypted_audio)

            await loop.run_in_executor(None, _encrypt_and_save)
        else:
            pl_logger.debug(
                "[Pipeline] Skipping audio encryption (no public key loaded)."
            )

        # ── Run translation pipeline ───────────────────────────────────────────
        result = await loop.run_in_executor(
            None,
            lambda: router.process_audio(
                audio,
                target_lang=state.target_lang,
                skip_vad=True,
                no_speech_threshold=state.no_speech_threshold,
                rms_gate=state.rms_gate,
                cancel_event=state.cancel_event,
            )
        )

        if not result.get("translated_text"):
            return

        pl_logger.info(
            "[Pipeline] %s → %s | '%s' → '%s' | %.0fms",
            result.get("src_lang"), result.get("tgt_lang"),
            result.get("raw_text", "")[:60],
            result.get("translated_text", "")[:60],
            result.get("total_time", 0) * 1000,
        )

        # ── Save to DB (Fix 9: failures are logged with traceback) ─────────────
        if state.meeting_id:
            def _save():
                try:
                    save_utterance(
                        meeting_id=state.meeting_id,
                        source_text=result.get("punctuated_text") or result.get("raw_text") or "",
                        translated_text=result.get("translated_text", ""),
                        source_language=result.get("src_lang", ""),
                        target_language=result.get("tgt_lang", ""),
                        total_latency_ms=int(result.get("total_time", 0) * 1000),
                    )
                except Exception:
                    # ERROR level + exc_info=True → full traceback in log file
                    logging.getLogger("ispeak.db").error(
                        "[DB] save_utterance failed for meeting %s",
                        state.meeting_id,
                        exc_info=True,
                    )

            await loop.run_in_executor(None, _save)

        # ── Send subtitle ──────────────────────────────────────────────────────
        await manager.send_json(websocket, {
            "type":         "subtitle",
            "utterance_id": utt_id,
            "text":         result["translated_text"],
            "source_text":  result.get("punctuated_text") or result.get("raw_text") or "",
            "src_lang":     result["src_lang"],
            "tgt_lang":     result["tgt_lang"],
            "confidence":   result.get("language_prob"),
        })

        # ── Stream TTS audio chunks ────────────────────────────────────────────
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
            "type":         "done",
            "total_chunks": chunks_sent,
            "latency_ms":   round(result.get("total_time", 0) * 1000),
        })

    except Exception:
        pl_logger.exception("[Pipeline] Unhandled error in pipeline task.")
        try:
            await manager.send_json(websocket, {
                "type": "error",
                "message": "Internal pipeline error — see server logs."
            })
        except Exception:
            pass

    finally:
        state.active_tasks = max(0, state.active_tasks - 1)
        if state.pending_utterance is not None:
            next_utt               = state.pending_utterance
            state.pending_utterance = None
            state.active_tasks     += 1
            asyncio.create_task(_run_pipeline(websocket, next_utt, state, loop))
