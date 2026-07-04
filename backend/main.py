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

# ── Force fully-offline mode for HuggingFace / Transformers ────────────────────
# All models are already cached locally. This prevents the 30s retry delay
# when the machine has no internet, and ensures zero external network calls.
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
import secrets
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
    MAX_PIPELINE_QUEUE,
    TRANSLATION_WINDOW_SIZE,
    TURN_TAKING_SILENCE_SEC,     # controls language lock + window clear threshold
    ENABLE_SLIDING_WINDOW,       # master switch for two-pass window translation
    CORS_ALLOWED_ORIGINS,        # comma-separated allowed origins for CORS
)

from backend.connection_manager import ConnectionManager
from services.router_service import RouterService
from database.connection import init_pool, close_pool   # Fix 1: pool lifecycle
from database.queries import create_meeting, save_utterance
from auth import validate_ws_token, reject_websocket, set_runtime_token
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
    allow_origins=(
        [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
        if CORS_ALLOWED_ORIGINS
        else ["*"]
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Runtime WebSocket token ──────────────────────────────────────────────────
# Generated once per server start. Injected into the HTML at serve-time so
# the token is never hardcoded in source code or committed to git.
_RUNTIME_WS_TOKEN = secrets.token_hex(32)

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

    # Register the runtime token with the auth module
    set_runtime_token(_RUNTIME_WS_TOKEN)
    logger.info("[Startup] Runtime WebSocket token generated (rotates on restart).")

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
        html = f.read()
    # Inject the runtime token into the page so the frontend can authenticate
    # WebSocket connections without a hardcoded token in the source code.
    token_script = f'<script>window.__WS_TOKEN__="{_RUNTIME_WS_TOKEN}";</script>'
    html = html.replace("</head>", f"    {token_script}\n</head>")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


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
        self.source_lang       = ""
        self.detected_language_lock = ""
        self.language_divergence_count = 0
        self._stt_domain_seed = (
            "இராம்ராஜ் காட்டன் திருப்பூர் வேட்டி சட்டை நெசவு தொழில் ஏற்றுமதி "
            "நாகராஜன் கொங்கு தமிழ் அவிநாசி கலாச்சாரம் பிசினஸ் "
            "Ramraj Cotton Tirupur veshti shirt weaving export quality "
            "Nagarajan Kongu Tamil Avinashi business culture brand"
        )
        self.stt_context       = self._stt_domain_seed
        self.meeting_id        = None   # set at connect time, not lazily
        self.cancel_event      = threading.Event()
        self.turn_taking_fired = False

        # ── Sliding-window translation context ────────────────────────────
        # Stores the last TRANSLATION_WINDOW_SIZE-1 corrected source texts.
        # On each new utterance, the window + new chunk are re-translated
        # together for better contextual accuracy. Cleared on turn-taking
        # silence timeout so cross-speaker context never bleeds through.
        self.translation_window: list[str] = []

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
                    if "source_lang" in ctrl:
                        state.source_lang = ctrl["source_lang"]
                        state.detected_language_lock = ""
                        logger.info("[Config] source_lang → %s", state.source_lang)
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
                state.turn_taking_fired = False  # reset when speech resumes
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

                # ── Turn-Taking Silence Timeout (Auto Language Fix) ──
                # If there is >2.0s of silence, we assume the speaker finished their turn.
                # Clear the lock and context so the next speaker's language is cleanly detected.
                if state.silence_samples >= int(TURN_TAKING_SILENCE_SEC * SAMPLE_RATE):
                    if not state.turn_taking_fired:
                        state.turn_taking_fired = True
                        if state.detected_language_lock or state.stt_context:
                            logger.info("[Pipeline] Turn-taking detected (2.0s silence). Resetting context.")
                            # Only clear the language lock if the session is truly multilingual.
                            # In a monolingual Tamil session, preserve the lock so every post-silence
                            # utterance still routes directly to the fine-tuned Tamil model.
                            if state.detected_language_lock == "ta" and state.source_lang == "":
                                # Keep the Tamil lock — don't force utterance 1 of each turn through generic medium
                                pass
                            else:
                                state.detected_language_lock = ""
                            state.stt_context = state._stt_domain_seed
                            state.language_divergence_count = 0
                        if state.translation_window:
                            logger.info("[Window] Clearing translation window on turn-taking timeout.")
                            state.translation_window = []

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


# ── Pipeline task ──────────────────────────────────────────────────────────
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

        # Fire encryption in parallel — not needed by translation pipeline
        enc_task = None
        if _server_public_key is not None:
            def _encrypt_and_save():
                import io
                wav_io = io.BytesIO()
                scipy.io.wavfile.write(wav_io, SAMPLE_RATE, audio)
                box            = nacl.public.SealedBox(_server_public_key)  # key already in RAM
                encrypted_audio = box.encrypt(wav_io.getvalue())
                with open(enc_temp_path, "wb") as f:
                    f.write(encrypted_audio)

            enc_task = loop.run_in_executor(None, _encrypt_and_save)
        else:
            pl_logger.debug(
                "[Pipeline] Skipping audio encryption (no public key loaded)."
            )

        # ── Step 1: Run STT pipeline (VAD → STT → Correction → Punctuation) ──
        result = await loop.run_in_executor(
            None,
            lambda: router.process_audio(
                audio,
                source_lang=state.detected_language_lock or state.source_lang,
                target_lang=state.target_lang,
                stt_context=state.stt_context,
                skip_vad=True,
                no_speech_threshold=state.no_speech_threshold,
                rms_gate=state.rms_gate,
                cancel_event=state.cancel_event,
            )
        )

        # Gate check: process_audio now returns punctuated_text (no translation).
        # If STT produced nothing (silence, hallucination, fragment), skip.
        input_text = result.get("punctuated_text", "")
        if not input_text.strip():
            if enc_task is not None:
                await enc_task
            return

        src_lang = result.get("src_lang", "")

        # ── Language Switch Fallback Drop ──
        if state.detected_language_lock:
            if src_lang != state.detected_language_lock:
                state.language_divergence_count += 1
                pl_logger.warning(
                    "[Pipeline] Language divergence detected. Locked: %s, Output: %s. Strike %d/2.",
                    state.detected_language_lock, src_lang, state.language_divergence_count
                )
                if state.language_divergence_count >= 2:
                    pl_logger.warning("[Pipeline] Breaking language lock to force re-detection.")
                    state.detected_language_lock = ""
                    state.language_divergence_count = 0
            else:
                state.language_divergence_count = 0

        # Only apply the auto-lock if the user is in 'Auto' mode (source_lang is empty)
        if (state.source_lang == ""
                and result.get("language_prob", 0) > 0.85
                and src_lang in ("ta", "en", "hi", "te", "kn", "ml")
                and not state.detected_language_lock):
            state.detected_language_lock = src_lang
            pl_logger.info("[Pipeline] Language lock set: %s", state.detected_language_lock)

        # ── Update STT context (last ~5 words for Whisper prompt) ──────────────
        words = result.get("raw_text", "").split()
        if words:
            state.stt_context = state._stt_domain_seed + " " + " ".join(words[-5:])

        # ── Step 2: Sliding-window two-pass translation ────────────────────────
        #
        # How it works:
        #   Pass 1 (draft)    — translate current chunk alone.
        #                       Sent to frontend immediately → perceived latency unchanged.
        #   Pass 2 (accurate) — translate window[-1] + current chunk combined.
        #                       Sent as subtitle_update → frontend quietly corrects the card.
        #
        # Both passes run inside translate_with_window() under ONE lock acquisition
        # so no other pipeline request can interleave between them.
        #
        # Window management:
        #   - Append current chunk after both passes complete.
        #   - Keep only the last (TRANSLATION_WINDOW_SIZE - 1) chunks.
        #     With TRANSLATION_WINDOW_SIZE=2, we always keep exactly 1 previous chunk.
        #   - Window is cleared by the turn-taking silence timeout in the VAD loop.

        window_result = await loop.run_in_executor(
            None,
            lambda: router.translate_with_window(
                current_text=input_text,
                # Pass empty window when ENABLE_SLIDING_WINDOW=False to force
                # single-pass translation (draft only, no Pass 2).
                window=list(state.translation_window) if ENABLE_SLIDING_WINDOW else [],
                src_lang=src_lang,
                target_lang=state.target_lang,
            )
        )

        draft_text    = window_result["draft_translation"]
        accurate_text = window_result["accurate_translation"]
        window_used   = window_result["window_was_used"]
        tgt_indic     = window_result["tgt_indic"]
        trans_time    = window_result["translation_time"]

        # Apply refinement to both outputs
        draft_refined    = router.refinement_service.refine(draft_text,    tgt_lang=tgt_indic)
        accurate_refined = router.refinement_service.refine(accurate_text, tgt_lang=tgt_indic)

        # Append current chunk to window; keep only last (TRANSLATION_WINDOW_SIZE - 1) entries
        lang_prob   = result.get("language_prob", 0.0)
        avg_logprob = result.get("avg_logprob", -1.0)
        if input_text.strip() and lang_prob >= 0.80 and avg_logprob >= -1.0:
            state.translation_window.append(input_text)
            state.translation_window = state.translation_window[-(TRANSLATION_WINDOW_SIZE - 1):]
        else:
            pl_logger.info(
                "[Window] Skipping utterance from window (prob=%.2f, logprob=%.2f)",
                lang_prob, avg_logprob
            )

        total_time = result.get("stt_time", 0) + trans_time

        pl_logger.info(
            "[Pipeline] %s → %s | '%s' → '%s' | STT=%.0fms TRANS=%.0fms TOTAL=%.0fms",
            src_lang, tgt_indic,
            result.get("raw_text", "")[:60],
            draft_refined[:60],
            result.get("stt_time", 0) * 1000,
            trans_time * 1000,
            total_time * 1000,
        )

        if window_used:
            pl_logger.info(
                "[Window] draft='%s' | accurate='%s'",
                draft_refined[:60], accurate_refined[:60],
            )

        # ── Send draft subtitle FIRST (user sees it at same time as before) ────
        await manager.send_json(websocket, {
            "type":         "subtitle",
            "utterance_id": utt_id,
            "text":         draft_refined,
            "source_text":  input_text,
            "src_lang":     src_lang,
            "tgt_lang":     tgt_indic,
            "confidence":   result.get("language_prob"),
            "is_draft":     window_used,   # True only if an update will follow
            "stt_time_ms":  int(result.get("stt_time", 0) * 1000),
            "trans_time_ms": int(trans_time * 1000),
            "total_time_ms": int(total_time * 1000),
        })

        # ── Send accurate update if the window improved the translation ─────────
        if window_used and accurate_refined.strip() != draft_refined.strip():
            await manager.send_json(websocket, {
                "type":         "subtitle_update",
                "utterance_id": utt_id,
                "text":         accurate_refined,
                "source_text":  input_text,
                "src_lang":     src_lang,
                "tgt_lang":     tgt_indic,
                "trans_time_ms": int(trans_time * 1000),
                "total_time_ms": int((result.get("stt_time", 0) + trans_time) * 1000),
            })
            pl_logger.info("[Window] subtitle_update sent for %s", utt_id)

        # Use accurate_refined as the DB record (most correct version)
        final_text = accurate_refined

        # ── Ensure encryption completed (was running in parallel) ──────────────
        if enc_task is not None:
            await enc_task

        # ── Save to DB (Fix 9: failures are logged with traceback) ─────────────
        if state.meeting_id:
            def _save():
                try:
                    save_utterance(
                        meeting_id=state.meeting_id,
                        source_text=input_text,
                        translated_text=final_text,
                        source_language=src_lang,
                        target_language=tgt_indic,
                        total_latency_ms=int(total_time * 1000),
                    )
                except Exception:
                    logging.getLogger("ispeak.db").error(
                        "[DB] save_utterance failed for meeting %s",
                        state.meeting_id,
                        exc_info=True,
                    )

            await loop.run_in_executor(None, _save)

        # ── Stream TTS audio chunks ────────────────────────────────────────────
        if ENABLE_TTS:
            tts_chunks = router.chunking_service.split_text_for_tts(final_text)
            total_chunks = len(tts_chunks)
            for idx, chunk in enumerate(tts_chunks, start=1):
                router.tts_input_queue.put({
                    "text":         chunk,
                    "chunk_index":  idx,
                    "total_chunks": total_chunks,
                })
        else:
            total_chunks = 0

        chunks_sent = 0
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
            "latency_ms":   round(total_time * 1000),
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

