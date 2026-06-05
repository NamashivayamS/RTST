
============================================================
FILE: backend\connection_manager.py
============================================================

import asyncio
from fastapi import WebSocket
from typing import Optional
from starlette.websockets import WebSocketState


class ConnectionManager:
    """
    Manages active WebSocket client connections.

    Supports multiple simultaneous clients. Each client gets its own
    session identified by the WebSocket object itself.
    """

    def __init__(self):
        # Set of currently connected WebSocket clients
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        """Send a JSON message to a single client."""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                self.disconnect(websocket)
                return False

            await websocket.send_json(data)
            return True
        except Exception as e:
            print(f"[WS] Failed to send to client: {e}")
            self.disconnect(websocket)
            return False

    async def send_bytes(self, websocket: WebSocket, data: bytes):
        """Send raw bytes (audio) to a single client."""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                self.disconnect(websocket)
                return False

            await websocket.send_bytes(data)
            return True
        except Exception as e:
            print(f"[WS] Failed to send bytes to client: {e}")
            self.disconnect(websocket)
            return False

    async def broadcast_json(self, data: dict):
        """Broadcast a JSON message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        for ws in disconnected:
            self.disconnect(ws)


============================================================
FILE: backend\main.py
============================================================

"""
RealTimeSpeechTranslator — FastAPI Backend
==========================================

Endpoints:
  GET  /           → health check
  GET  /status     → pipeline readiness
  WS   /ws/translate → main real-time translation pipeline

WebSocket message protocol:
  Client → Server:
    Binary frames  : raw PCM audio bytes (float32, 16 kHz, mono)
    Text frame     : JSON control message, e.g. {"action": "ping"}

  Server → Client:
    {"type": "subtitle",  "text": "...", "src_lang": "ta", "tgt_lang": "tam_Taml"}
    {"type": "audio",     "chunk_index": 1, "total_chunks": 3,
                          "sample_rate": 24000, "encoding": "pcm_float32"}
    <binary frame>       : raw PCM audio for the preceding "audio" metadata message
    {"type": "done",      "total_chunks": 3}
    {"type": "error",     "message": "..."}
    {"type": "pong"}

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import queue
import sys
import os
import traceback

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the path when running via uvicorn from any cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.connection_manager import ConnectionManager
from services.router_service import RouterService

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RealTimeSpeechTranslator",
    description="Tamil ↔ English real-time speech translation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# RouterService is initialised once at startup.
router: RouterService | None = None


@app.on_event("startup")
async def startup_event():
    global router

    # Guard: only initialise once per process
    if router is not None:
        return

    print("[Backend] Loading pipeline models...")
    loop = asyncio.get_event_loop()
    router = await loop.run_in_executor(None, RouterService)

    # ── Warm up all lazy-loaded models so the first real request is fast ──
    # Without this, the first utterance triggers IndicF5 + IndicTrans2
    # loading mid-request, adding 20-40 seconds of latency.
    print("[Backend] Warming up models...")

    # 1. Warm up both IndicTrans2 translation directions
    await loop.run_in_executor(
        None,
        lambda: router.translation_service.translate("Hello", "en", "tam_Taml")
    )
    await loop.run_in_executor(
        None,
        lambda: router.translation_service.translate("வணக்கம்", "ta", "eng_Latn")
    )

    # 2. Warm up IndicF5 TTS — triggers lazy load and first CUDA kernel compile
    await loop.run_in_executor(
        None,
        lambda: router.tts_service.generate_audio("வணக்கம்.")
    )

    print("[Backend] All models warmed up. Server ready for requests.")


@app.on_event("shutdown")
async def shutdown_event():
    if router:
        router.shutdown()
    print("[Backend] Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "RealTimeSpeechTranslator is running."}


@app.get("/status")
async def pipeline_status():
    return {
        "pipeline_ready": router is not None,
        "active_connections": len(manager.active_connections),
    }


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """
    Main real-time translation WebSocket endpoint.

    The client streams raw audio (binary frames, float32 PCM at 16 kHz).
    For each frame the server:
      1. Runs the full pipeline in a thread pool executor (event loop stays free).
      2. Sends a subtitle JSON message immediately.
      3. Streams audio chunks back as they are generated by the TTS worker.
    """
    await manager.connect(websocket)
    
    if router:
        router.cancel_event.clear()
        
    loop = asyncio.get_event_loop()

    try:
        while True:
            message = await websocket.receive()

            # ── Graceful disconnect handling ───────────────────────────────
            # Without this check, receiving a disconnect message and then
            # calling receive() again raises:
            #   RuntimeError: Cannot call "receive" once a disconnect message
            #   has been received.
            if message.get("type") == "websocket.disconnect":
                break

            # ── Control message (text JSON) ────────────────────────────────
            if "text" in message:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": "Invalid JSON control message."
                    })
                    continue

                if control.get("action") == "ping":
                    await manager.send_json(websocket, {"type": "pong"})
                continue

            # ── Audio frame (binary) ───────────────────────────────────────
            if "bytes" in message:
                raw_bytes = message["bytes"]
                if not raw_bytes:
                    continue

                audio_array = np.frombuffer(raw_bytes, dtype=np.float32)

                try:
                    pipeline_result = await loop.run_in_executor(
                        None,
                        router.process_audio,
                        audio_array
                    )
                except Exception as e:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": f"Pipeline error: {str(e)}"
                    })
                    traceback.print_exc()
                    continue

                total_chunks = pipeline_result["total_chunks"]

                if total_chunks == 0:
                    if pipeline_result["translated_text"]:
                        await manager.send_json(websocket, {
                            "type": "subtitle",
                            "text": pipeline_result["translated_text"],
                            "src_lang": pipeline_result["src_lang"],
                            "tgt_lang": pipeline_result["tgt_lang"],
                        })

                    await manager.send_json(websocket, {
                        "type": "done",
                        "total_chunks": 0
                    })
                    continue


                # ── 1. Send subtitle immediately ───────────────────────────
                await manager.send_json(websocket, {
                    "type":     "subtitle",
                    "text":     pipeline_result["translated_text"],
                    "src_lang": pipeline_result["src_lang"],
                    "tgt_lang": pipeline_result["tgt_lang"],
                })

                # ── 2. Stream audio chunks as the TTS worker generates them ─
                chunks_received = 0
                for _ in range(total_chunks):
                    audio_payload = await loop.run_in_executor(
                        None,
                        lambda: router.get_generated_audio(block=True, timeout=120)
                    )

                    if audio_payload is None:
                        await manager.send_json(websocket, {
                            "type":    "error",
                            "message": f"Audio generation timed out "
                                       f"(chunk {chunks_received + 1}/{total_chunks})."
                        })
                        break

                    chunks_received += 1

                    # Send JSON metadata first, then raw PCM bytes
                    await manager.send_json(websocket, {
                        "type":         "audio",
                        "chunk_index":  audio_payload["chunk_index"],
                        "total_chunks": audio_payload["total_chunks"],
                        "sample_rate":  audio_payload["sample_rate"],
                        "encoding":     "pcm_float32",
                        "text":         audio_payload["text"],
                    })

                    # Avoid unnecessary copy: only cast if not already float32
                    audio = audio_payload["audio"]
                    audio_bytes = audio.tobytes() if audio.dtype == np.float32 \
                                  else audio.astype(np.float32).tobytes()
                    await manager.send_bytes(websocket, audio_bytes)

                # ── 3. Signal utterance complete ───────────────────────────
                await manager.send_json(websocket, {
                    "type":         "done",
                    "total_chunks": chunks_received,
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        traceback.print_exc()
        try:
            await manager.send_json(websocket, {
                "type":    "error",
                "message": f"Unexpected server error: {str(e)}"
            })
        except Exception:
            pass
        manager.disconnect(websocket)

    finally:
        # Always clean up connection on exit, regardless of how we got here
        manager.disconnect(websocket)
        
        # Cancel any ongoing TTS generation and flush pending work
        if router:
            router.cancel_event.set()
            while not router.tts_input_queue.empty():
                try:
                    router.tts_input_queue.get_nowait()
                    router.tts_input_queue.task_done()
                except queue.Empty:
                    break


============================================================
FILE: backend\__init__.py
============================================================

# Marks the backend folder as a python package

