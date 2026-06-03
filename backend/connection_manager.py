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
