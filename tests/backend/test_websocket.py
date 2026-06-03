import asyncio
import websockets

async def test():
    uri = "wss://echo.websocket.events"

    async with websockets.connect(uri) as websocket:

        message = "Hello WebSocket"

        await websocket.send(message)

        response = await websocket.recv()

        print("Sent:", message)
        print("Received:", response)

asyncio.run(test())