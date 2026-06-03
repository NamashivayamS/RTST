import asyncio
import websockets

async def test():

    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:

        message = "Hello Local WebSocket"

        await websocket.send(message)

        response = await websocket.recv()

        print("Sent:", message)
        print("Received:", response)

asyncio.run(test())