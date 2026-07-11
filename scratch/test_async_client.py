import asyncio
import os
import re
import sys
import numpy as np
import soundfile as sf
import websockets
import json

def load_session_token():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"SESSION_TOKEN\s*=\s*([^\s]+)", content)
        if match:
            return match.group(1)
    return "dev-insecure-token"

async def stream_audio():
    token = load_session_token()
    uri = f"ws://localhost:8000/ws/translate?token={token}"
    print(f"Connecting to WebSocket URI: {uri}")
    
    audio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "audio_samples", "reference.wav")
    print(f"Reading audio file: {audio_path}")
    audio, sr = sf.read(audio_path)
    audio = audio.astype(np.float32)
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Sending config...")
        await websocket.send(json.dumps({
            "action": "config",
            "target_lang": "en",
            "environment": "quiet"
        }))
        
        await asyncio.sleep(0.5)
        
        chunk_size = 512
        print(f"Streaming {len(audio)} samples in chunks of {chunk_size}...")
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            await websocket.send(chunk.tobytes())
            await asyncio.sleep(0.01)
            
        print("Finished streaming audio! Sending turn-taking silence trigger...")
        silence_chunk = np.zeros(chunk_size, dtype=np.float32)
        # 16000 * 2.5 seconds = 40000 samples of silence
        for _ in range(int(16000 * 2.5 / chunk_size)):
            await websocket.send(silence_chunk.tobytes())
            await asyncio.sleep(0.005)
            
        print("Waiting for responses...")
        # Listen for messages for a maximum of 20 seconds
        done_count = 0
        try:
            while True:
                msg = await asyncio.wait_for(websocket.recv(), timeout=20.0)
                if isinstance(msg, bytes):
                    print(f"Received binary frame ({len(msg)} bytes)")
                else:
                    data = json.loads(msg)
                    print("Received JSON:", data)
                    if data.get("type") == "done":
                        done_count += 1
                        print(f"Done received ({done_count}/2)...")
                        if done_count >= 2:
                            print("All expected done messages received. Exiting...")
                            break
        except asyncio.TimeoutError:
            print("Timeout waiting for messages.")
        except Exception as e:
            print("Connection closed or error:", e)

if __name__ == "__main__":
    # Ensure UTF-8 output
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            
    asyncio.run(stream_audio())
