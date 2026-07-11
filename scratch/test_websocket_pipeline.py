import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app, _RUNTIME_WS_TOKEN
import soundfile as sf
import numpy as np
import time

def test_websocket_end_to_end():
    print("Initializing TestClient and triggering startup events...")
    with TestClient(app) as client:
        print("Startup event completed successfully.")
        
        print("Reading audio file...")
        audio_path = "tests/audio_samples/ref_cropped.wav"
        audio, sr = sf.read(audio_path)
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        print("Connecting to WebSocket...")
        with client.websocket_connect(f"/ws/translate?token={_RUNTIME_WS_TOKEN}") as websocket:
            print("Connected! Sending config...")
            # Send config to set target_lang to English and environment to 'quiet' (so VAD detects speech easily)
            websocket.send_json({
                "action": "config",
                "target_lang": "en",
                "environment": "quiet"
            })
            
            # Give a small pause
            time.sleep(0.5)
            
            # Send audio in chunks of 512 float32 samples
            chunk_size = 512
            print(f"Streaming {len(audio)} samples in chunks of {chunk_size}...")
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # pad last chunk
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                websocket.send_bytes(chunk.tobytes())
                # Sleep briefly to simulate real-time streaming
                time.sleep(0.01)
                
            print("Finished streaming audio! Sending turn-taking silence trigger...")
            # Send silent chunks to trigger VAD silence timeout (TURN_TAKING_SILENCE_SEC = 2.0)
            # 16000 * 2.5 seconds = 40000 samples of silence
            silence_chunk = np.zeros(chunk_size, dtype=np.float32)
            for _ in range(int(16000 * 2.5 / chunk_size)):
                websocket.send_bytes(silence_chunk.tobytes())
                time.sleep(0.005)
                
            # Keep connection open for a few seconds to let the pipeline task execute
            print("Waiting for pipeline response...")
            time.sleep(12)
            
            print("Checking for messages from websocket...")
            # Receive any responses that came back
            try:
                while True:
                    msg = websocket.receive_json()
                    print("Received from server:", msg)
            except Exception as e:
                print("Finished reading messages:", e)

if __name__ == "__main__":
    test_websocket_end_to_end()
