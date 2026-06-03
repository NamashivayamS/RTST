import sys
sys.stdout.reconfigure(encoding='utf-8')

from TTS.api import TTS
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)

print("\nLoading XTTS model...")

start_load = time.time()

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
).to(device)

end_load = time.time()

print(f"\nModel Loaded in {end_load - start_load:.2f} seconds")

text = "Hello friends, welcome to our realtime speech translation system."

print("\nGenerating speech...")

start = time.time()

tts.tts_to_file(
    text=text,
    speaker="Ana Florence",
    language="en",
    file_path="output_xtts.wav"
)

end = time.time()

print("\nAudio generated successfully!")

print(f"\nGeneration Time: {end-start:.2f} seconds")