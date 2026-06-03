# Consolidated Test Files

This file contains copies of all files under the `tests/` folder, with filenames and their code.

---

## tests/vad/test_vad.py

```python
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import torch

model = load_silero_vad()

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St1.wav"

wav, sr = sf.read(audio_path)

# Convert to float32 tensor
wav = torch.tensor(wav, dtype=torch.float32)

speech_timestamps = get_speech_timestamps(
	wav,
	model,
	sampling_rate=sr,
	return_seconds=True
)

print("\nSpeech Segments:\n")

for segment in speech_timestamps:
	print(segment)
```

---

## tests/tts/test_xtts.py

```python
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
```

---

## tests/translation/test_translation.py

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch
import time

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
	model_name,
	trust_remote_code=True
)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
	model_name,
	trust_remote_code=True,
	torch_dtype=torch.float16
).to(DEVICE)

print("Loading IndicProcessor...")

ip = IndicProcessor(inference=True)

# Input sentence
input_text = "Hello friends"

# Preprocess
batch = ip.preprocess_batch(
	[input_text],
	src_lang="eng_Latn",
	tgt_lang="tam_Taml"
)

inputs = tokenizer(
	batch,
	truncation=True,
	padding="longest",
	return_tensors="pt"
).to(DEVICE)

start = time.time()

with torch.no_grad():
	generated_tokens = model.generate(
		**inputs,
		max_new_tokens=50
	)

generated_tokens = generated_tokens.cpu()

translated = tokenizer.batch_decode(
	generated_tokens,
	skip_special_tokens=True
)

# Postprocess
translated = ip.postprocess_batch(
	translated,
	lang="tam_Taml"
)

end = time.time()

print("\nTranslated Text:")
print(translated[0])

print(f"\nTime Taken: {end-start:.2f} seconds")
```

---

## tests/translation/test_punctuation.py

```python
from deepmultilingualpunctuation import PunctuationModel

print("Loading punctuation model...")

model = PunctuationModel()

text = "bro inniku namma project demo panrom"

result = model.restore_punctuation(text)

print("\nPunctuated Text:")
print(result)
```

---

## tests/translation/test_muril.py

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "google/muril-base-cased"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

model = AutoModel.from_pretrained(model_name)

text = "Meeting postpone panniduvom"

inputs = tokenizer(
	text,
	return_tensors="pt",
	padding=True,
	truncation=True
)

with torch.no_grad():
	outputs = model(**inputs)

print("\nEmbedding Shape:")
print(outputs.last_hidden_state.shape)
```

---

## tests/stt/test_whisper.py

```python
from faster_whisper import WhisperModel
import time

model_size = "medium"

model = WhisperModel(
	model_size,
	device="cuda",
	compute_type="float16"
)

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St2.wav"

start = time.time()

segments, info = model.transcribe(
	audio_path,
	beam_size=5
)

print("Detected language:", info.language)

print("\nTranscription:")

for segment in segments:
	print(segment.text)

end = time.time()

print(f"\nTime Taken: {end-start:.2f} seconds")
```

---

## tests/backend/test_websocket_server.py

```python
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

	await websocket.accept()

	while True:
		data = await websocket.receive_text()

		print("Received:", data)

		await websocket.send_text(f"Echo: {data}")

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

## tests/backend/test_websocket_client.py

```python
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
```

---

## tests/backend/test_websocket.py

```python
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
```

---

## tests/backend/test_uvicorn.py

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
	return {"message": "Uvicorn Working"}
```

---

## tests/backend/test_fastapi_import.py

```python
from fastapi import FastAPI

app = FastAPI()

print("FastAPI Installed Successfully")
```

---

## tests/audio/test_recording.py

```python
import sounddevice as sd
from scipy.io.wavfile import write
sd.default.latency = 'low'

fs = 48000
seconds = 5

print("Recording started... Speak now.")

audio = sd.rec(
	int(seconds * fs),
	samplerate=fs,
	channels=1,
	dtype='float32',
	device=15
)

sd.wait()

write("test_recording.wav", fs, audio)

print("Recording completed!")
print("Saved as test_recording.wav")
```

---

## tests/audio/test_microphone.py

```python
import sounddevice as sd

print(sd.query_devices())
```

---

## tests/audio/test_ffmpeg.py

```python
import ffmpeg

print("FFmpeg Python Working")
```

