
============================================================
FILE: tests\benchmark_all.py
============================================================

"""
LATENCY BENCHMARK — Dell G15 RTX 3050
======================================
SAFE TO RUN — this script:
  - Only uses models already installed in your venv
  - Does NOT install anything new
  - Does NOT modify any existing files or configs
  - Does NOT touch your venv, pip, or any dependencies
  - Only creates 3 small .wav files in your project root (deletable after)
  - If any model fails, it catches the error and continues — nothing breaks

Save as: tests/benchmark_all.py
Run:     python tests/benchmark_all.py
"""

import sys
import time
import torch
import soundfile as sf
import os

sys.stdout.reconfigure(encoding='utf-8')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Change this path if your audio file is in a different location ──
AUDIO_PATH = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St1.wav"

RESULTS = {}

print("=" * 60)
print("DELL G15 RTX 3050 — LATENCY BENCHMARK")
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 60)
print("NOTE: Each model does a warmup run before measuring.")
print("      This gives real steady-state latency, not cold-start.")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. SILERO VAD
# ─────────────────────────────────────────────────────────────
print("\n[1/6] Silero VAD...")
try:
    from silero_vad import load_silero_vad, get_speech_timestamps

    vad_model = load_silero_vad()
    wav, sr = sf.read(AUDIO_PATH)
    wav_tensor = torch.tensor(wav, dtype=torch.float32)

    # Warmup — not measured
    get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sr)

    # Measure 5 runs
    times = []
    for _ in range(5):
        t0 = time.time()
        segments = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sr)
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    RESULTS["Silero VAD"] = f"{avg:.1f} ms"
    print(f"  Segments found : {len(segments)}")
    print(f"  Latency        : {avg:.1f} ms average over 5 runs")

except Exception as e:
    RESULTS["Silero VAD"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# 2. WHISPER MEDIUM  (not large-v3 — keeping VRAM for other models)
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Whisper medium (cuda, float16)...")
try:
    from faster_whisper import WhisperModel

    wmodel = WhisperModel("medium", device="cuda", compute_type="float16")

    # Warmup
    segs, _ = wmodel.transcribe(AUDIO_PATH, beam_size=5)
    _ = [s.text for s in segs]

    # Measure 3 runs
    times = []
    transcription = ""
    lang = ""
    for i in range(3):
        t0 = time.time()
        segs, info = wmodel.transcribe(AUDIO_PATH, beam_size=5)
        text = " ".join([s.text for s in segs])
        times.append(time.time() - t0)
        if i == 0:
            transcription = text.strip()
            lang = info.language

    avg = sum(times) / len(times)
    RESULTS["Whisper medium"] = f"{avg:.2f} sec"
    print(f"  Language       : {lang}")
    print(f"  Transcription  : {transcription}")
    print(f"  Latency        : {avg:.2f} sec average over 3 runs")

    # Free VRAM before next model
    del wmodel
    torch.cuda.empty_cache()

except Exception as e:
    RESULTS["Whisper medium"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# 3. MURIL — Tanglish token-level detection
# ─────────────────────────────────────────────────────────────
print("\n[3/6] MuRIL (token-level language detection)...")
try:
    from transformers import AutoTokenizer, AutoModel

    muril_tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril_mod = AutoModel.from_pretrained("google/muril-base-cased").to("cpu")
    # MuRIL runs on CPU to save VRAM — this is the intended production strategy

    sentences = [
        "Meeting postpone panniduvom, rescheduling panrom",
        "Ennanga pandringa sir, ippo time illai",
        "Budget approve aachu, project start pannalaam"
    ]

    # Warmup
    inp = muril_tok(sentences[0], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        muril_mod(**inp)

    # Measure
    times = []
    for s in sentences:
        inp = muril_tok(s, return_tensors="pt", padding=True, truncation=True)
        t0 = time.time()
        with torch.no_grad():
            out = muril_mod(**inp)
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    RESULTS["MuRIL (CPU)"] = f"{avg:.1f} ms"
    print(f"  Embedding shape: {out.last_hidden_state.shape}")
    print(f"  Latency        : {avg:.1f} ms average over 3 Tanglish sentences")

except Exception as e:
    RESULTS["MuRIL (CPU)"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# 4. INDICTRANS2 — Tamil → English
# ─────────────────────────────────────────────────────────────
print("\n[4/6] IndicTrans2 200M — Tamil → English (cuda, float16)...")
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as ITok
    from IndicTransToolkit import IndicProcessor

    it_tok = ITok.from_pretrained(
        "ai4bharat/indictrans2-indic-en-dist-200M",
        trust_remote_code=True
    )
    it_mod = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-indic-en-dist-200M",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(DEVICE)
    ip = IndicProcessor(inference=True)

    test_inputs = [
        "நாளைக்கு காலை கூட்டம் நடைபெறும்",
        "Meeting postpone panniduvom",
        "Budget approve aachu, project start pannalaam"
    ]

    # Warmup
    b = ip.preprocess_batch([test_inputs[0]], src_lang="tam_Taml", tgt_lang="eng_Latn")
    inp = it_tok(b, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        it_mod.generate(**inp, max_new_tokens=50)

    # Measure
    times = []
    for text in test_inputs:
        b = ip.preprocess_batch([text], src_lang="tam_Taml", tgt_lang="eng_Latn")
        inp = it_tok(b, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            gen = it_mod.generate(**inp, max_new_tokens=50)
        gen = gen.cpu()
        out = it_tok.batch_decode(gen, skip_special_tokens=True)
        out = ip.postprocess_batch(out, lang="eng_Latn")
        times.append(time.time() - t0)
        print(f"  '{text}'")
        print(f"  → '{out[0]}'")

    avg = sum(times) / len(times)
    RESULTS["IndicTrans2 (Tamil→EN)"] = f"{avg:.2f} sec"
    print(f"  Latency        : {avg:.2f} sec average over 3 sentences")

    del it_mod
    torch.cuda.empty_cache()

except Exception as e:
    RESULTS["IndicTrans2 (Tamil→EN)"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# 5. PUNCTUATION MODEL
# ─────────────────────────────────────────────────────────────
print("\n[5/6] DeepMultilingualPunctuation...")
try:
    from deepmultilingualpunctuation import PunctuationModel

    pmodel = PunctuationModel()

    texts = [
        "the meeting will start tomorrow morning at eight",
        "bro inniku namma project demo panrom",
        "hello friends welcome to our translation system"
    ]

    # Warmup
    pmodel.restore_punctuation(texts[0])

    # Measure
    times = []
    for text in texts:
        t0 = time.time()
        result = pmodel.restore_punctuation(text)
        times.append((time.time() - t0) * 1000)
        print(f"  '{text}'")
        print(f"  → '{result}'")

    avg = sum(times) / len(times)
    RESULTS["Punctuation model"] = f"{avg:.1f} ms"
    print(f"  Latency        : {avg:.1f} ms average over 3 sentences")

except Exception as e:
    RESULTS["Punctuation model"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# 6. XTTS-V2
# ─────────────────────────────────────────────────────────────
print("\n[6/6] XTTS-v2 (cuda)...")
try:
    from TTS.api import TTS

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

    texts = [
        "The meeting will start tomorrow morning.",
        "Please review the attached document.",
        "The budget has been approved. We can start the project."
    ]

    # Warmup (first inference is slow due to CUDA kernel compilation)
    tts.tts_to_file(
        text=texts[0],
        speaker="Ana Florence",
        language="en",
        file_path="output_warmup.wav"
    )

    # Measure 3 sentences
    times = []
    for i, text in enumerate(texts):
        t0 = time.time()
        tts.tts_to_file(
            text=text,
            speaker="Ana Florence",
            language="en",
            file_path=f"output_bench_{i}.wav"
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Sentence {i+1} ({len(text.split())} words): {elapsed:.2f}s")

    avg = sum(times) / len(times)
    RESULTS["XTTS-v2 (cuda)"] = f"{avg:.2f} sec"
    print(f"  Latency        : {avg:.2f} sec average over 3 sentences")
    print(f"  Note: Warmup file (output_warmup.wav) and")
    print(f"        bench files (output_bench_0/1/2.wav) created.")
    print(f"        Safe to delete these after benchmarking.")

except Exception as e:
    RESULTS["XTTS-v2 (cuda)"] = f"ERROR: {e}"
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY — Dell G15 RTX 3050 6GB")
print("=" * 60)
print(f"{'Stage':<30} {'Your machine':<18} {'Expected range'}")
print("-" * 60)

expected = {
    "Silero VAD":             "5–20 ms",
    "Whisper medium":         "1.0–2.0 sec",
    "MuRIL (CPU)":            "30–80 ms",
    "IndicTrans2 (Tamil→EN)": "0.3–0.8 sec",
    "Punctuation model":      "100–400 ms",
    "XTTS-v2 (cuda)":         "1.5–3.0 sec",
}

for key, exp in expected.items():
    val = RESULTS.get(key, "Not run")
    print(f"  {key:<28} {val:<18} {exp}")

print("=" * 60)
print("\nTotal pipeline estimate (Tamil→English, GPU):")

# Safe total calculation
try:
    vad_ms   = float(RESULTS.get("Silero VAD","0 ms").replace(" ms","").replace("ERROR","0").split()[0])
    wh_s     = float(RESULTS.get("Whisper medium","0 sec").replace(" sec","").replace("ERROR","0").split()[0])
    muril_ms = float(RESULTS.get("MuRIL (CPU)","0 ms").replace(" ms","").replace("ERROR","0").split()[0])
    it_s     = float(RESULTS.get("IndicTrans2 (Tamil→EN)","0 sec").replace(" sec","").replace("ERROR","0").split()[0])
    punc_ms  = float(RESULTS.get("Punctuation model","0 ms").replace(" ms","").replace("ERROR","0").split()[0])
    tts_s    = float(RESULTS.get("XTTS-v2 (cuda)","0 sec").replace(" sec","").replace("ERROR","0").split()[0])
    delivery_ms = 80
    total = vad_ms/1000 + wh_s + muril_ms/1000 + it_s + punc_ms/1000 + tts_s + delivery_ms/1000
    print(f"  VAD ({vad_ms:.0f}ms) + Whisper ({wh_s:.2f}s) + MuRIL ({muril_ms:.0f}ms) +")
    print(f"  IndicTrans2 ({it_s:.2f}s) + Punct ({punc_ms:.0f}ms) + XTTS ({tts_s:.2f}s) + Delivery (~80ms)")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL: ~{total:.2f} seconds end-to-end on your Dell G15")
except:
    print("  (Could not calculate total — check for errors above)")

print("\nThese are YOUR actual numbers. Use them in the project document.")


============================================================
FILE: tests\backend\test_fastapi_import.py
============================================================

from fastapi import FastAPI

app = FastAPI()

print("FastAPI Installed Successfully")

============================================================
FILE: tests\backend\test_uvicorn.py
============================================================

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Uvicorn Working"}

============================================================
FILE: tests\backend\test_websocket.py
============================================================

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

============================================================
FILE: tests\backend\test_websocket_client.py
============================================================

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

============================================================
FILE: tests\backend\test_websocket_server.py
============================================================

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

============================================================
FILE: tests\pipeline\test_phase1_stt.py
============================================================

from faster_whisper import WhisperModel
import torch
import time

# =========================
# DEVICE SETUP
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {DEVICE}")

# =========================
# LOAD WHISPER MODEL
# =========================

print("\nLoading Faster-Whisper Model...")

model = WhisperModel(
    "medium",
    device=DEVICE,
    compute_type="float16"
)

print("Model Loaded Successfully!")

# =========================
# AUDIO FILE
# =========================

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Dialect_tamil\D1.wav"

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

segments, info = model.transcribe(
    audio_path,
    beam_size=5,
)

# =========================
# DISPLAY LANGUAGE
# =========================

print(f"\nDetected Language: {info.language}")

# =========================
# DISPLAY SEGMENTS
# =========================

full_text = ""

print("\nSegments:\n")

for segment in segments:
    print(
        f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
        f"{segment.text}"
    )

    full_text += segment.text + " "

# =========================
# FINAL TEXT
# =========================

full_text = full_text.strip()

print("\n======================")
print("FINAL TRANSCRIPTION")
print("======================")

print(full_text)

# =========================
# END TIMER
# =========================

end_time = time.time()

print(f"\nInference Time: {end_time - start_time:.2f} seconds")

============================================================
FILE: tests\pipeline\test_phase2_cleanup.py
============================================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from faster_whisper import WhisperModel
import torch
import time

from utils.corrections.correction_engine import apply_corrections

# =========================
# DEVICE SETUP
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {DEVICE}")

# =========================
# LOAD WHISPER MODEL
# =========================

print("\nLoading Faster-Whisper Model...")

model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16"
)

print("Model Loaded Successfully!")

# =========================
# AUDIO FILE
# =========================

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Tanglish\T1.wav"

# Example:
# audio_path = r"audio/Std Tamil/St1.wav"

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

segments, info = model.transcribe(
    audio_path,
    beam_size=5
)

# =========================
# LANGUAGE DETECTION
# =========================

print(f"\nDetected Language: {info.language}")

# =========================
# COLLECT RAW TEXT
# =========================

raw_text = ""

print("\nSegments:\n")

for segment in segments:

    print(
        f"[{segment.start:.2f}s -> "
        f"{segment.end:.2f}s] "
        f"{segment.text}"
    )

    raw_text += segment.text + " "

raw_text = raw_text.strip()

# =========================
# RAW OUTPUT
# =========================

print("\n======================")
print("RAW TRANSCRIPTION")
print("======================")

print(raw_text)

# =========================
# RULE-BASED CLEANUP
# =========================

cleaned_text = apply_corrections(raw_text)

# =========================
# CLEANED OUTPUT
# =========================

print("\n======================")
print("CLEANED TRANSCRIPTION")
print("======================")

print(cleaned_text)

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nProcessing Time: "
    f"{end_time - start_time:.2f} seconds"
)

============================================================
FILE: tests\pipeline\test_phase3_punctuation.py
============================================================

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from faster_whisper import WhisperModel
from deepmultilingualpunctuation import PunctuationModel

import torch
import time

from utils.corrections.correction_engine import apply_corrections

# =========================
# DEVICE SETUP
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {DEVICE}")

# =========================
# LOAD WHISPER MODEL
# =========================

print("\nLoading Faster-Whisper Model...")

whisper_model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16"
)

print("Whisper Model Loaded Successfully!")

# =========================
# LOAD PUNCTUATION MODEL
# =========================

print("\nLoading Punctuation Model...")

punctuation_model = PunctuationModel()

print("Punctuation Model Loaded Successfully!")

# =========================
# AUDIO FILE
# =========================

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St3.wav"

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

segments, info = whisper_model.transcribe(
    audio_path,
    beam_size=5
)

# =========================
# LANGUAGE DETECTION
# =========================

print(f"\nDetected Language: {info.language}")

# =========================
# COLLECT RAW TEXT
# =========================

raw_text = ""

print("\nSegments:\n")

for segment in segments:

    print(
        f"[{segment.start:.2f}s -> "
        f"{segment.end:.2f}s] "
        f"{segment.text}"
    )

    raw_text += segment.text + " "

raw_text = raw_text.strip()

# =========================
# RAW OUTPUT
# =========================

print("\n======================")
print("RAW TRANSCRIPTION")
print("======================")

print(raw_text)

# =========================
# CLEANUP
# =========================

cleaned_text = apply_corrections(raw_text)

print("\n======================")
print("CLEANED TEXT")
print("======================")

print(cleaned_text)

# =========================
# PUNCTUATION RESTORATION
# =========================

punctuated_text = punctuation_model.restore_punctuation(
    cleaned_text
)

# =========================
# FINAL OUTPUT
# =========================

print("\n======================")
print("PUNCTUATED TEXT")
print("======================")

print(punctuated_text)

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nTotal Processing Time: "
    f"{end_time - start_time:.2f} seconds"
)

============================================================
FILE: tests\pipeline\test_phase4_translation.py
============================================================

import sys
import os
import time
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from faster_whisper import WhisperModel

from deepmultilingualpunctuation import (
    PunctuationModel
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit.processor import (
    IndicProcessor
)

from TTS.api import TTS

from utils.corrections.correction_engine import (
    apply_corrections
)

from utils.translation_refinement.translation_refiner import (
    refine_translation
)

# =========================
# DEVICE SETUP
# =========================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# =========================
# LOAD WHISPER MODEL
# =========================

print("\nLoading Faster-Whisper Model...")

whisper_model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16"
)

print("Whisper Model Loaded Successfully!")

# =========================
# LOAD PUNCTUATION MODEL
# =========================

print("\nLoading Punctuation Model...")

punctuation_model = PunctuationModel()

print("Punctuation Model Loaded Successfully!")

# =========================
# LOAD INDICTRANS2 MODEL
# =========================

print("\nLoading IndicTrans2 Model...")

MODEL_NAME = (
    "ai4bharat/"
    "indictrans2-en-indic-dist-200M"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

translation_model = (
    AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(DEVICE)
)

print("IndicTrans2 Model Loaded Successfully!")

# =========================
# LOAD INDIC PROCESSOR
# =========================

print("\nLoading IndicProcessor...")

ip = IndicProcessor(
    inference=True
)

print("IndicProcessor Loaded Successfully!")

# =========================
# LOAD XTTS MODEL
# =========================

print("\nLoading XTTS-v2 Model...")

tts = TTS(
    model_name=(
        "tts_models/"
        "multilingual/"
        "multi-dataset/"
        "xtts_v2"
    )
).to(DEVICE)

print("XTTS-v2 Model Loaded Successfully!")

# =========================
# AUDIO FILE
# =========================

audio_path = (
    r"D:\NEED\Sem\Sem 7\Ramraj Intern"
    r"\RealTimeSpeechTranslator"
    r"\audio\English\E1.wav"
)

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

segments, info = whisper_model.transcribe(
    audio_path,
    beam_size=5
)

# =========================
# LANGUAGE DETECTION
# =========================

print(
    f"\nDetected Language: "
    f"{info.language}"
)

# =========================
# COLLECT RAW TEXT
# =========================

raw_text = ""

print("\nSegments:\n")

for segment in segments:

    print(
        f"[{segment.start:.2f}s "
        f"-> "
        f"{segment.end:.2f}s] "
        f"{segment.text}"
    )

    raw_text += segment.text + " "

raw_text = raw_text.strip()

# =========================
# RAW TRANSCRIPTION
# =========================

print("\n======================")
print("RAW TRANSCRIPTION")
print("======================")

print(raw_text)

# =========================
# CLEANUP
# =========================

cleaned_text = apply_corrections(
    raw_text
)

print("\n======================")
print("CLEANED TEXT")
print("======================")

print(cleaned_text)

# =========================
# PUNCTUATION
# =========================

punctuated_text = (
    punctuation_model.restore_punctuation(
        cleaned_text
    )
)

print("\n======================")
print("PUNCTUATED TEXT")
print("======================")

print(punctuated_text)

# =========================
# TRANSLATION
# =========================

print("\nTranslating Text...")

SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tam_Taml"

batch = [punctuated_text]

# =========================
# PREPROCESS
# =========================

processed_batch = (
    ip.preprocess_batch(
        batch,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG
    )
)

# =========================
# TOKENIZATION
# =========================

inputs = tokenizer(
    processed_batch,
    truncation=True,
    padding="longest",
    return_tensors="pt"
).to(DEVICE)

# =========================
# GENERATION
# =========================

with torch.no_grad():

    generated_tokens = (
        translation_model.generate(
            **inputs,
            max_length=256
        )
    )

# =========================
# DECODE
# =========================

generated_tokens = (
    generated_tokens.cpu().tolist()
)

decoded_batch = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

# =========================
# POSTPROCESS
# =========================

translations = (
    ip.postprocess_batch(
        decoded_batch,
        lang=TARGET_LANG
    )
)

translated_text = translations[0]

# =========================
# TRANSLATION REFINEMENT
# =========================

translated_text = refine_translation(
    translated_text
)

# =========================
# FINAL TRANSLATED OUTPUT
# =========================

print("\n======================")
print("TRANSLATED TEXT")
print("======================")

print(translated_text)

# =========================
# TEXT TO SPEECH
# =========================

print("\nGenerating Tamil Speech...")

speaker_wav_path = (
    "tests/audio_samples/reference.wav"
)

output_path = "output_tamil.wav"

tts.tts_to_file(
    text=translated_text,
    speaker_wav=speaker_wav_path,
    language="ta",
    file_path=output_path
)

# =========================
# TTS OUTPUT
# =========================

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Output File: {output_path}")

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nTotal Processing Time: "
    f"{end_time - start_time:.2f} seconds"
)

============================================================
FILE: tests\pipeline\test_phase5_indic_parler_tts.py
============================================================

import sys
import os
import time
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from models.whisper_model import (
    whisper_model
)

from models.punctuation_model import (
    punctuation_model
)

from models.indictrans_model import (
    tokenizer,
    translation_model,
    ip
)

from models.parler_tts_model import (
    tts_model,
    tts_tokenizer,
    feature_extractor
)

print(
    next(tts_model.parameters()).device
)

import soundfile as sf


from utils.corrections.correction_engine import (
    apply_corrections
)

from utils.translation_refinement.translation_refiner import (
    refine_translation
)

# =========================
# DEVICE SETUP
# =========================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")



# =========================
# AUDIO FILE
# =========================

audio_path = (
    r"D:\NEED\Sem\Sem 7\Ramraj Intern"
    r"\RealTimeSpeechTranslator"
    r"\audio\English\E1.wav"
)

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

stt_start = time.time()

segments, info = whisper_model.transcribe(
    audio_path,
    beam_size=5
)

stt_end = time.time()

# =========================
# LANGUAGE DETECTION
# =========================

print(
    f"\nDetected Language: "
    f"{info.language}"
)

# =========================
# COLLECT RAW TEXT
# =========================

raw_text = ""

print("\nSegments:\n")

for segment in segments:

    print(
        f"[{segment.start:.2f}s "
        f"-> "
        f"{segment.end:.2f}s] "
        f"{segment.text}"
    )

    raw_text += segment.text + " "

raw_text = raw_text.strip()

# =========================
# RAW TRANSCRIPTION
# =========================

print("\n======================")
print("RAW TRANSCRIPTION")
print("======================")

print(raw_text)

# =========================
# CLEANUP
# =========================

cleaned_text = apply_corrections(
    raw_text
)

print("\n======================")
print("CLEANED TEXT")
print("======================")

print(cleaned_text)

# =========================
# PUNCTUATION
# =========================

punctuated_text = (
    punctuation_model.restore_punctuation(
        cleaned_text
    )
)

print("\n======================")
print("PUNCTUATED TEXT")
print("======================")

print(punctuated_text)

# =========================
# TRANSLATION
# =========================

print("\nTranslating Text...")

SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tam_Taml"

batch = [punctuated_text]

translation_start = time.time()

# =========================
# PREPROCESS
# =========================

processed_batch = (
    ip.preprocess_batch(
        batch,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG
    )
)

# =========================
# TOKENIZATION
# =========================

inputs = tokenizer(
    processed_batch,
    truncation=True,
    padding="longest",
    return_tensors="pt"
).to(DEVICE)

# =========================
# GENERATION
# =========================

with torch.no_grad():

    generated_tokens = (
        translation_model.generate(
            **inputs,
            max_length=256
        )
    )

# =========================
# DECODE
# =========================

generated_tokens = (
    generated_tokens.cpu().tolist()
)

decoded_batch = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

# =========================
# POSTPROCESS
# =========================

translations = (
    ip.postprocess_batch(
        decoded_batch,
        lang=TARGET_LANG
    )
)

translated_text = translations[0]

# =========================
# TRANSLATION REFINEMENT
# =========================

translated_text = refine_translation(
    translated_text
)

translation_end = time.time()

# =========================
# FINAL TRANSLATED OUTPUT
# =========================

print("\n======================")
print("TRANSLATED TEXT")
print("======================")

print(translated_text)

# =========================
# TEXT TO SPEECH
# =========================

print("\nGenerating Tamil Speech...")

description = (
    "A native Tamil male speaker. "
    "Clear and fluent Tamil pronunciation. "
    "Professional studio recording. "
    "Consistent volume. "
    "Natural Tamil accent."
)

description_inputs = tts_tokenizer(
    description,
    return_tensors="pt",
    padding=True
)

input_ids = (
    description_inputs.input_ids
    .to(DEVICE)
)

attention_mask = (
    description_inputs.attention_mask
    .to(DEVICE)
)

prompt_inputs = tts_tokenizer(
    translated_text,
    return_tensors="pt",
    padding=True
)

prompt_input_ids = (
    prompt_inputs.input_ids
    .to(DEVICE)
)

prompt_attention_mask = (
    prompt_inputs.attention_mask
    .to(DEVICE)
)

tts_start = time.time()

generation = tts_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    prompt_input_ids=prompt_input_ids,
    prompt_attention_mask=prompt_attention_mask
)

tts_end = time.time()

audio_arr = (
    generation.cpu()
    .numpy()
    .squeeze()
)

output_path = (
    "output_tamil.wav"
)

sf.write(
    output_path,
    audio_arr,
    feature_extractor.sampling_rate
)

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Saved File: {output_path}")


print(
    f"\nSTT Time: "
    f"{stt_end - stt_start:.2f}s"
)

print(
    f"Translation Time: "
    f"{translation_end - translation_start:.2f}s"
)

print(
    f"TTS Time: "
    f"{tts_end - tts_start:.2f}s"
)

============================================================
FILE: tests\stt\test_whisper.py
============================================================

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

============================================================
FILE: tests\translation\test_muril.py
============================================================

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

============================================================
FILE: tests\translation\test_punctuation.py
============================================================

from deepmultilingualpunctuation import PunctuationModel

print("Loading punctuation model...")

model = PunctuationModel()

text = "bro inniku namma project demo panrom"

result = model.restore_punctuation(text)

print("\nPunctuated Text:")
print(result)

============================================================
FILE: tests\translation\test_translation.py
============================================================

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

============================================================
FILE: tests\tts\test_indic_f5_tts.py
============================================================

import time
import soundfile as sf
import torch
from transformers import AutoModel
import numpy as np
import inspect

# =========================
# DEVICE SETUP
# =========================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# =========================
# LOAD MODEL
# =========================

print("\nLoading AI4Bharat IndicF5 Model...")

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
    remove_sil=False
).to(DEVICE)

model.config.remove_sil = False

print("\n======================")
print("MODEL TYPE")
print("======================")

print(type(model))
print(model)

print(
    "\nModel Loaded:",
    model.__class__.__name__
)

print("\n======================")
print("AVAILABLE TTS METHODS")
print("======================")

methods = [
    name
    for name in dir(model)
    if "infer" in name.lower()
    or "generate" in name.lower()
    or "tts" in name.lower()
]

for method in methods:
    print(method)

print("IndicF5 Loaded Successfully!")

# =========================
# PATH MAPPING & PARAMS
# =========================

ref_audio = "tests/audio_samples/ref_cropped.wav"

ref_text = "தாமிரபரணி ஆற்றின் கரையுரங்களில் வசிக்கும்."

gen_text = (
    " வணக்கம் அனைவருக்கும். இன்று நாம் பேச்சு உருவாக்க அமைப்பை சோதிக்கிறோம்."
)

audio, sr = sf.read(ref_audio)

print(
    f"\nReference Sample Rate: {sr} Hz"
)
print(
    f"Reference Duration: "
    f"{len(audio)/sr:.2f} seconds"
)

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# GENERATE
# =========================

print("\nGenerating Tamil Speech...")

from f5_tts.infer.utils_infer import infer_process

# We BYPASS preprocess_ref_audio_text completely! It cuts off soft Tamil syllables.
cleaned_ref_audio = ref_audio
cleaned_ref_text = ref_text

cleaned_audio, cleaned_sr = sf.read(cleaned_ref_audio)
print(f"Raw Reference Duration: {len(cleaned_audio)/cleaned_sr:.2f} seconds")
print(f"Preprocessed Reference Text: {cleaned_ref_text.encode('ascii', 'backslashreplace').decode('ascii')}")

from f5_tts.model.utils import convert_char_to_pinyin
raw_combined = ref_text + gen_text
token_list = convert_char_to_pinyin([raw_combined])[0]
print(f"Total tokens count: {len(token_list)}")
print(f"First 50 tokens: {repr(token_list[:50]).encode('ascii', 'backslashreplace').decode('ascii')}")
print(f"Last 50 tokens: {repr(token_list[-50:]).encode('ascii', 'backslashreplace').decode('ascii')}")

# Call infer_process with natural duration and ideal pacing
audio_arr, final_sample_rate, _ = infer_process(
    cleaned_ref_audio,
    cleaned_ref_text,
    gen_text,
    model.ema_model,
    model.vocoder,
    mel_spec_type="vocos",
    speed=0.85, # Ideal speed for Tamil byte-density
    device=model.device,
    fix_duration=None # Let the model calculate perfect physical timing
)

print(f"Raw model output type: {type(audio_arr)}")
if hasattr(audio_arr, "shape"):
    print(f"Raw model output shape: {audio_arr.shape}")
else:
    print(f"Raw model output length: {len(audio_arr)}")

# Convert torch tensor to numpy array if returned
if hasattr(audio_arr, "cpu"):
    audio_arr = audio_arr.cpu().numpy()
else:
    audio_arr = np.array(audio_arr)

# Normalize if int16 (infer_process returns float32 directly)
if audio_arr.dtype == np.int16:
    print("Detected raw int16 data, normalizing to float32...")
    audio_arr = audio_arr.astype(np.float32) / 32768.0

print(f"Final output data type: {audio_arr.dtype}, length: {len(audio_arr)} samples")

output_path = "output_f5_tamil.wav"

# IndicF5 generates 24kHz audio
sf.write(
    output_path,
    audio_arr,
    24000
)

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nTTS Time: "
    f"{end_time - start_time:.2f}s"
)
print(f"Saved File: {output_path}")

============================================================
FILE: tests\tts\test_indic_parler_tts.py
============================================================

import torch
import soundfile as sf

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor
)

from parler_tts import (
    ParlerTTSForConditionalGeneration
)

# =========================
# DEVICE SETUP
# =========================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# =========================
# MODEL NAME               
# =========================

MODEL_NAME = (
    "ai4bharat/indic-parler-tts"
)

# =========================
# LOAD MODEL
# =========================

print("\nLoading Indic Parler-TTS...")

model = (
    ParlerTTSForConditionalGeneration
    .from_pretrained(MODEL_NAME)
    .to(DEVICE)
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

feature_extractor = (
    AutoFeatureExtractor
    .from_pretrained(MODEL_NAME)
)

print(
    "Indic Parler-TTS "
    "Loaded Successfully!"
)

# =========================
# INPUT TEXT
# =========================

text = (
    "வணக்கம் அனைவருக்கும். இது ஒரு தமிழ் உரை-குரல் மாற்று அமைப்பின் சோதனை ஆகும். இந்த அமைப்பு தமிழ் மொழியில் உள்ள சொற்களை தெளிவாகவும் இயல்பாகவும் உச்சரிக்கிறதா என்பதை நாம் இப்போது பரிசோதிக்கிறோம்."
)

# =========================
# SPEAKER DESCRIPTION      
# =========================

description = (
    "A native Tamil male speaker. "
    "Clear and fluent Tamil pronunciation. "
    "Professional studio recording. "
    "Consistent volume. "
    "No background noise. "
    "Natural Tamil accent."
)

# =========================
# TOKENIZATION
# =========================

input_ids = tokenizer(
    description,
    return_tensors="pt"
).input_ids.to(DEVICE)

prompt_input_ids = tokenizer(
    text,
    return_tensors="pt"
).input_ids.to(DEVICE)

# =========================
# GENERATE AUDIO
# =========================

print("\nGenerating Tamil Speech...")

generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids
)

audio_arr = (
    generation.cpu()
    .numpy()
    .squeeze()
)

# =========================
# SAVE AUDIO
# =========================

output_path = (
    "output_indic_parler_tts.wav"
)

sf.write(
    output_path,
    audio_arr,
    feature_extractor.sampling_rate
)

# =========================
# FINAL OUTPUT
# =========================

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Saved File: {output_path}")

============================================================
FILE: tests\tts\test_xtts.py
============================================================

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

============================================================
FILE: tests\vad\test_vad.py
============================================================

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
