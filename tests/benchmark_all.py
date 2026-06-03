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
