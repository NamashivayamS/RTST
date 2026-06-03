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