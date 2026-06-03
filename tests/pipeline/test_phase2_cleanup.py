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