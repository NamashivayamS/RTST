from faster_whisper import WhisperModel
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRIMARY_MODEL = "medium"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMIL_MODEL_PATH = os.path.join(PROJECT_ROOT, "whisper-tamil-medium-ct2")

print(f"\nLoading Primary Whisper Model ({PRIMARY_MODEL} on {DEVICE})...")

try:
    whisper_model = WhisperModel(
        PRIMARY_MODEL,
        device=DEVICE,
        compute_type="int8"
    )
except Exception as e:
    print(f"Error loading primary model: {e}. Downloading '{PRIMARY_MODEL}'...")
    whisper_model = WhisperModel(
        PRIMARY_MODEL,
        device=DEVICE,
        compute_type="int8",
        local_files_only=False
    )

print("Primary Model Loaded Successfully!")

print(f"\nLoading specialized Tamil Whisper model...")
try:
    if os.path.isdir(TAMIL_MODEL_PATH):
        tamil_whisper_model = WhisperModel(
            TAMIL_MODEL_PATH,
            device=DEVICE,
            compute_type="int8"
        )
        print("Tamil Model Loaded Successfully!")
    else:
        tamil_whisper_model = None
        print(f"Custom Tamil model not found at {TAMIL_MODEL_PATH}. Dual-model routing disabled.")
except Exception as e:
    print(f"Failed to load Tamil model: {e}")
    tamil_whisper_model = None
