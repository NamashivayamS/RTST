from faster_whisper import WhisperModel
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We use the specialized Tamil-fine-tuned Whisper Medium model (int8).
# This single model is used for all speech-to-text to avoid RAM/VRAM exhaustion.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMIL_MODEL_PATH = os.path.join(PROJECT_ROOT, "whisper-tamil-medium-ct2")

# Set the primary fallback model name if the custom directory isn't found
FALLBACK_MODEL = "medium"

print(f"\nLoading Whisper Model ({DEVICE})...")

try:
    if os.path.isdir(TAMIL_MODEL_PATH):
        print(f"Loading custom Tamil Whisper model from: {TAMIL_MODEL_PATH}")
        whisper_model = WhisperModel(
            TAMIL_MODEL_PATH,
            device=DEVICE,
            compute_type="int8"
        )
    else:
        print(f"Custom Tamil model not found at {TAMIL_MODEL_PATH}. Loading fallback: {FALLBACK_MODEL}")
        whisper_model = WhisperModel(
            FALLBACK_MODEL,
            device=DEVICE,
            compute_type="int8"
        )
except Exception as e:
    print(f"Error loading model: {e}. Trying to download fallback model '{FALLBACK_MODEL}'...")
    whisper_model = WhisperModel(
        FALLBACK_MODEL,
        device=DEVICE,
        compute_type="int8",
        local_files_only=False
    )

print("Whisper Model Loaded Successfully!")

# Dual-model routing is disabled to prevent RAM/VRAM exhaustion.
tamil_whisper_model = None
