from faster_whisper import WhisperModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SIZE = "medium"

print(f"\nLoading Faster-Whisper Model ({MODEL_SIZE} on {DEVICE})...")

try:
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type="int8",
        local_files_only=True
    )
except Exception:
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type="int8",
        local_files_only=False
    )

print("Whisper Model Loaded Successfully!")
