from faster_whisper import WhisperModel
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# int8_float16 uses tensor cores (faster) but needs FP16 support.
# RTX 3050 supports FP16, so this is safe. Falls back to int8 on CPU.
COMPUTE_TYPE = "int8_float16" if DEVICE == "cuda" else "int8"

from config import PRIMARY_WHISPER_MODEL as PRIMARY_MODEL
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMIL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "whisper-tamil-medium-ct2")

print(f"\nLoading Primary Whisper Model ({PRIMARY_MODEL} on {DEVICE})...")

try:
    whisper_model = WhisperModel(
        PRIMARY_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
except Exception as e:
    print(f"Error loading primary model: {e}. Downloading '{PRIMARY_MODEL}'...")
    whisper_model = WhisperModel(
        PRIMARY_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        local_files_only=False
    )

print("Primary Model Loaded Successfully!")

print(f"\nLoading specialized Tamil Whisper model...")
try:
    if os.path.isdir(TAMIL_MODEL_PATH):
        # If config files are missing, copy them from the cached medium model offline
        preprocessor_path = os.path.join(TAMIL_MODEL_PATH, "preprocessor_config.json")
        if not os.path.exists(preprocessor_path):
            try:
                from faster_whisper import download_model
                import shutil
                primary_dir = download_model(PRIMARY_MODEL, local_files_only=True)
                for filename in ["tokenizer.json", "preprocessor_config.json"]:
                    dst_file = os.path.join(TAMIL_MODEL_PATH, filename)
                    if not os.path.exists(dst_file):
                        src_file = os.path.join(primary_dir, filename)
                        if os.path.exists(src_file):
                            shutil.copy(src_file, dst_file)
                            print(f"Copied missing config {filename} to {TAMIL_MODEL_PATH} offline.")
            except Exception as copy_err:
                print(f"Could not copy offline configs to Tamil model path: {copy_err}")

        tamil_whisper_model = WhisperModel(
            TAMIL_MODEL_PATH,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("Tamil Model Loaded Successfully!")
    else:
        tamil_whisper_model = None
        print(f"Custom Tamil model not found at {TAMIL_MODEL_PATH}. Dual-model routing disabled.")
except Exception as e:
    print(f"Failed to load Tamil model: {e}")
    tamil_whisper_model = None
