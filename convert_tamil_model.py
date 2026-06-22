"""
One-time script to convert vasista22/whisper-tamil-medium to CTranslate2 format.
Patches around the transformers 4.46+ dtype bug by monkey-patching the loader.
"""
import os
import sys

MODEL_NAME = "vasista22/whisper-tamil-medium"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-tamil-medium-ct2")

if os.path.isdir(OUTPUT_DIR) and os.path.exists(os.path.join(OUTPUT_DIR, "model.bin")):
    print(f"Already converted: {OUTPUT_DIR}")
    sys.exit(0)

# ── Monkey-patch to fix the dtype bug ─────────────────────────────────────────
from transformers import WhisperForConditionalGeneration

_original_init = WhisperForConditionalGeneration.__init__

def _patched_init(self, config, *args, **kwargs):
    kwargs.pop("dtype", None)  # Remove the problematic kwarg
    _original_init(self, config, *args, **kwargs)

WhisperForConditionalGeneration.__init__ = _patched_init
# ──────────────────────────────────────────────────────────────────────────────

import ctranslate2

print(f"Converting {MODEL_NAME} -> CTranslate2 int8...")
print(f"Output: {OUTPUT_DIR}")

converter = ctranslate2.converters.TransformersConverter(MODEL_NAME)
converter.convert(
    output_dir=OUTPUT_DIR,
    quantization="int8",
    force=True,
)

# Copy tokenizer files from HF cache
from huggingface_hub import snapshot_download
import shutil

model_path = snapshot_download(MODEL_NAME, local_files_only=True)
for fname in ["tokenizer.json", "preprocessor_config.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt",
              "added_tokens.json", "normalizer.json"]:
    src = os.path.join(model_path, fname)
    if os.path.exists(src):
        shutil.copy2(src, OUTPUT_DIR)
        print(f"  Copied {fname}")

print(f"\nDone! Tamil model ready at: {OUTPUT_DIR}")
