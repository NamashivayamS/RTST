# download_models.py
"""
Utility script used during Docker image build to pre-download, cache, and bake all 
required neural network weights into the container image filesystem. This enables 
complete offline deployment on the production server without any runtime downloads.
"""

import os
import torch
import shutil
import silero_vad
from faster_whisper import WhisperModel
from transformers import AutoModel
from huggingface_hub import snapshot_download
from speechbrain.inference.classifiers import EncoderClassifier
from ctranslate2.converters import TransformersConverter

# Monkeypatch transformers to handle ctranslate2 converter's old dtype argument
import transformers.modeling_utils
original_from_pretrained = transformers.modeling_utils.PreTrainedModel.from_pretrained.__func__

@classmethod
def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    if 'dtype' in kwargs:
        kwargs['torch_dtype'] = kwargs.pop('dtype')
    # Force low_cpu_mem_usage to False to bypass the check_support_param_buffer_assignment bug in transformers
    kwargs['low_cpu_mem_usage'] = False
    return original_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)

transformers.modeling_utils.PreTrainedModel.from_pretrained = patched_from_pretrained
transformers.modeling_utils.check_support_param_buffer_assignment = lambda *args, **kwargs: False

# Force downloads to resolve online during this build step
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print("=========================================================")
print("Pre-downloading & caching all models for offline Docker image")
print("=========================================================")

print("\n1. Pre-downloading Silero VAD...")
silero_vad.load_silero_vad()

print("\n2. Pre-downloading Faster-Whisper 'medium' model...")
# This downloads and caches the model weights inside the standard Hugging Face cache
medium_dir = snapshot_download("Systran/faster-whisper-medium")

print("\n3. Pre-downloading SpeechBrain Speaker ID Classifier...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)
del classifier
import gc
gc.collect()

print("\n4. Pre-downloading IndicTrans2 Translation Models (CTranslate2 version)...")
# adalat-ai is public and contains both CTranslate2 weights and vocab/model.SRC + vocab/model.TGT SentencePiece files.
snapshot_download("adalat-ai/ct2-rotary-indictrans2-en-indic-dist-200M")
snapshot_download("adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M")
gc.collect()

print("\n5. Pre-downloading Deep Multilingual Punctuation Model...")
# snapshot_download only downloads the files and caches them without loading the model into RAM, saving >2GB RAM
snapshot_download("oliverguhr/fullstop-punctuation-multilang-large")
gc.collect()

print("\n=========================================================")
print("All models successfully cached inside the Docker image!")
print("=========================================================")
