import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit.processor import (
    IndicProcessor
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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

print("\nLoading IndicProcessor...")

ip = IndicProcessor(
    inference=True
)

print("IndicProcessor Loaded Successfully!")