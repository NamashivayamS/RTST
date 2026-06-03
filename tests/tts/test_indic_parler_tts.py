import torch
import soundfile as sf

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor
)

from parler_tts import (
    ParlerTTSForConditionalGeneration
)

# =========================
# DEVICE SETUP
# =========================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing Device: {DEVICE}")

# =========================
# MODEL NAME               
# =========================

MODEL_NAME = (
    "ai4bharat/indic-parler-tts"
)

# =========================
# LOAD MODEL
# =========================

print("\nLoading Indic Parler-TTS...")

model = (
    ParlerTTSForConditionalGeneration
    .from_pretrained(MODEL_NAME)
    .to(DEVICE)
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

feature_extractor = (
    AutoFeatureExtractor
    .from_pretrained(MODEL_NAME)
)

print(
    "Indic Parler-TTS "
    "Loaded Successfully!"
)

# =========================
# INPUT TEXT
# =========================

text = (
    "வணக்கம் அனைவருக்கும். இது ஒரு தமிழ் உரை-குரல் மாற்று அமைப்பின் சோதனை ஆகும். இந்த அமைப்பு தமிழ் மொழியில் உள்ள சொற்களை தெளிவாகவும் இயல்பாகவும் உச்சரிக்கிறதா என்பதை நாம் இப்போது பரிசோதிக்கிறோம்."
)

# =========================
# SPEAKER DESCRIPTION      
# =========================

description = (
    "A native Tamil male speaker. "
    "Clear and fluent Tamil pronunciation. "
    "Professional studio recording. "
    "Consistent volume. "
    "No background noise. "
    "Natural Tamil accent."
)

# =========================
# TOKENIZATION
# =========================

input_ids = tokenizer(
    description,
    return_tensors="pt"
).input_ids.to(DEVICE)

prompt_input_ids = tokenizer(
    text,
    return_tensors="pt"
).input_ids.to(DEVICE)

# =========================
# GENERATE AUDIO
# =========================

print("\nGenerating Tamil Speech...")

generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids
)

audio_arr = (
    generation.cpu()
    .numpy()
    .squeeze()
)

# =========================
# SAVE AUDIO
# =========================

output_path = (
    "output_indic_parler_tts.wav"
)

sf.write(
    output_path,
    audio_arr,
    feature_extractor.sampling_rate
)

# =========================
# FINAL OUTPUT
# =========================

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Saved File: {output_path}")