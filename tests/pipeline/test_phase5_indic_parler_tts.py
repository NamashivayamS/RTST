import sys
import os
import time
import torch

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from models.whisper_model import (
    whisper_model
)

from models.punctuation_model import (
    punctuation_model
)

from models.indictrans_model import (
    tokenizer,
    translation_model,
    ip
)

from models.parler_tts_model import (
    tts_model,
    tts_tokenizer,
    feature_extractor
)

print(
    next(tts_model.parameters()).device
)

import soundfile as sf


from utils.corrections.correction_engine import (
    apply_corrections
)

from utils.translation_refinement.translation_refiner import (
    refine_translation
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
# AUDIO FILE
# =========================

audio_path = (
    r"D:\NEED\Sem\Sem 7\Ramraj Intern"
    r"\RealTimeSpeechTranslator"
    r"\audio\English\E1.wav"
)

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# TRANSCRIPTION
# =========================

print("\nTranscribing Audio...")

stt_start = time.time()

segments, info = whisper_model.transcribe(
    audio_path,
    beam_size=5
)

stt_end = time.time()

# =========================
# LANGUAGE DETECTION
# =========================

print(
    f"\nDetected Language: "
    f"{info.language}"
)

# =========================
# COLLECT RAW TEXT
# =========================

raw_text = ""

print("\nSegments:\n")

for segment in segments:

    print(
        f"[{segment.start:.2f}s "
        f"-> "
        f"{segment.end:.2f}s] "
        f"{segment.text}"
    )

    raw_text += segment.text + " "

raw_text = raw_text.strip()

# =========================
# RAW TRANSCRIPTION
# =========================

print("\n======================")
print("RAW TRANSCRIPTION")
print("======================")

print(raw_text)

# =========================
# CLEANUP
# =========================

cleaned_text = apply_corrections(
    raw_text
)

print("\n======================")
print("CLEANED TEXT")
print("======================")

print(cleaned_text)

# =========================
# PUNCTUATION
# =========================

punctuated_text = (
    punctuation_model.restore_punctuation(
        cleaned_text
    )
)

print("\n======================")
print("PUNCTUATED TEXT")
print("======================")

print(punctuated_text)

# =========================
# TRANSLATION
# =========================

print("\nTranslating Text...")

SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tam_Taml"

batch = [punctuated_text]

translation_start = time.time()

# =========================
# PREPROCESS
# =========================

processed_batch = (
    ip.preprocess_batch(
        batch,
        src_lang=SOURCE_LANG,
        tgt_lang=TARGET_LANG
    )
)

# =========================
# TOKENIZATION
# =========================

inputs = tokenizer(
    processed_batch,
    truncation=True,
    padding="longest",
    return_tensors="pt"
).to(DEVICE)

# =========================
# GENERATION
# =========================

with torch.no_grad():

    generated_tokens = (
        translation_model.generate(
            **inputs,
            max_length=256
        )
    )

# =========================
# DECODE
# =========================

generated_tokens = (
    generated_tokens.cpu().tolist()
)

decoded_batch = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

# =========================
# POSTPROCESS
# =========================

translations = (
    ip.postprocess_batch(
        decoded_batch,
        lang=TARGET_LANG
    )
)

translated_text = translations[0]

# =========================
# TRANSLATION REFINEMENT
# =========================

translated_text = refine_translation(
    translated_text
)

translation_end = time.time()

# =========================
# FINAL TRANSLATED OUTPUT
# =========================

print("\n======================")
print("TRANSLATED TEXT")
print("======================")

print(translated_text)

# =========================
# TEXT TO SPEECH
# =========================

print("\nGenerating Tamil Speech...")

description = (
    "A native Tamil male speaker. "
    "Clear and fluent Tamil pronunciation. "
    "Professional studio recording. "
    "Consistent volume. "
    "Natural Tamil accent."
)

description_inputs = tts_tokenizer(
    description,
    return_tensors="pt",
    padding=True
)

input_ids = (
    description_inputs.input_ids
    .to(DEVICE)
)

attention_mask = (
    description_inputs.attention_mask
    .to(DEVICE)
)

prompt_inputs = tts_tokenizer(
    translated_text,
    return_tensors="pt",
    padding=True
)

prompt_input_ids = (
    prompt_inputs.input_ids
    .to(DEVICE)
)

prompt_attention_mask = (
    prompt_inputs.attention_mask
    .to(DEVICE)
)

tts_start = time.time()

generation = tts_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    prompt_input_ids=prompt_input_ids,
    prompt_attention_mask=prompt_attention_mask
)

tts_end = time.time()

audio_arr = (
    generation.cpu()
    .numpy()
    .squeeze()
)

output_path = (
    "output_tamil.wav"
)

sf.write(
    output_path,
    audio_arr,
    feature_extractor.sampling_rate
)

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Saved File: {output_path}")


print(
    f"\nSTT Time: "
    f"{stt_end - stt_start:.2f}s"
)

print(
    f"Translation Time: "
    f"{translation_end - translation_start:.2f}s"
)

print(
    f"TTS Time: "
    f"{tts_end - tts_start:.2f}s"
)