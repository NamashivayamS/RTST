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

from faster_whisper import WhisperModel

from deepmultilingualpunctuation import (
    PunctuationModel
)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit.processor import (
    IndicProcessor
)

from TTS.api import TTS

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
# LOAD WHISPER MODEL
# =========================

print("\nLoading Faster-Whisper Model...")

whisper_model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16"
)

print("Whisper Model Loaded Successfully!")

# =========================
# LOAD PUNCTUATION MODEL
# =========================

print("\nLoading Punctuation Model...")

punctuation_model = PunctuationModel()

print("Punctuation Model Loaded Successfully!")

# =========================
# LOAD INDICTRANS2 MODEL
# =========================

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

# =========================
# LOAD INDIC PROCESSOR
# =========================

print("\nLoading IndicProcessor...")

ip = IndicProcessor(
    inference=True
)

print("IndicProcessor Loaded Successfully!")

# =========================
# LOAD XTTS MODEL
# =========================

print("\nLoading XTTS-v2 Model...")

tts = TTS(
    model_name=(
        "tts_models/"
        "multilingual/"
        "multi-dataset/"
        "xtts_v2"
    )
).to(DEVICE)

print("XTTS-v2 Model Loaded Successfully!")

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

segments, info = whisper_model.transcribe(
    audio_path,
    beam_size=5
)

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

speaker_wav_path = (
    "tests/audio_samples/reference.wav"
)

output_path = "output_tamil.wav"

tts.tts_to_file(
    text=translated_text,
    speaker_wav=speaker_wav_path,
    language="ta",
    file_path=output_path
)

# =========================
# TTS OUTPUT
# =========================

print("\n======================")
print("TTS GENERATED")
print("======================")

print(f"Output File: {output_path}")

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nTotal Processing Time: "
    f"{end_time - start_time:.2f} seconds"
)