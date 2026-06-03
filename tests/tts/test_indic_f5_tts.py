import time
import soundfile as sf
import torch
from transformers import AutoModel
import numpy as np
import inspect

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
# LOAD MODEL
# =========================

print("\nLoading AI4Bharat IndicF5 Model...")

model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
    remove_sil=False
).to(DEVICE)

model.config.remove_sil = False

print("\n======================")
print("MODEL TYPE")
print("======================")

print(type(model))
print(model)

print(
    "\nModel Loaded:",
    model.__class__.__name__
)

print("\n======================")
print("AVAILABLE TTS METHODS")
print("======================")

methods = [
    name
    for name in dir(model)
    if "infer" in name.lower()
    or "generate" in name.lower()
    or "tts" in name.lower()
]

for method in methods:
    print(method)

print("IndicF5 Loaded Successfully!")

# =========================
# PATH MAPPING & PARAMS
# =========================

ref_audio = "tests/audio_samples/ref_cropped.wav"

ref_text = "தாமிரபரணி ஆற்றின் கரையுரங்களில் வசிக்கும்."

gen_text = (
    " வணக்கம் அனைவருக்கும். இன்று நாம் பேச்சு உருவாக்க அமைப்பை சோதிக்கிறோம்."
)

audio, sr = sf.read(ref_audio)

print(
    f"\nReference Sample Rate: {sr} Hz"
)
print(
    f"Reference Duration: "
    f"{len(audio)/sr:.2f} seconds"
)

# =========================
# START TIMER
# =========================

start_time = time.time()

# =========================
# GENERATE
# =========================

print("\nGenerating Tamil Speech...")

from f5_tts.infer.utils_infer import infer_process

# We BYPASS preprocess_ref_audio_text completely! It cuts off soft Tamil syllables.
cleaned_ref_audio = ref_audio
cleaned_ref_text = ref_text

cleaned_audio, cleaned_sr = sf.read(cleaned_ref_audio)
print(f"Raw Reference Duration: {len(cleaned_audio)/cleaned_sr:.2f} seconds")
print(f"Preprocessed Reference Text: {cleaned_ref_text.encode('ascii', 'backslashreplace').decode('ascii')}")

from f5_tts.model.utils import convert_char_to_pinyin
raw_combined = ref_text + gen_text
token_list = convert_char_to_pinyin([raw_combined])[0]
print(f"Total tokens count: {len(token_list)}")
print(f"First 50 tokens: {repr(token_list[:50]).encode('ascii', 'backslashreplace').decode('ascii')}")
print(f"Last 50 tokens: {repr(token_list[-50:]).encode('ascii', 'backslashreplace').decode('ascii')}")

# Call infer_process with natural duration and ideal pacing
audio_arr, final_sample_rate, _ = infer_process(
    cleaned_ref_audio,
    cleaned_ref_text,
    gen_text,
    model.ema_model,
    model.vocoder,
    mel_spec_type="vocos",
    speed=0.85, # Ideal speed for Tamil byte-density
    device=model.device,
    fix_duration=None # Let the model calculate perfect physical timing
)

print(f"Raw model output type: {type(audio_arr)}")
if hasattr(audio_arr, "shape"):
    print(f"Raw model output shape: {audio_arr.shape}")
else:
    print(f"Raw model output length: {len(audio_arr)}")

# Convert torch tensor to numpy array if returned
if hasattr(audio_arr, "cpu"):
    audio_arr = audio_arr.cpu().numpy()
else:
    audio_arr = np.array(audio_arr)

# Normalize if int16 (infer_process returns float32 directly)
if audio_arr.dtype == np.int16:
    print("Detected raw int16 data, normalizing to float32...")
    audio_arr = audio_arr.astype(np.float32) / 32768.0

print(f"Final output data type: {audio_arr.dtype}, length: {len(audio_arr)} samples")

output_path = "output_f5_tamil.wav"

# IndicF5 generates 24kHz audio
sf.write(
    output_path,
    audio_arr,
    24000
)

# =========================
# END TIMER
# =========================

end_time = time.time()

print(
    f"\nTTS Time: "
    f"{end_time - start_time:.2f}s"
)
print(f"Saved File: {output_path}")