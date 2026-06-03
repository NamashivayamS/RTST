from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch
import time

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(DEVICE)

print("Loading IndicProcessor...")

ip = IndicProcessor(inference=True)

# Input sentence
input_text = "Hello friends"

# Preprocess
batch = ip.preprocess_batch(
    [input_text],
    src_lang="eng_Latn",
    tgt_lang="tam_Taml"
)

inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt"
).to(DEVICE)

start = time.time()

with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=50
    )

generated_tokens = generated_tokens.cpu()

translated = tokenizer.batch_decode(
    generated_tokens,
    skip_special_tokens=True
)

# Postprocess
translated = ip.postprocess_batch(
    translated,
    lang="tam_Taml"
)

end = time.time()

print("\nTranslated Text:")
print(translated[0])

print(f"\nTime Taken: {end-start:.2f} seconds")