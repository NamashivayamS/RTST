from transformers import AutoTokenizer, AutoModel
import torch

model_name = "google/muril-base-cased"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

model = AutoModel.from_pretrained(model_name)

text = "Meeting postpone panniduvom"

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    outputs = model(**inputs)

print("\nEmbedding Shape:")
print(outputs.last_hidden_state.shape)