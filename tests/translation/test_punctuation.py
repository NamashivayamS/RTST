from deepmultilingualpunctuation import PunctuationModel

print("Loading punctuation model...")

model = PunctuationModel()

text = "bro inniku namma project demo panrom"

result = model.restore_punctuation(text)

print("\nPunctuated Text:")
print(result)