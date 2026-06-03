import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.punctuation_model import punctuation_model


class PunctuationService:
    """
    Punctuation restoration using DeepMultilingualPunctuation.
    CPU-bound intentionally. Only applied to English input in the pipeline —
    skipped for Tamil (Latin-script model, adds latency without benefit).
    """

    def __init__(self):
        self.model = punctuation_model
        print("PunctuationService initialized and ready.")

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        return self.model.restore_punctuation(text)

    def restore_if_needed(self, text: str) -> str:
        """Skip if text already has punctuation (e.g. numbers like 3.5 count)."""
        punctuation_chars = set(".!?,;:")
        # Only skip if there's an actual sentence-ending punctuation
        # not just any dot (avoids the 3.5 false positive)
        sentence_enders = set(".!?")
        already_punctuated = any(ch in text for ch in sentence_enders)
        if already_punctuated:
            return text
        return self.restore(text)


if __name__ == "__main__":
    service = PunctuationService()
    samples = [
        "the meeting will start tomorrow morning at eight",
        "hello friends welcome to our translation system",
        "bro inniku namma project demo panrom"
    ]
    for sample in samples:
        result = service.restore(sample)
        print(f"  IN : {sample}")
        print(f"  OUT: {result}\n")
