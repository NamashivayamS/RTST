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
        text = text.strip()
        if not text:
            return text

        words = text.split()
        word_count = len(words)

        # ── Rule 1: Already ends with sentence punctuation ────────────────────
        # Whisper medium with temperature=0.0 almost always adds . ? ! correctly.
        # If the last character is a sentence ender, trust Whisper entirely.
        sentence_enders = {'.', '!', '?'}
        if text[-1] in sentence_enders:
            return text   # skip model — 200ms saved

        # ── Rule 2: Very short utterance (≤3 words) ───────────────────────────
        # "Yes" / "No" / "Okay" / "Sure" — neural model is overkill.
        # Append a period and return immediately.
        if word_count <= 3:
            return text + '.'   # skip model — 200ms saved

        # ── Rule 3: Medium utterance (4–8 words), no sentence ender ──────────
        # e.g. "Well I think so" or "Maybe not exactly right"
        # Whisper dropped the period OR there's a missing comma after "Well".
        # Short enough that the model runs fast (~80ms). Worth running.
        if word_count <= 8:
            print(f"[Punctuation] Running model (Rule 3 — {word_count} words, no ender)")
            return self.restore(text)

        # ── Rule 4: Long utterance, NO internal punctuation at all ───────────
        # e.g. "I went to the store and I bought milk bread eggs butter and cheese"
        # Whisper missed everything. The model will genuinely help here.
        has_internal_punct = any(ch in text for ch in {',', ';', ':', '-'})
        if not has_internal_punct:
            print(f"[Punctuation] Running model (Rule 4 — {word_count} words, no punctuation)")
            return self.restore(text)

        # ── Rule 5: Long utterance WITH internal punctuation, no ender ───────
        # e.g. "I visited Chennai, Coimbatore, and Trichy"
        # Whisper got the commas right but dropped the final period.
        # Just append a period — no need to run the model.
        return text + '.'   # skip model — 200ms saved


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
