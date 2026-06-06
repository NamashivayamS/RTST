import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.corrections.correction_engine import apply_corrections
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS


class CorrectionService:
    """
    Post-STT text correction service.

    Language-aware: Tamil/Tanglish corrections only applied when source
    language is Tamil ('ta'). English input returned as-is.
    """

    def __init__(self):
        print("CorrectionService initialized and ready.")

    def correct(self, text: str, language: str = "ta") -> str:
        if not text.strip():
            return text
        if language == "ta":
            return apply_corrections(text)   # Tamil + Tanglish + proper nouns
        if language == "en":
            # English only gets proper noun corrections, not Tamil script substitutions
            from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS
            corrected = text.strip()
            for wrong, right in PROPER_NOUN_CORRECTIONS.items():
                corrected = corrected.replace(wrong, right)
            return corrected
        return text.strip()

    def correct_tamil_only(self, text: str) -> str:
        text = text.strip()
        corrected = text
        for wrong, correct in TAMIL_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected

    def correct_tanglish_only(self, text: str) -> str:
        text = text.strip()
        corrected = text
        for wrong, correct in TANGLISH_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected


if __name__ == "__main__":
    service = CorrectionService()
    tamil_raw = "நேத்து மீட்டிங்ல என்னடா நடந்துச்சு"
    english_raw = "  hello friends how are you  "
    print("Tamil correction:")
    print(f"  IN : {tamil_raw}")
    print(f"  OUT: {service.correct(tamil_raw, language='ta')}")
    print("\nEnglish (no-op):")
    print(f"  IN : {english_raw!r}")
    print(f"  OUT: {service.correct(english_raw, language='en')!r}")
