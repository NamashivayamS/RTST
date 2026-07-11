# scratch/test_generalization.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.correction_service import CorrectionService
from services.translation_service import TranslationService

def test():
    corr_service = CorrectionService()
    trans_service = TranslationService()

    test_cases = [
        {
            "desc": "Case 1: Quotative / Complementizer ('saying that' / 'as')",
            "colloquial": "அவர் நாளைக்கு வராரு அப்டினு சொன்னாங்க",
            "expected_formal": "அவர் நாளைக்கு வராரு அப்படி என்று சொன்னாங்க"
        },
        {
            "desc": "Case 2: Appositional / Conjunction ('that is' / 'namely' / 'if so')",
            "colloquial": "அப்டினா நாம நாளைக்கு போகலாமா",
            "expected_formal": "அப்படி என்றால் நாம நாளைக்கு போகலாமா"
        },
        {
            "desc": "Case 3: Manner Adverbial ('like that' / 'in that manner')",
            "colloquial": "ஏன் அப்டி செய்றீங்கனு தெரியல",
            "expected_formal": "ஏன் அப்படி செய்றீங்கனு தெரியல"
        },
        {
            "desc": "Case 4: Appositional ('that is to say') with compound",
            "colloquial": "வேட்டி அப்டின்கிறது ஒரு பாரம்பரியமான உடை",
            "expected_formal": "வேட்டி அப்படிங்கிறது ஒரு பாரம்பரியமான உடை"
        }
    ]

    print("=== STARTING GENERALIZATION TEST ===\n")
    for case in test_cases:
        print(f"--- {case['desc']} ---")
        raw = case["colloquial"]
        
        # 1. No correction translation
        trans_raw = trans_service.translate(raw, src_lang="ta", tgt_lang="en")
        
        # 2. Correction applied
        corrected = corr_service.correct(raw, language="ta")
        trans_corrected = trans_service.translate(corrected, src_lang="ta", tgt_lang="en")
        
        print("Raw Input    :", raw)
        print("Raw Trans    :", trans_raw)
        print("Corrected    :", corrected)
        print("Corr Trans   :", trans_corrected)
        print()

if __name__ == "__main__":
    test()
