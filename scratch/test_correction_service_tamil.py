# scratch/test_correction_service_tamil.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.correction_service import CorrectionService
from services.translation_service import TranslationService

def test():
    corr_service = CorrectionService()
    trans_service = TranslationService()

    raw_text = "இன்று நம்முடைய இன்டர்வியூ கெஸ்ட் அப்டியினா வேட்டி உலகின் சக்கரவர்த்தி அப்டியினு சொல்லலாம்"
    print("Raw text      :", raw_text)

    # 1. Apply corrections
    corrected = corr_service.correct(raw_text, language="ta", is_tanglish=False)
    print("Corrected text:", corrected)

    # 2. Translate
    translated = trans_service.translate(corrected, src_lang="ta", tgt_lang="en")
    print("Translated    :", translated)

if __name__ == "__main__":
    test()
