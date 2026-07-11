# scratch/test_translation_corrections.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.translation_service import TranslationService

def test():
    service = TranslationService()

    # Base colloquial text
    text1 = "இன்று நம்முடைய இன்டர்வியூ கெஸ்ட் அப்டியினா வேட்டி உலகின் சக்கரவர்த்தி அப்டியினு சொல்லலாம்"
    
    # 1. Option 1: Colloquial-to-formal Tamil
    text2 = "இன்று நம்முடைய இன்டர்வியூ கெஸ்ட் அப்படி என்றால் வேட்டி உலகின் சக்கரவர்த்தி அப்படி என்று சொல்லலாம்"
    
    # 2. Option 2: Direct English word injection
    text3 = "இன்று நம்முடைய இன்டர்வியூ கெஸ்ட் that is வேட்டி உலகின் சக்கரவர்த்தி that is சொல்லலாம்"

    print("--- TRANSLATING ---")
    
    res1 = service.translate(text1, src_lang="ta", tgt_lang="en")
    print(f"Original  : {text1}")
    print(f"Translated: {res1}\n")

    res2 = service.translate(text2, src_lang="ta", tgt_lang="en")
    print(f"Option 1 (Tamil Formal): {text2}")
    print(f"Translated             : {res2}\n")

    res3 = service.translate(text3, src_lang="ta", tgt_lang="en")
    print(f"Option 2 (English Word): {text3}")
    print(f"Translated             : {res3}\n")

if __name__ == "__main__":
    test()
