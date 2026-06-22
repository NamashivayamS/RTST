import re
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

def apply_tamil_corrections(text: str) -> str:
    text = text.strip()
    # 1. Regex Normalization Pass for Ramraj in Tamil (e.g. ராம்ராஜ், ராமராஜின், ராம்ராஜுக்கு -> ராமராஜ்)
    text = re.sub(r'(?<!\w)ராம்?ராஜ\w*(?!\w)', 'ராமராஜ்', text)
    # Standardize Ramraj Cotton in Tamil
    text = re.sub(r'(?<!\w)ராமராஜ்\s*காட்ட\w*(?!\w)', 'ராமராஜ் காட்டன்', text)
    
    # First apply proper noun corrections
    for wrong, correct in PROPER_NOUN_CORRECTIONS.items():
        pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
        text = re.sub(pattern, correct, text)

    # Then apply Tamil corrections
    for wrong, correct in TAMIL_CORRECTIONS.items():
        pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
        text = re.sub(pattern, correct, text)

    return text

def apply_tanglish_corrections(text: str) -> str:
    # Tanglish transliteration substitutions (Tamil -> English text)
    for wrong, correct in TANGLISH_CORRECTIONS.items():
        pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
        text = re.sub(pattern, correct, text)
        
    return text