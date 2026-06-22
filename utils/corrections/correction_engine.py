import re
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

_sorted_proper_nouns = sorted(PROPER_NOUN_CORRECTIONS.keys(), key=len, reverse=True)
_proper_noun_pattern = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, _sorted_proper_nouns)) + r')(?!\w)') if _sorted_proper_nouns else None

_sorted_tamil = sorted(TAMIL_CORRECTIONS.keys(), key=len, reverse=True)
_tamil_pattern = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, _sorted_tamil)) + r')(?!\w)') if _sorted_tamil else None

_sorted_tanglish = sorted(TANGLISH_CORRECTIONS.keys(), key=len, reverse=True)
_tanglish_pattern = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, _sorted_tanglish)) + r')(?!\w)') if _sorted_tanglish else None

def apply_proper_nouns(text: str) -> str:
    if _proper_noun_pattern:
        text = _proper_noun_pattern.sub(lambda m: PROPER_NOUN_CORRECTIONS[m.group(1)], text)
    return text

def apply_tamil_corrections(text: str) -> str:
    text = text.strip()
    # 1. Regex Normalization Pass for Ramraj in Tamil (e.g. ராம்ராஜ், ராமராஜின், ராம்ராஜுக்கு -> ராமராஜ்)
    text = re.sub(r'(?<!\w)ராம்?ராஜ\w*(?!\w)', 'ராமராஜ்', text)
    # Standardize Ramraj Cotton in Tamil
    text = re.sub(r'(?<!\w)ராமராஜ்\s*காட்ட\w*(?!\w)', 'ராமராஜ் காட்டன்', text)
    
    # First apply proper noun corrections
    text = apply_proper_nouns(text)

    # Then apply Tamil corrections
    if _tamil_pattern:
        text = _tamil_pattern.sub(lambda m: TAMIL_CORRECTIONS[m.group(1)], text)

    return text

def apply_tanglish_corrections(text: str) -> str:
    # Tanglish transliteration substitutions (Tamil -> English text)
    if _tanglish_pattern:
        text = _tanglish_pattern.sub(lambda m: TANGLISH_CORRECTIONS[m.group(1)], text)
        
    return text