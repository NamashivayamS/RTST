import re
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

def apply_proper_nouns(text: str) -> str:
    text = text.strip()
    for wrong, correct in PROPER_NOUN_CORRECTIONS.items():
        try:
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text)
        except re.error:
            text = text.replace(wrong, correct)
    return text

def apply_tamil_corrections(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\bராம்?ராஜ\w*\b', 'ராமராஜ்', text)
    text = re.sub(r'\bராமராஜ்\s*காட்ட\w*\b', 'ராமராஜ் காட்டன்', text)
    
    text = apply_proper_nouns(text)

    for wrong, correct in TAMIL_CORRECTIONS.items():
        try:
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text)
        except re.error:
            text = text.replace(wrong, correct)
    return text

def apply_tanglish_corrections(text: str) -> str:
    text = text.strip()
    for wrong, correct in TANGLISH_CORRECTIONS.items():
        try:
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text)
        except re.error:
            text = text.replace(wrong, correct)
    return text