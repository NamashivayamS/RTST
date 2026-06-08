import re
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

# Merge all correction dictionaries
ALL_CORRECTIONS = {
    **PROPER_NOUN_CORRECTIONS,  # ← applied to all languages first
    **TAMIL_CORRECTIONS,
    **TANGLISH_CORRECTIONS,
}

def apply_corrections(text: str) -> str:
    # Remove unwanted spaces/newlines
    text = text.strip()

    for wrong, correct in ALL_CORRECTIONS.items():
        # Word boundary match — won't corrupt substrings
        # We check both English word boundaries (\b) and general word boundaries for Tamil
        # Actually \w covers alphanumeric. For Unicode Tamil, \w often matches correctly in Python 3.
        pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
        text = re.sub(pattern, correct, text)

    return text