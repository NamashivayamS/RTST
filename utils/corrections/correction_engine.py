import re
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

# Unicode-aware word characters: includes ASCII alphanumeric/underscore,
# Devanagari block (U+0900-U+097F), and Tamil block (U+0B80-U+0BFF) which contains
# both letters and combining vowel/consonant marks.
WORD_CHARS = r'[a-zA-Z0-9_\u0b80-\u0bff\u0900-\u097f]'


def _safe_sub(wrong: str, correct: str, text: str) -> str:
    """
    Substitutes 'wrong' with 'correct' in 'text' using lookarounds
    to respect word boundaries for both English and Indic scripts.
    """
    pattern = r'(?<!' + WORD_CHARS + r')' + re.escape(wrong) + r'(?!' + WORD_CHARS + r')'
    try:
        return re.sub(pattern, correct, text)
    except re.error:
        # Fallback to literal replace if regex compilation fails
        return text.replace(wrong, correct)


def apply_proper_nouns(text: str) -> str:
    """
    Applies proper noun corrections to any input regardless of script.
    Called from both apply_tamil_corrections AND apply_tanglish_corrections
    so that brand names (Ramraj Cotton, Tirupur, etc.) are normalised whether
    the speaker used Tamil script, transliterated Latin, or mixed both.
    """
    text = text.strip()
    for wrong, correct in PROPER_NOUN_CORRECTIONS.items():
        text = _safe_sub(wrong, correct, text)
    return text


def apply_tamil_corrections(text: str) -> str:
    text = text.strip()

    # Normalize Ramraj / Ramraj Cotton variations using lookarounds
    text = re.sub(
        r'(?<!' + WORD_CHARS + r')ராம்?ராஜ' + WORD_CHARS + r'*(?!' + WORD_CHARS + r')',
        'ராமராஜ்',
        text
    )
    text = re.sub(
        r'(?<!' + WORD_CHARS + r')ராமராஜ்\s*காட்ட' + WORD_CHARS + r'*(?!' + WORD_CHARS + r')',
        'ராமராஜ் காட்டன்',
        text
    )

    # Proper noun corrections apply to Tamil text (brand names in Tamil script)
    text = apply_proper_nouns(text)

    for wrong, correct in TAMIL_CORRECTIONS.items():
        text = _safe_sub(wrong, correct, text)
    return text


def apply_tanglish_corrections(text: str) -> str:
    """
    Applies Tanglish (code-switched Tamil+English) corrections.
    Also runs proper noun corrections because Tanglish speech frequently
    contains brand names and place names in Latin script (e.g. 'Ramraj Cotton',
    'Tirupur') that need the same normalisation as pure Tamil input.
    """
    text = text.strip()

    # Proper noun corrections apply to Tanglish text too (Latin-script brand names)
    text = apply_proper_nouns(text)

    for wrong, correct in TANGLISH_CORRECTIONS.items():
        text = _safe_sub(wrong, correct, text)
    return text