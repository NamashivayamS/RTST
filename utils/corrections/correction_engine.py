from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS
from utils.corrections.proper_noun_corrections import PROPER_NOUN_CORRECTIONS

# Merge all correction dictionaries
ALL_CORRECTIONS = {
    **TAMIL_CORRECTIONS,
    **TANGLISH_CORRECTIONS,
    **PROPER_NOUN_CORRECTIONS   # ← applied to all languages
}


def apply_corrections(text):

    # Remove unwanted spaces/newlines
    text = text.strip()

    corrected_text = text

    for wrong, correct in ALL_CORRECTIONS.items():

        corrected_text = corrected_text.replace(
            wrong,
            correct
        )

    return corrected_text