import re
from utils.translation_refinement.phrase_dictionary import (
    PHRASE_REPLACEMENTS
)

def refine_translation(text):
    refined_text = text

    for wrong, correct in PHRASE_REPLACEMENTS.items():
        # Use regex with boundary checks to avoid corrupting substrings.
        # (?<![\u0B80-\u0BFFa-zA-Z0-9_]) ensures no preceding Tamil/Latin letter
        # (?![\u0B80-\u0BFFa-zA-Z0-9_]) ensures no succeeding Tamil/Latin letter
        pattern = r'(?<![\u0B80-\u0BFFa-zA-Z0-9_])' + re.escape(wrong) + r'(?![\u0B80-\u0BFFa-zA-Z0-9_])'
        refined_text = re.sub(pattern, correct, refined_text)

    return refined_text