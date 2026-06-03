from utils.translation_refinement.phrase_dictionary import (
    PHRASE_REPLACEMENTS
)

def refine_translation(text):

    refined_text = text

    for wrong, correct in PHRASE_REPLACEMENTS.items():

        refined_text = refined_text.replace(
            wrong,
            correct
        )

    return refined_text