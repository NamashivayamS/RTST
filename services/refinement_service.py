import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.translation_refinement.translation_refiner import refine_translation


class RefinementService:
    """
    Post-translation refinement.
    Only refines Tamil output (en→ta). English output from IndicTrans2
    is clean enough to not need rule-based refinement.
    """

    def __init__(self):
        print("RefinementService initialized and ready.")

    def refine(self, text: str, tgt_lang: str = "tam_Taml") -> str:
        if not text.strip():
            return text
            
        # Strip leading punctuation/artifacts often left by translation models
        # e.g., when "So, I think" becomes ", அது..."
        text = text.lstrip(" ,.-")
        
        if tgt_lang == "tam_Taml":
            return refine_translation(text)
        return text

    def refine_auto(self, translation_result: dict) -> str:
        return self.refine(
            translation_result["translated_text"],
            tgt_lang=translation_result["tgt_lang"]
        )


if __name__ == "__main__":
    service = RefinementService()
    raw_tamil = "குட் மார்னிங் ஹலோ எல்லாருக்கும் தாங்க் யூ"
    refined = service.refine(raw_tamil, tgt_lang="tam_Taml")
    print(f"IN : {raw_tamil}")
    print(f"OUT: {refined}")
