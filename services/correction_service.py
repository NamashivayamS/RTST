import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.corrections.correction_engine import apply_tamil_corrections, apply_tanglish_corrections
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS


class CorrectionService:
    """
    Post-STT text correction service.

    Language-aware: Tamil/Tanglish corrections only applied when source
    language is Tamil ('ta'). English input gets pronoun injection + proper noun fixes.
    """

    def __init__(self):
        print("CorrectionService initialized and ready.")

    def correct(self, text: str, language: str = "ta", is_tanglish: bool = False) -> str:
        if not text.strip():
            return text
            
        if language == "ta":
            corrected = apply_tamil_corrections(text)
            if is_tanglish:
                corrected = apply_tanglish_corrections(corrected)
            return corrected
        if language == "en":
            from utils.corrections.correction_engine import apply_proper_nouns
            import re
            corrected = text.strip()
            # Regex Normalization for Ramraj variants in English (e.g. Ram raj, Raamraj -> Ramraj)
            corrected = re.sub(r'(?<!\w)[Rr]a+m\s*[Rr]a+j\w*(?!\w)', 'Ramraj', corrected)
            corrected = re.sub(r'(?<!\w)Ramraj\s*[Cc]ott?on(?!\w)', 'Ramraj Cotton', corrected)
            
            corrected = self._inject_missing_pronouns(corrected)
            corrected = apply_proper_nouns(corrected)
            return corrected
        return text.strip()

    def correct_tamil_only(self, text: str) -> str:
        import re
        text = text.strip()
        corrected = text
        for wrong, correct in TAMIL_CORRECTIONS.items():
            pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
            corrected = re.sub(pattern, correct, corrected)
        return corrected

    def correct_tanglish_only(self, text: str) -> str:
        import re
        text = text.strip()
        corrected = text
        for wrong, correct in TANGLISH_CORRECTIONS.items():
            pattern = r'(?<!\w)' + re.escape(wrong) + r'(?!\w)'
            corrected = re.sub(pattern, correct, corrected)
        return corrected

    # -------------------------------------------------------------------------
    # NLP Heuristic sets
    # -------------------------------------------------------------------------

    # Rule 1 — words that can front a first-person statement without a subject
    _CONJUNCTIONS_AND_ADVERBS = {
        # Original conjunctions
        "and", "so", "but", "then", "also", "because",
        # Fronted adverbs (new — Rule 1 extension)
        "already", "still", "just", "now", "finally",
        "currently", "always", "never", "usually", "sometimes",
        "today", "tomorrow", "yesterday",
    }

    # Words that may follow a conjunction but are NOT verbs — skip injection
    _EXCLUSIONS = {
        "i", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "our", "their",
        "this", "that", "these", "those", "there", "here",
        "the", "a", "an", "some", "any", "all", "most",
        "many", "someone", "everyone", "no", "every",
        "to", "for", "in", "on", "at", "with", "by", "from",
        "about", "as", "into", "like", "through", "after",
        "over", "between", "out", "against", "during",
        "without", "before", "under", "around", "among", "of",
        "what", "who", "where", "when", "why", "how", "which",
        "very", "really", "quite", "rather", "somewhat", "not",
        "is", "are", "was", "were", "will", "would",
        "can", "could", "should", "shall",
        "has", "have", "had", "am",
    }

    # Rule 3 — -ing words that are NOT progressive verbs
    _ING_EXCLUSIONS = {
        "morning", "evening", "something", "everything",
        "anything", "nothing", "including", "interesting",
        "following", "according", "during", "regarding",
    }

    # Rule 4 — irregular past-tense verbs
    _IRREGULAR_PAST = {
        "went", "ate", "had", "was", "were", "did",
        "came", "saw", "got", "made", "took", "said",
        "left", "felt", "met", "ran", "sat", "slept",
        "bought", "brought", "thought", "found", "gave",
        "kept", "knew", "heard", "read", "told", "spoke",
        "wrote", "drove", "flew", "swam", "won", "lost",
        "paid", "sent", "spent", "stood", "understood",
    }

    # Rule 2 — base-form action verbs (original list, unchanged)
    _ACTION_VERBS = {
        "go", "eat", "do", "make", "take", "get", "want",
        "need", "like", "think", "know", "see", "look",
        "use", "work", "try", "start", "stop", "tell",
        "ask", "come", "leave", "feel", "put", "bring",
        "buy", "play", "run", "walk", "speak", "talk",
        "say", "call", "find", "give", "keep", "let",
        "begin", "show", "hear", "write", "learn",
        "study", "change", "help", "watch", "understand",
    }

    def _inject_missing_pronouns(self, text: str) -> str:
        """
        Four-rule heuristic that injects 'I' or 'I am' into subject-less
        clauses before they reach the translation engine.

        Rule 1 — Conjunction / fronted adverb + verb:
            "And go home to eat"      →  "And I go home to eat"
            "Already finished report" →  "Already I finished report"
            "Still working on it"     →  "Still I am working on it"
            "Today going to office"   →  "Today I am going to office"

        Rule 2 — Bare base-form action verb:
            "Eat chicken every day"   →  "I eat chicken every day"
            "Go to office tomorrow"   →  "I go to office tomorrow"

        Rule 3 — Bare -ing verb (continuous tense):
            "Working from home today" →  "I am working from home today"
            "Traveling to Chennai"    →  "I am traveling to Chennai"

        Rule 4 — Irregular OR regular past-tense verb:
            "Went to the market"      →  "I went to the market"
            "Finished the meeting"    →  "I finished the meeting"

        Global guard — utterances under 4 words are likely genuine commands
        ("Try this", "Start now") — no injection performed.
        """
        import re

        words = text.split()

        # ── Global guard ────────────────────────────────────────────────────
        if len(words) < 4:
            return text

        # ── Question guard — never inject pronouns into questions ─────────
        if text.rstrip().endswith('?'):
            return text

        fw_raw        = words[0]
        fw            = re.sub(r"[^a-z]", "", fw_raw.lower())

        # ── Rule 1: conjunction / fronted adverb ─────────────────────────────
        if fw in self._CONJUNCTIONS_AND_ADVERBS:
            if len(words) < 2:
                return text

            sw_raw = words[1]
            sw     = re.sub(r"[^a-z]", "", sw_raw.lower())

            # Whisper capitalises proper nouns — skip injection for those
            if sw_raw[0].isupper() and sw not in ("i",):
                return text

            if sw in self._EXCLUSIONS or not sw:
                return text

            prefix    = fw_raw + " "
            remainder = text[len(prefix):]

            if sw.endswith("ing") and sw not in self._ING_EXCLUSIONS:
                return prefix + "I am " + remainder
            return prefix + "I " + remainder

        # ── Rule 3: bare -ing verb (checked BEFORE Rule 2) ───────────────────
        if (fw.endswith("ing")
                and fw not in self._ING_EXCLUSIONS
                and fw not in self._EXCLUSIONS):
            return "I am " + text[0].lower() + text[1:]

        # ── Rule 4: irregular past-tense verb ────────────────────────────────
        if fw in self._IRREGULAR_PAST:
            return "I " + text[0].lower() + text[1:]

        # ── Rule 4b: regular -ed past tense ──────────────────────────────────
        if fw.endswith("ed") and fw not in self._EXCLUSIONS and len(fw) > 4:
            return "I " + text[0].lower() + text[1:]

        # ── Rule 2: bare base-form action verb ───────────────────────────────
        if fw in self._ACTION_VERBS:
            return "I " + text[0].lower() + text[1:]

        return text


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  →  python correction_service.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    tests = [
        # Rule 1 — original conjunctions
        ("And go to my home and eat chicken.",    "And I go to my home"),
        ("But eating lunch right now.",           "But I am eating lunch"),
        # Rule 1 — fronted adverbs (extension)
        ("Already finished the report today.",    "Already I finished"),
        ("Still working on the backend now.",     "Still I am working"),
        ("Just completed the task today.",        "Just I completed"),
        ("Today going to the office here.",       "Today I am going"),
        ("Tomorrow leaving for the conference.",  "Tomorrow I am leaving"),
        # Rule 2 — bare action verb
        ("Eat chicken every single day.",         "I eat chicken"),
        ("Go to office every morning now.",       "I go to office"),
        # Rule 3 — bare -ing verb (continuous)
        ("Working from home today morning.",      "I am working from home"),
        ("Traveling to Chennai next week.",       "I am traveling to Chennai"),
        # Rule 4 — irregular past
        ("Went to the market yesterday.",         "I went to the market"),
        ("Ate outside with my friends.",          "I ate outside"),
        # Rule 4b — regular -ed past
        ("Finished the meeting just now.",        "I finished the meeting"),
        ("Completed all the tasks today.",        "I completed all"),
        # Global guard — short sentences untouched
        ("Try this.",                             "Try this."),
        ("Start now.",                            "Start now."),
        # Must NOT be touched
        ("I went to the market.",                 "I went to the market."),
        ("The food is ready now.",                "The food is ready"),
        ("He is coming to office.",               "He is coming to office"),
    ]

    class _MockService(CorrectionService):
        def correct(self, text, language="en"):
            return self._inject_missing_pronouns(text.strip())

    svc = _MockService()
    print(f"\n{'INPUT':<46} {'OUTPUT':<46} PASS")
    print("─" * 106)
    all_pass = True
    for inp, frag in tests:
        result = svc.correct(inp)
        ok     = frag in result
        all_pass = all_pass and ok
        mark   = "✓" if ok else "✗  FAIL"
        print(f"{inp:<46} {result:<46} {mark}")
    print("─" * 106)
    print(f"\n{'All ' + str(len(tests)) + ' tests passed ✓' if all_pass else 'Some tests FAILED ✗'}\n")
