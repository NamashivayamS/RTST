# utils/corrections/proper_noun_corrections.py
"""
Proper noun corrections for Indian names, places, and organizations
that Whisper medium consistently mistranscribes.

Add entries as you discover new mistranscriptions from the terminal logs.
Format: "Whisper's wrong output": "Correct spelling"
"""

PROPER_NOUN_CORRECTIONS = {
    # ── People names ──────────────────────────────────────────────────────
    "Namah Shivayam":  "Namashivayam",  
    "Namah Namashivayam": "Namashivayam", 
    "Namaskaram":      "Namashivayam",
    "Namashrayam":     "Namashivayam",
    "Namashwayam":     "Namashivayam",
    "Shivayam":        "Namashivayam",

    # ── Places ────────────────────────────────────────────────────────────
    "Lamal":          "Lamel",
    "Lamal Vidyasram":"Chelammal Vidyashram",
    "Coimbatore Institute of Technology": "Coimbatore Institute of Technology",

    # ── Organizations ─────────────────────────────────────────────────────
    "Ramraj":         "Ramraj",   # already correct, keep as anchor

    # ── Add more as you see them in logs ──────────────────────────────────
}
