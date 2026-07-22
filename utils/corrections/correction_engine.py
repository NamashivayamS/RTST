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


def generate_tamil_number_map():
    ones = ["", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
    ones_alt = ["", "ஒன்னு", "ரெண்டு", "மூணு", "நாலு", "அஞ்சு", "ஆறு", "ஏழு", "எட்டு", "ஒம்பது"]
    ones_eng = ["", "ஒன்", "டூ", "திரீ", "போர்", "பைவ்", "சிக்ஸ்", "செவன்", "எய்ட்", "நைன்"]
    
    teens = {
        "பத்து": 10, "பதினொன்று": 11, "பன்னிரண்டு": 12, "பதின்மூன்று": 13, "பதினான்கு": 14,
        "பதினைந்து": 15, "பதினாறு": 16, "பதினேழு": 17, "பதினெட்டு": 18, "பத்தொன்பது": 19
    }
    teens_alt = {
        "பதினொன்னு": 11, "பன்னெண்டு": 12, "பதின்மூணு": 13, "பதினாலு": 14,
        "பதினைஞ்சு": 15, "பத்தொம்பது": 19,
        # English transliterated
        "லெவன்": 11, "டுவெல்வ்": 12, "தர்ட்டீன்": 13, "போர்ட்டீன்": 14,
        "பிப்டீன்": 15, "சிக்ஸ்டீன்": 16, "செவண்டீன்": 17, "எய்ட்டீன்": 18, "நைன்டீன்": 19,
    }
    
    tens_names = ["", "பத்து", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"]
    tens_alt = ["", "பத்து", "இருபது", "முப்பது", "நாப்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"]
    
    num_map = {}
    
    # 1 to 9
    for i in range(1, 10):
        if ones[i]: num_map[ones[i]] = i
        if ones_alt[i]: num_map[ones_alt[i]] = i
        if ones_eng[i]: num_map[ones_eng[i]] = i
        
    # 10 to 19
    for k, v in teens.items(): num_map[k] = v
    for k, v in teens_alt.items(): num_map[k] = v
    
    # 20 to 99
    tens_bases = [
        (2, "இருபத்து", "இருபத்தி"),
        (3, "முப்பத்து", "முப்பத்தி"),
        (4, "நாற்பத்து", "நாப்பத்து"),
        (5, "ஐம்பத்து", "ஐம்பத்தி"),
        (6, "அறுபத்து", "அறுபத்தி"),
        (7, "எழுபத்து", "எழுபத்தி"),
        (8, "எண்பத்து", "எண்பத்தி"),
        (9, "தொண்ணூற்று", "தொண்ணூற்றி")
    ]
    
    # Also add standard tens:
    for i in range(2, 10):
        num_map[tens_names[i]] = i * 10
        num_map[tens_alt[i]] = i * 10
        
    # Add combinations:
    for t_val, base_std, base_alt in tens_bases:
        ten_mult = t_val * 10
        for i in range(1, 10):
            # Space or direct concat
            num_map[f"{base_std} {ones[i]}"] = ten_mult + i
            num_map[f"{base_std}{ones[i]}"] = ten_mult + i
            num_map[f"{base_alt} {ones_alt[i]}"] = ten_mult + i
            num_map[f"{base_alt}{ones_alt[i]}"] = ten_mult + i
            num_map[f"{base_std} {ones_alt[i]}"] = ten_mult + i
            num_map[f"{base_std}{ones_alt[i]}"] = ten_mult + i
            num_map[f"{base_alt} {ones[i]}"] = ten_mult + i
            num_map[f"{base_alt}{ones[i]}"] = ten_mult + i
            
            # Sandhi combinations (e.g. இருபத்தொன்று, இருபத்தெட்டு, அறுபத்தெட்டு)
            sandhi_map = {
                1: ["தொன்று", "தொன்னு", "தொன்"],
                2: ["திரண்டு", "ரெண்டு", "டூ"],
                3: ["மூன்று", "மூணு", "திரீ"],
                4: ["நான்கு", "நாலு", "போர்"],
                5: ["தைந்து", "தைஞ்சு", "பைவ்"],
                6: ["தாறு", "ஆறு", "சிக்ஸ்"],
                7: ["தேழு", "ஏழு", "செவன்"],
                8: ["தெட்டு", "எட்டு", "எய்ட்"],
                9: ["தொன்பது", "தொம்பது", "நைன்"]
            }
            base_t = base_std[:-2] # e.g. "இருபத்து" -> "இருபத்"
            if base_std == "தொண்ணூற்று":
                base_t = "தொண்ணூற்"
                
            for s_word in sandhi_map[i]:
                num_map[f"{base_t}{s_word}"] = ten_mult + i
                
    # Generate ordinal variants for all numbers ending with 'ு' (\u0bc1)
    # This transforms e.g. ரெண்டு (2) -> ரெண்டாம் (2nd), ரெண்டாவது (2nd)
    ordinals = {}
    for k, v in num_map.items():
        if k.endswith('\u0bc1'):
            ordinals[k[:-1] + '\u0bbe\u0bae\u0bcd'] = v      # 'ாம்' suffix
            ordinals[k[:-1] + '\u0bbe\u0bb5\u0ba4\u0bc1'] = v  # 'ாவது' suffix
            
    num_map.update(ordinals)
    return num_map

TAMIL_NUM_MAP = generate_tamil_number_map()

# Specific prefixes of number words (including scales) to prevent partial year corruption
NUMBER_PREFIXES = {
    "ஒன்று", "ஒன்னு", "ஒன்",
    "இரண்டு", "இரண்ட", "ரெண்டு", "ரெண்ட", "டூ",
    "மூன்று", "மூணு", "திரீ", "முப்ப",
    "நான்கு", "நாலு", "போர்", "நாற்ப", "நாப்ப",
    "ஐந்து", "அஞ்சு", "பைவ்", "ஐம்ப",
    "ஆறு", "சிக்ஸ்", "அறுப",
    "ஏழு", "செவன்", "எழுப",
    "எட்டு", "எய்ட்", "எண்ப",
    "ஒன்பது", "ஒம்பது", "நைன்", "தொண்ணூ",
    "பத்து", "பதின", "பன்னெ", "பன்னிர", "லெவன்", "டுவெல்", "தர்ட்டீ", "போர்்ட்டீ", "பிப்டீ", "சிக்ஸ்டீ", "செவண்டீ", "எய்ட்டீ", "நைண்டீ",
    "இருப", "இருபத்",
    "நூறு", "நூற்", "நூறி",
    "ஆயி", "தொள்ளாயி", "தொளாயி",
    "லட்ச", "இலட்ச", "கோடி"
}

BLOCKED_NUM_PATTERN = (
    r'(?:' + "|".join(re.escape(w) for w in NUMBER_PREFIXES) + r')'
)

# Sort suffixes by length descending to match longest-first
sorted_suffix_keys = sorted(TAMIL_NUM_MAP.keys(), key=len, reverse=True)
TAMIL_SUFFIX_PATTERN = "|".join(re.escape(k) for k in sorted_suffix_keys)

# Define prefixes for 1900s
PREFIX_1900_PATTERN = r'(?<!' + WORD_CHARS + r')(?:ஆயிரத்து\s+)?(?:தொள்ளாயிரத்து|தொள்ளாயிரம்|தொளாயிரத்து|தொளாயிரம்|தொளாயிறத்து|தொள்ளாயிறத்து)(?!' + WORD_CHARS + r')'
# Define prefixes for 2000s
PREFIX_2000_PATTERN = r'(?<!' + WORD_CHARS + r')(?:இரண்டாயிரத்து|இரண்டாயிரம்|ரெண்டாயிரத்து|ரெண்டாயிரம்|ரெண்டாயிரத்தி|இரண்டாயிரத்தி)(?!' + WORD_CHARS + r')'
# English transliterated 2000s
ENGLISH_2000_PATTERN = r'(?<!' + WORD_CHARS + r')டூ\s+(?:தௌசண்ட்|தௌசன்|தௌசந்த்|தவுசண்ட்|தவுசன்|தவுசந்த்)(?!' + WORD_CHARS + r')'

# Compile the patterns at module level using the all-or-nothing logic with lookahead
PATTERN_1900 = re.compile(
    r'(?:' + PREFIX_1900_PATTERN + r'\s+(' + TAMIL_SUFFIX_PATTERN + r')(?!' + WORD_CHARS + r')|' +
    PREFIX_1900_PATTERN + r'(?!\s+' + BLOCKED_NUM_PATTERN + r'))',
    re.IGNORECASE
)

PATTERN_2000 = re.compile(
    r'(?:' + PREFIX_2000_PATTERN + r'\s+(' + TAMIL_SUFFIX_PATTERN + r')(?!' + WORD_CHARS + r')|' +
    PREFIX_2000_PATTERN + r'(?!\s+' + BLOCKED_NUM_PATTERN + r'))',
    re.IGNORECASE
)

PATTERN_ENG_2000 = re.compile(
    r'(?:' + ENGLISH_2000_PATTERN + r'\s+(' + TAMIL_SUFFIX_PATTERN + r')(?!' + WORD_CHARS + r')|' +
    ENGLISH_2000_PATTERN + r'(?!\s+' + BLOCKED_NUM_PATTERN + r'))',
    re.IGNORECASE
)

def has_preceding_number(match):
    full_text = match.string
    preceding = full_text[:match.start()].strip()
    if preceding:
        last_word = preceding.split()[-1]
        # Scales used in numbers (independent U+0B86 and dependent U+0BBE variants)
        scales = {"நூறு", "நூற்று", "நூற்றி", "ஆயிரம்", "ஆயிரத்து", "லட்ச", "இலட்ச", "கோடி"}
        if any(last_word.endswith(w) for w in scales) or any(last_word.startswith(p) for p in NUMBER_PREFIXES):
            return True
    return False

def replace_1900(match):
    if has_preceding_number(match):
        return match.group(0)
    suffix_str = match.group(1)
    if not suffix_str:
        return "1900"
    val = TAMIL_NUM_MAP.get(suffix_str.strip(), 0)
    year = 1900 + val
    
    # Validation checks
    if year < 1900 or year > 2099:
        return match.group(0) # Out of target range, return unmodified
        
    if year > 2000 and (year % 100) == 0 and year != 2000:
        return match.group(0) # Rejects e.g. "2100", "2200", etc. which may just be "2100 rupees"
        
    # Preserve ordinal endings if present in the matched string
    ordinal_suffix = ""
    if suffix_str and suffix_str.endswith('\u0bbe\u0bae\u0bcd'):  # "ாம்"
        ordinal_suffix = "ஆம்"
    elif suffix_str and suffix_str.endswith('\u0bbe\u0bb5\u0ba4\u0bc1'):  # "ாவது"
        ordinal_suffix = "ஆவது"
        
    return f"{year}{ordinal_suffix}"

def replace_2000(match):
    if has_preceding_number(match):
        return match.group(0)
    suffix_str = match.group(1)
    if not suffix_str:
        return "2000"
    val = TAMIL_NUM_MAP.get(suffix_str.strip(), 0)
    year = 2000 + val
    
    # Validation checks
    if year < 1900 or year > 2099:
        return match.group(0) # Out of target range, return unmodified
        
    if year > 2000 and (year % 100) == 0 and year != 2000:
        return match.group(0) # Rejects e.g. "2100", "2200", etc. which may just be "2100 rupees"
        
    # Preserve ordinal endings if present in the matched string
    ordinal_suffix = ""
    if suffix_str and suffix_str.endswith('\u0bbe\u0bae\u0bcd'):  # "ாம்"
        ordinal_suffix = "ஆம்"
    elif suffix_str and suffix_str.endswith('\u0bbe\u0bb5\u0ba4\u0bc1'):  # "ாவது"
        ordinal_suffix = "ஆவது"
        
    return f"{year}{ordinal_suffix}"

def replace_eng_2000(match):
    if has_preceding_number(match):
        return match.group(0)
    suffix_str = match.group(1)
    if not suffix_str:
        return "2000"
    val = TAMIL_NUM_MAP.get(suffix_str.strip(), 0)
    year = 2000 + val
    
    # Validation checks
    if year < 1900 or year > 2099:
        return match.group(0) # Out of target range, return unmodified
        
    if year > 2000 and (year % 100) == 0 and year != 2000:
        return match.group(0) # Rejects e.g. "2100", "2200", etc. which may just be "2100 rupees"
        
    # Preserve ordinal endings if present in the matched string
    ordinal_suffix = ""
    if suffix_str and suffix_str.endswith('\u0bbe\u0bae\u0bcd'):  # "ாம்"
        ordinal_suffix = "ஆம்"
    elif suffix_str and suffix_str.endswith('\u0bbe\u0bb5\u0ba4\u0bc1'):  # "ாவது"
        ordinal_suffix = "ஆவது"
        
    return f"{year}{ordinal_suffix}"

def normalize_tamil_years(text: str) -> str:
    text = PATTERN_1900.sub(replace_1900, text)
    text = PATTERN_2000.sub(replace_2000, text)
    text = PATTERN_ENG_2000.sub(replace_eng_2000, text)
    
    # Normalize 2.0 / two point zero
    text = re.sub(r'(?<!' + WORD_CHARS + r')டூ\s+பாயிண்ட்\s+(?:ஒரு|ஜீரோ|ஓ|ஒன்பது)(?!' + WORD_CHARS + r')', '2.0', text)
    
    return text


def apply_tamil_corrections(text: str) -> str:
    text = text.strip()

    # Pre-normalize numbers and years in Tamil text to digit format
    text = normalize_tamil_years(text)

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