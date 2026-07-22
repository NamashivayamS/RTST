import re

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
            num_map[f"{base_alt}{ones_alt[i]}"] = ten_mult + i
            num_map[f"{base_std}{ones_alt[i]}"] = ten_mult + i
            
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
                
    return num_map

TAMIL_NUM_MAP = generate_tamil_number_map()

def normalize_tamil_years(text: str) -> str:
    WORD_CHARS = r'[a-zA-Z0-9_\u0b80-\u0bff\u0900-\u097f]'
    
    # Define prefixes for 1900s
    prefix_1900_pattern = r'(?<!' + WORD_CHARS + r')(?:ஆயிரத்து\s+)?(?:தொள்ளாயிரத்து|தொள்ளாயிரம்|தொளாயிரத்து|தொளாயிரம்|தொளாயிறத்து|தொள்ளாயிறத்து)(?!' + WORD_CHARS + r')'
    # Define prefixes for 2000s
    prefix_2000_pattern = r'(?<!' + WORD_CHARS + r')(?:இரண்டாயிரத்து|இரண்டாயிரம்|ரெண்டாயிரத்து|ரெண்டாயிரம்|ரெண்டாயிரத்தி|இரண்டாயிரத்தி)(?!' + WORD_CHARS + r')'
    
    def parse_suffix_value(suffix_str: str) -> int:
        if not suffix_str:
            return 0
        suffix_str = suffix_str.strip()
        return TAMIL_NUM_MAP.get(suffix_str, 0)

    # Match 1900s
    pattern_1900 = re.compile(prefix_1900_pattern + r'(?:\s+([^\s]+))?', re.IGNORECASE)
    
    def replace_1900(match):
        suffix = match.group(1)
        if not suffix:
            return "1900"
        val = parse_suffix_value(suffix)
        if val > 0:
            return str(1900 + val)
        return "1900 " + suffix

    # Match 2000s
    pattern_2000 = re.compile(prefix_2000_pattern + r'(?:\s+([^\s]+))?', re.IGNORECASE)
    
    def replace_2000(match):
        suffix = match.group(1)
        if not suffix:
            return "2000"
        val = parse_suffix_value(suffix)
        if val > 0:
            return str(2000 + val)
        return "2000 " + suffix

    # Handle basic English year transliterations in Tamil script
    # e.g., "டூ தௌசண்ட் ஒன்"
    english_2000_pattern = r'(?<!' + WORD_CHARS + r')டூ\s+(?:தௌசண்ட்|தௌசன்|தௌசந்த்|தவுசண்ட்|தவுசன்|தவுசந்த்)(?!' + WORD_CHARS + r')'
    pattern_eng_2000 = re.compile(english_2000_pattern + r'(?:\s+([^\s]+))?', re.IGNORECASE)
    
    def replace_eng_2000(match):
        suffix = match.group(1)
        if not suffix:
            return "2000"
        val = parse_suffix_value(suffix)
        if val > 0:
            return str(2000 + val)
        return "2000 " + suffix

    text = pattern_1900.sub(replace_1900, text)
    text = pattern_2000.sub(replace_2000, text)
    text = pattern_eng_2000.sub(replace_eng_2000, text)
    
    # Normalize 2.0 / two point zero
    text = re.sub(r'(?<!' + WORD_CHARS + r')டூ\s+பாயிண்ட்\s+(?:ஒரு|ஜீரோ|ஓ|ஒன்பது)(?!' + WORD_CHARS + r')', '2.0', text)
    
    return text

# Test cases
test_cases = [
    ("ஆயிரத்து தொள்ளாயிரம் அறுபத்தெட்டு", "1968"),
    ("ஆயிரத்து தொளாயிறத்து அறுபத்தெட்டு", "1968"),
    ("இரண்டாயிரத்து பதினைந்து", "2015"),
    ("இரண்டாயிரத்து ஒன்று", "2001"),
    ("டூ தௌசண்ட் ஒன்", "2001"),
    ("டூ தௌசண்ட் பதினைந்து", "2015"),
    ("டூ பாயிண்ட் ஒரு", "2.0"),
    ("டூ பாயிண்ட் ஜீரோ", "2.0"),
    ("வந்தது ஆயிரத்து தொள்ளாயிரம் அறுபத்தெட்டு படம்", "வந்தது 1968 படம்"),
]

for src, expected in test_cases:
    res = normalize_tamil_years(src)
    assert res == expected, f"Failed: {repr(src)} -> {repr(res)}, expected {repr(expected)}"
    print(f"PASS: {repr(src)} -> {repr(res)}")
