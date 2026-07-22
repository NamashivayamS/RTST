import sys
import os

# Set project root to sys.path so we can import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.corrections.correction_engine import normalize_tamil_years

# Test cases
test_cases = [
    # 1. Base test cases
    ("ஆயிரத்து தொள்ளாயிரம் அறுபத்தெட்டு", "1968"),
    ("ஆயிரத்து தொளாயிறத்து அறுபத்தெட்டு", "1968"),
    ("இரண்டாயிரத்து பதினைந்து", "2015"),
    ("இரண்டாயிரத்து ஒன்று", "2001"),
    ("டூ தௌசண்ட் ஒன்", "2001"),
    ("டூ தௌசண்ட் பதினைந்து", "2015"),
    ("டூ பாயிண்ட் ஒரு", "2.0"),
    ("டூ பாயிண்ட் ஜீரோ", "2.0"),
    ("வந்தது ஆயிரத்து தொள்ளாயிரம் அறுபத்தெட்டு படம்", "வந்தது 1968 படம்"),

    # 2. Spaced suffixes
    ("ஆயிரத்து தொள்ளாயிரத்து இருபத்து ஐந்து", "1925"),
    ("இரண்டாயிரத்து இருபத்து ஐந்து", "2025"),
    ("ஆயிரத்து தொள்ளாயிரத்து அறுபத்தெட்டு", "1968"),

    # 3. Preservation of particle "ஆம்" (no greedy consumption)
    ("ஆயிரத்து தொள்ளாயிரத்து ஒன்று ஆம் ஆண்டு", "1901 ஆம் ஆண்டு"),

    # 4. Out-of-range rejection (all-or-nothing policy)
    ("இரண்டாயிரத்து நூறு", "இரண்டாயிரத்து நூறு"),
    ("இரண்டாயிரத்து தொள்ளாயிரத்து தொண்ணூற்று ஒன்பது", "இரண்டாயிரத்து தொள்ளாயிரத்து தொண்ணூற்று ஒன்பது"),

    # 5. Non-number word target lookahead (lookahead should not be too strict)
    ("இரண்டாயிரத்து படம்", "2000 படம்"),
    ("இரண்டாயிரத்து", "2000"),

    # 6. Suffix word boundary checks (preventing partial matches on ordinal numbers)
    ("ஆயிரத்து தொள்ளாயிரத்து இருபத்து ஐந்தாவது", "ஆயிரத்து தொள்ளாயிரத்து இருபத்து ஐந்தாவது"),
]

print("=== Running Year Normalizer Tests ===")
failed = 0
for i, (src, expected) in enumerate(test_cases, 1):
    res = normalize_tamil_years(src)
    if res == expected:
        print(f"[{i:02d}] PASS: {repr(src)} -> {repr(res)}")
    else:
        print(f"[{i:02d}] FAIL: {repr(src)} -> Got {repr(res)}, Expected {repr(expected)}")
        failed += 1

print("=====================================")
if failed == 0:
    print(f"All {len(test_cases)} tests passed successfully!")
    sys.exit(0)
else:
    print(f"{failed} test(s) failed.")
    sys.exit(1)
