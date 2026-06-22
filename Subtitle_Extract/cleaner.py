import re

vtt_file = r"Angikaaram – The Journey of Ramraj and the Man Behind It ｜ Ramraj Brand Documentary Film [xUbc_hYGSbg].ta-orig.vtt"

with open(vtt_file, "r", encoding="utf-8") as f:
    text = f.read()

# Remove WEBVTT header
text = re.sub(r"WEBVTT.*?\n", "", text)

# Remove timestamp lines
text = re.sub(
    r"\d\d:\d\d:\d\d\.\d+\s+-->\s+\d\d:\d\d:\d\d\.\d+.*",
    "",
    text
)

# Remove inline timestamps
text = re.sub(r"<\d\d:\d\d:\d\d\.\d+>", "", text)

# Remove caption tags
text = re.sub(r"</?c>", "", text)

# Remove alignment metadata
text = re.sub(r"align:start.*", "", text)

# Remove music markers
text = re.sub(r"\[.*?\]", "", text)

lines = []

for line in text.splitlines():
    line = line.strip()

    if not line:
        continue

    if line not in lines:  # remove consecutive duplicates
        lines.append(line)

with open("reference_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Saved reference_clean.txt")