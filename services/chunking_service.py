class ChunkingService:
    def __init__(self, min_words: int = 3, max_words: int = 5):
        """
        Initializes the chunking service with safe word limits for the TTS engine.
        IndicF5 performs best on 3-5 word chunks.

        min_words: minimum words before a punctuation boundary triggers a split.
                   Prevents single-word chunks like "வணக்கம்." which degrade TTS quality.
        max_words: hard upper limit — always split here regardless of punctuation.
        """
        self.min_words = min_words
        self.max_words = max_words

    def split_text_for_tts(self, text: str) -> list[str]:
        """
        Intelligently splits Tamil/English text into safe TTS chunks.

        Split priority:
          1. Punctuation boundary — only if current chunk has >= min_words.
             This prevents tiny 1-word chunks like "வணக்கம்." from being sent alone.
          2. max_words hard limit — always split here to prevent TTS degradation.
          3. Remaining words — flushed as the final chunk regardless of size.
        """
        if not text:
            return []

        words = text.split()
        chunks = []
        current_chunk = []

        punctuation_marks = {'.', ',', '!', '?', ';', ':', '।'}

        for word in words:
            current_chunk.append(word)

            has_punctuation = any(word.endswith(p) for p in punctuation_marks)

            # Condition 1: Natural pause AND chunk is large enough to stand alone
            if has_punctuation and len(current_chunk) >= self.min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                continue

            # Condition 2: Hard word limit reached — must split
            if len(current_chunk) >= self.max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Flush remaining words as the final chunk
        if current_chunk:
            # If remaining words are very short AND there are previous chunks,
            # merge into the last chunk rather than creating a tiny orphan chunk.
            if chunks and len(current_chunk) < self.min_words:
                chunks[-1] = chunks[-1] + " " + " ".join(current_chunk)
            else:
                chunks.append(" ".join(current_chunk))

        return chunks


# Quick test when run directly
if __name__ == "__main__":
    service = ChunkingService(min_words=3, max_words=5)

    tests = [
        # Standard multi-sentence Tamil
        "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்.",
        # Single short word with punctuation — should NOT become a 1-word chunk
        "வணக்கம். நான் நலமாக இருக்கிறேன்.",
        # English sentence
        "Hello friends, the meeting will start tomorrow morning at eight o'clock.",
        # No punctuation at all
        "இது ஒரு நீண்ட வாக்கியம் எந்த நிறுத்தற்குறியும் இல்லாமல் தொடர்கிறது",
    ]

    for test in tests:
        print(f"\nInput : {test}")
        chunks = service.split_text_for_tts(test)
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i} ({len(chunk.split())} words): '{chunk}'")
