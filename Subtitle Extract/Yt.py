from youtube_transcript_api import YouTubeTranscriptApi

try:
    transcripts = YouTubeTranscriptApi.list_transcripts("xUbc_hYGSbg")

    print("Available transcripts:")
    for t in transcripts:
        print(
            f"Language: {t.language}, "
            f"Code: {t.language_code}, "
            f"Generated: {t.is_generated}"
        )

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR:", e)