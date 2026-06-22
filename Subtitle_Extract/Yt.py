from youtube_transcript_api import YouTubeTranscriptApi

video_id = "xUbc_hYGSbg"

transcript = YouTubeTranscriptApi.get_transcript(
    video_id,
    languages=['ta']
)

for item in transcript:
    print(item['text'])