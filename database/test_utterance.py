from database.queries import save_utterance

utterance_id = save_utterance(
    meeting_id="ac172601-5486-44ea-9e9c-2c0030502164",
    source_text="Hello everyone",
    translated_text="அனைவருக்கும் வணக்கம்",
    source_language="en",
    target_language="ta",
    total_latency_ms=850
)

print(utterance_id)