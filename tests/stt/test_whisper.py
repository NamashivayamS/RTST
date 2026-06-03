from faster_whisper import WhisperModel
import time

model_size = "medium"

model = WhisperModel(
    model_size,
    device="cuda",
    compute_type="float16"
)

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St2.wav"

start = time.time()

segments, info = model.transcribe(
    audio_path,
    beam_size=5
)

print("Detected language:", info.language)

print("\nTranscription:")

for segment in segments:
    print(segment.text)

end = time.time()

print(f"\nTime Taken: {end-start:.2f} seconds")