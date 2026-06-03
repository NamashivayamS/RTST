from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
import torch

model = load_silero_vad()

audio_path = r"D:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator\audio\Std Tamil\St1.wav"

wav, sr = sf.read(audio_path)

# Convert to float32 tensor
wav = torch.tensor(wav, dtype=torch.float32)

speech_timestamps = get_speech_timestamps(
    wav,
    model,
    sampling_rate=sr,
    return_seconds=True
)

print("\nSpeech Segments:\n")

for segment in speech_timestamps:
    print(segment)