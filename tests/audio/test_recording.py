import sounddevice as sd
from scipy.io.wavfile import write
sd.default.latency = 'low'

fs = 48000
seconds = 5

print("Recording started... Speak now.")

audio = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1,
    dtype='float32',
    device=15
)

sd.wait()

write("test_recording.wav", fs, audio)

print("Recording completed!")
print("Saved as test_recording.wav")