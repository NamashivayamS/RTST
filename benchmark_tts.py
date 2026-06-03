import time
import soundfile as sf
import sys
import numpy as np

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading Real-Time Speech Translator Models...")

# Import the integrated TTS model
from models.indic_f5_model import generate_tamil_speech
from transcribe_tamil import transcribe_tamil

# ==========================================
# 1. GPU WARMUP (Simulating Production State)
# ==========================================
print("\nPerforming tiny GPU warmup (to avoid first-run CUDA allocation overhead)...")
_ = generate_tamil_speech("வணக்கம்.")
print("Warmup complete. System is in steady-state production mode.")

# ==========================================
# 2. THE REAL BENCHMARK
# ==========================================
unseen_target_text = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."
print("\n" + "="*50)
print(f"BENCHMARK TARGET: '{unseen_target_text}'")
print("="*50)

# Start High-Precision Timer (only wrapping the TTS generation)
start_time = time.perf_counter()

# Generate the speech
audio_arr, sr = generate_tamil_speech(unseen_target_text)

# End Timer
end_time = time.perf_counter()
latency = end_time - start_time
audio_duration = len(audio_arr) / sr
real_time_factor = latency / audio_duration

print(f"\n⏱️  GENERATION LATENCY: {latency:.2f} seconds")
print(f"🔊  AUDIO DURATION: {audio_duration:.2f} seconds")
print(f"⚡  REAL-TIME FACTOR (RTF): {real_time_factor:.2f}x (Lower is better)")

if real_time_factor < 1.0:
    print("✅ Performance is faster than real-time! Excellent for live translation.")
else:
    print("⚠️ Performance is slower than real-time.")

# ==========================================
# 3. SAVE AND VERIFY
# ==========================================
output_path = "output_realworld_benchmark.wav"
sf.write(output_path, audio_arr, sr)
print(f"\nSaved Audio to: {output_path}")

print("\nVerifying Unseen Pronunciation with Whisper...")
transcription = transcribe_tamil(
    audio_path=output_path,
    model_size="medium",
    device="cuda",
    compute_type="float16"
)

print(f"\n🎯 WHISPER TRANSCRIPTION: '{transcription}'")
print("="*50)
