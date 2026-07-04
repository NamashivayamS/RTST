import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
import numpy as np
from services.speaker_id_service import SpeakerIDService

# ── Fill these in — one enrollment clip + one test clip per person ─────
SPEAKERS = {
    "Namashivayam": {
        "enroll": "tests/audio_samples/me_clip1.wav",
        "test":   "tests/audio_samples/me_clip2.wav",
    },
    "Gobinath": {
        "enroll": "tests/audio_samples/Gobinath1.wav",
        "test":   "tests/audio_samples/Gobinath2.wav",
    },
    "Nagarajan": {
        "enroll": "tests/audio_samples/Nagarajan1.wav",
        "test":   "tests/audio_samples/Nagarajan2.wav",  # need a 2nd Nagarajan clip
    },
}
THRESHOLD = 0.45
# ─────────────────────────────────────────────────────────────────────

def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr

service = SpeakerIDService()

# ── Enroll everyone first ───────────────────────────────────────────
print("\n=== ENROLLMENT ===")
for name, clips in SPEAKERS.items():
    audio, sr = load_audio(clips["enroll"])
    print(f"Enrolling {name} ({len(audio)/sr:.1f}s)...")
    service.enroll_speaker(name, audio, sample_rate=sr)

# ── Cross-check every test clip against every enrolled profile ─────
print("\n=== CROSS-CHECK MATRIX ===")
print(f"{'Test clip':<15}", end="")
for name in SPEAKERS:
    print(f"{name:>15}", end="")
print()

results_summary = []

for test_name, clips in SPEAKERS.items():
    audio, sr = load_audio(clips["test"])
    result = service.identify_speaker(audio, sample_rate=sr, threshold=THRESHOLD)
    scores = result["scores"]

    print(f"{test_name:<15}", end="")
    for enrolled_name in SPEAKERS:
        score = scores.get(enrolled_name, 0.0)
        print(f"{score:>15.4f}", end="")
    print()

    correct = (result["speaker_id"] == test_name)
    results_summary.append((test_name, result["speaker_id"], correct))

# ── Verdict ──────────────────────────────────────────────────────────
print("\n=== VERDICT ===")
all_correct = True
for test_name, matched, correct in results_summary:
    status = "✅ CORRECT" if correct else "❌ WRONG"
    print(f"{test_name}'s clip → matched as '{matched}'  {status}")
    if not correct:
        all_correct = False

print(f"\n{'All matches correct — threshold holds.' if all_correct else 'MISMATCH FOUND — threshold or clips need review.'}")