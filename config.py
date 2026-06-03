"""
config.py — Project-wide path constants.

Import from here instead of hardcoding absolute paths in test or service files.
All paths are resolved relative to this file's location (the project root),
so they work on any machine regardless of where the project is cloned.

Usage:
    from config import PROJECT_ROOT, REF_AUDIO_PATH, AUDIO_SAMPLES_DIR
"""

import os

# Absolute path to the project root directory (where this file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# tests/audio_samples/ — reference audio and test clips
AUDIO_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "tests", "audio_samples")

# Reference audio used by IndicF5 TTS for voice cloning
REF_AUDIO_PATH = os.path.join(AUDIO_SAMPLES_DIR, "ref_cropped.wav")

# Reference transcript matching the reference audio above
REF_TEXT = "தாமிரபரணி ஆற்றின் கரையுரங்களில் வசிக்கும்."

# Standard test audio clips (Tamil)
SAMPLE_TAMIL_1 = os.path.join(AUDIO_SAMPLES_DIR, "St1.wav")
SAMPLE_TAMIL_2 = os.path.join(AUDIO_SAMPLES_DIR, "St2.wav")
