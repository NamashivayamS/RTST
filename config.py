import os
from dotenv import load_dotenv

load_dotenv()

# config.py
"""
Central configuration for all tunable thresholds and logic gates.
"""

# ── Streaming / VAD thresholds ────────────────────────────────────────────────
SAMPLE_RATE          = 16000
VAD_SILENCE_SEC      = 0.3    # seconds of silence before utterance fires
VAD_MIN_SPEECH_SEC   = 0.35   # minimum utterance length to process
VAD_MAX_SPEECH_SEC   = 3.5    # force-fire after this many seconds of continuous speech
VAD_THRESHOLD        = 0.60   # Silero VAD sensitivity (0=sensitive, 1=strict)

# ── STT thresholds ────────────────────────────────────────────────────────────
STT_NO_SPEECH_THRESHOLD    = 0.95   # drop audio if Whisper this uncertain
STT_LANG_CONFIDENCE_FLOOR  = 0.70   # retry as English below this confidence
STT_BEAM_SIZE              = 2      # Whisper beam size (1=fast, 5=accurate)

# ── Pipeline flags ────────────────────────────────────────────────────────────
ENABLE_TTS = False   # set True to enable IndicF5 audio generation

# ── Environment Presets ───────────────────────────────────────────────────────
ENVIRONMENT_PRESETS = {
    "quiet": {
        "vad_threshold":       0.60,
        "silence_sec":         0.4,
        "min_speech_sec":      0.35,
        "no_speech_threshold": 0.95,
        "rms_gate":            0.005,
    },
    "conference": {
        "vad_threshold":       0.70,
        "silence_sec":         0.6,
        "min_speech_sec":      0.50,
        "no_speech_threshold": 0.85,
        "rms_gate":            0.010,
    },
    "noisy": {
        "vad_threshold":       0.80,
        "silence_sec":         0.8,
        "min_speech_sec":      0.60,
        "no_speech_threshold": 0.75,
        "rms_gate":            0.015,
    },
}

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ispeak_global")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "1234")

DEFAULT_DEPARTMENT_ID = "b6f8468a-477c-4045-a696-c402afae99a5"