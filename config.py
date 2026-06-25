# config.py
"""
Central configuration for all tunable thresholds and logic gates.

─── Secret Loading Order ────────────────────────────────────────────────────
Production: set environment variables before starting uvicorn.
  export POSTGRES_PASSWORD=your_real_password
  export POSTGRES_USER=ispeak_prod
  export POSTGRES_DB=ispeak_global
  export POSTGRES_HOST=your-db-host
  export DEFAULT_DEPARTMENT_ID=your-uuid

Development fallback: values below are used only when the env var is absent.
  This means the dev machine works out-of-the-box with no extra setup,
  while production NEVER uses hardcoded credentials.

  .env file support: if you install python-dotenv, add this to the very
  top of main.py BEFORE any other import:
      from dotenv import load_dotenv; load_dotenv()
  Then put your secrets in a .env file (git-ignored).
─────────────────────────────────────────────────────────────────────────────
"""

import os

# ── Streaming / VAD thresholds ────────────────────────────────────────────────
SAMPLE_RATE          = 16000
VAD_SILENCE_SEC      = 0.9    # seconds of silence before utterance fires (increased to prevent chopping Tamil words)
VAD_MIN_SPEECH_SEC   = 0.5    # minimum utterance length to process
VAD_MAX_SPEECH_SEC   = 7.0   # force-fire after this many seconds of continuous speech
VAD_THRESHOLD        = 0.60   # Silero VAD sensitivity (0=sensitive, 1=strict)

# ── STT thresholds ────────────────────────────────────────────────────────────
STT_NO_SPEECH_THRESHOLD    = 0.95
STT_LANG_CONFIDENCE_FLOOR  = 0.70
STT_BEAM_SIZE              = 2   # primary model (English detection)
STT_BEAM_SIZE_TAMIL        = 1   # Tamil fine-tune — greedy is sufficient

# ── Pipeline flags ────────────────────────────────────────────────────────────
ENABLE_TTS = False

# ── Concurrency ───────────────────────────────────────────────────────────────
# Maximum pipeline tasks that can queue per WebSocket connection.
# Tune this down (1) to reduce GPU pressure, up (3) to improve throughput
# when sentences arrive faster than they can be processed.
MAX_PIPELINE_QUEUE = 2

# ── Environment Presets ───────────────────────────────────────────────────────
ENVIRONMENT_PRESETS = {
    "quiet": {
        "vad_threshold":       0.60,
        "silence_sec":         0.7,
        "min_speech_sec":      0.50,
        "no_speech_threshold": 0.95,
        "rms_gate":            0.005,
    },
    "conference": {
        "vad_threshold":       0.70,
        "silence_sec":         0.7,
        "min_speech_sec":      0.50,
        "no_speech_threshold": 0.85,
        "rms_gate":            0.010,
    },
    "noisy": {
        "vad_threshold":       0.80,
        "silence_sec":         0.8,
        "min_speech_sec":      0.80,
        "no_speech_threshold": 0.75,
        "rms_gate":            0.015,
    },
}

# ── Database credentials (env-first, dev fallback) ────────────────────────────
# In production: set these as real environment variables.
# In development: the fallback values below are used automatically.
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST",     "localhost")
POSTGRES_DB       = os.environ.get("POSTGRES_DB",       "ispeak_global")
POSTGRES_USER     = os.environ.get("POSTGRES_USER",     "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "1234")      # dev only

DEFAULT_DEPARTMENT_ID = os.environ.get(
    "DEFAULT_DEPARTMENT_ID",
    "b6f8468a-477c-4045-a696-c402afae99a5"   # dev only
)

# ── Encryption key path (env-first, dev fallback) ─────────────────────────────
# Point this at wherever server_public.key lives on your deployment machine.
SERVER_PUBLIC_KEY_PATH = os.environ.get(
    "SERVER_PUBLIC_KEY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_public.key")
)

# ── Production safety check ───────────────────────────────────────────────────
# Uncomment these lines when deploying to production to make misconfiguration
# an instant hard crash rather than a silent security hole.
#
# _REQUIRED_ENV = ["POSTGRES_PASSWORD", "POSTGRES_USER", "DEFAULT_DEPARTMENT_ID"]
# for _var in _REQUIRED_ENV:
#     if not os.environ.get(_var):
#         raise RuntimeError(
#             f"[config] Required environment variable '{_var}' is not set. "
#             f"Set it before starting the server."
#         )