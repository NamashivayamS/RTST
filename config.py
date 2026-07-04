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

# ── Environment mode ──────────────────────────────────────────────────────────
# Set ENVIRONMENT=production on the company server to activate safety checks.
# In development: defaults to "development" (lenient, works out-of-the-box).
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# ── Streaming / VAD thresholds ────────────────────────────────────────────────
SAMPLE_RATE          = 16000
VAD_SILENCE_SEC      = 0.7    # seconds of silence before utterance fires
VAD_MIN_SPEECH_SEC   = 0.5    # minimum utterance length to process
VAD_MAX_SPEECH_SEC   = 7.0    # force-fire after this many seconds of continuous speech
VAD_THRESHOLD        = 0.60   # Silero VAD sensitivity (0=sensitive, 1=strict)

# ── Turn-taking silence threshold ─────────────────────────────────────────────
# This is SEPARATE from VAD_SILENCE_SEC which controls when an utterance fires.
# TURN_TAKING_SILENCE_SEC controls when we assume the *speaker has changed*:
#   - language lock is cleared (next speaker's language re-detected cleanly)
#   - translation window is cleared (no cross-speaker context bleed)
# Must be longer than VAD_SILENCE_SEC so it only triggers between turns,
# not between normal sentence pauses within a single speaker's turn.
TURN_TAKING_SILENCE_SEC = 2.0

# ── STT thresholds ────────────────────────────────────────────────────────────
STT_NO_SPEECH_THRESHOLD    = 0.95
STT_LANG_CONFIDENCE_FLOOR  = 0.70
# NOTE: STT_BEAM_SIZE / STT_BEAM_SIZE_TAMIL were removed.
# Beam size is controlled inside STTService directly (the Tamil fine-tune
# uses greedy decoding internally). Exposing them here was misleading because
# main.py and RouterService never imported or passed them.

# ── Pipeline flags ────────────────────────────────────────────────────────────
ENABLE_TTS = False

# ── Sliding-window translation ────────────────────────────────────────────────
# ENABLE_SLIDING_WINDOW: set to False to disable the two-pass window entirely.
#   Useful for debugging when you want to confirm whether a translation issue
#   is caused by the window extraction logic or the model itself.
#   When False, every utterance uses a single-pass translation (draft only).
ENABLE_SLIDING_WINDOW = True

# TRANSLATION_WINDOW_SIZE: how many chunks to keep in context.
#   2 = keep 1 previous chunk + current chunk = 2-chunk window.
#   Increase to 3 for longer-context topics (+~100ms latency per extra chunk).
#   WARNING: do not exceed 3 — IndicTrans2's decoder saturates around 256 tokens
#   and produces repetition artifacts on very long combined inputs.
TRANSLATION_WINDOW_SIZE = 2

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

# ── CORS allowed origins ──────────────────────────────────────────────────────
# In production: set to a comma-separated list of allowed domains.
#   export CORS_ALLOWED_ORIGINS="https://translate.company.com"
# In development: leave empty → allows all origins (works out-of-the-box).
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "")

# ── Encryption key path (env-first, dev fallback) ─────────────────────────────
# Point this at wherever server_public.key lives on your deployment machine.
SERVER_PUBLIC_KEY_PATH = os.environ.get(
    "SERVER_PUBLIC_KEY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_public.key")
)

# ── Production safety check ───────────────────────────────────────────────────
# When ENVIRONMENT=production, crash on startup if critical secrets are missing.
# This ensures the company server never runs with dev fallback credentials.
if ENVIRONMENT == "production":
    _REQUIRED_ENV = [
        "POSTGRES_PASSWORD", "POSTGRES_USER",
        "DEFAULT_DEPARTMENT_ID", "SESSION_TOKEN",
    ]
    for _var in _REQUIRED_ENV:
        if not os.environ.get(_var):
            raise RuntimeError(
                f"[config] ENVIRONMENT=production but required variable "
                f"'{_var}' is not set. Set it before starting the server."
            )