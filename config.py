# config.py

#Central configuration for all tunable thresholds and logic gates.

import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# ── Primary Whisper Model (Single Source of Truth) ────────────────────────────
# Both download_models.py (build) and whisper_model.py (runtime) import this.
PRIMARY_WHISPER_MODEL = "large-v3-turbo"

# ── Streaming / VAD thresholds ────────────────────────────────────────────────
SAMPLE_RATE          = 16000
VAD_SILENCE_SEC      = 0.7    # seconds of silence before utterance fires
VAD_MIN_SPEECH_SEC   = 0.5    # minimum utterance length to process
VAD_MAX_SPEECH_SEC   = 7.0    # force-fire after this many seconds of continuous speech
VAD_THRESHOLD        = 0.60   # Silero VAD sensitivity (0=sensitive, 1=strict)

# TURN_TAKING_SILENCE_SEC controls when we assume the *speaker has changed*:
TURN_TAKING_SILENCE_SEC = 2.0

# ── Speaker Identification ────────────────────────────────────────────────────
SPEAKER_ID_LOCAL_THRESHOLD     = 0.40  # Match threshold in current meeting session (lowered from 0.50 — logs showed valid matches at 0.43-0.48 being missed)
SPEAKER_ID_GLOBAL_THRESHOLD    = 0.70  # Strict threshold for global profile database
SPEAKER_ID_INTRO_THRESHOLD     = 0.45  # Lenient threshold used only during self-introduction name matching
SPEAKER_ID_MIN_ENROLL_DURATION = 3.0   # Min audio duration (sec) to auto-enroll a new speaker
SPEAKER_ID_DEVICE              = os.environ.get("SPEAKER_ID_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
SPEAKER_ID_MAX_TEMPLATES       = 5     # primary + up to 4 secondary voice templates per speaker
SPEAKER_ID_TEMPLATE_ACCEPT_SIM = 0.75  # only add a new template on a very confident passive match


# ── STT thresholds ────────────────────────────────────────────────────────────
STT_NO_SPEECH_THRESHOLD    = 0.95
STT_LANG_CONFIDENCE_FLOOR  = 0.70

ENABLE_TTS = False

ENABLE_SLIDING_WINDOW = True

# TRANSLATION_WINDOW_SIZE: how many chunks to keep in context.
TRANSLATION_WINDOW_SIZE = 2

# ── Concurrency ───────────────────────────────────────────────────────────────
# Maximum pipeline tasks that can queue per WebSocket connection.
MAX_PIPELINE_QUEUE = 2

# ── Environment Presets ───────────────────────────────────────────────────────
ENVIRONMENT_PRESETS = {
    "quiet": {
        "vad_threshold":       0.60,
        "silence_sec":         0.7,
        "min_speech_sec":      0.50,
        "no_speech_threshold": 0.95,
        "rms_gate":            0.002,
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
MSSQL_HOST     = os.environ.get("MSSQL_HOST",     "localhost")
MSSQL_PORT     = os.environ.get("MSSQL_PORT",     "1433")
MSSQL_DB       = os.environ.get("MSSQL_DB",       "ispeak_global")
MSSQL_USER     = os.environ.get("MSSQL_USER",     "sa")
MSSQL_PASSWORD = os.environ.get("MSSQL_PASSWORD", "1234")      # dev only
MSSQL_DRIVER   = os.environ.get("MSSQL_DRIVER",   "{ODBC Driver 18 for SQL Server}")
MSSQL_ENCRYPT  = os.environ.get("MSSQL_ENCRYPT",  "yes")        # "yes"/"no"

DEFAULT_DEPARTMENT_ID = os.environ.get(
    "DEFAULT_DEPARTMENT_ID",
    "b6f8468a-477c-4045-a696-c402afae99a5"   # dev only
)

# ── CORS allowed origins ──────────────────────────────────────────────────────
# In production: set to a comma-separated list of allowed domains.
# In development: leave empty → allows all origins (works out-of-the-box).
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "")

# Set LLM_API_KEY to enable meeting summarization. Leave empty to disable.
LLM_API_KEY    = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama-3.1-8b-instant")

# Point this at wherever server_public.key lives on your deployment machine.
SERVER_PUBLIC_KEY_PATH = os.environ.get(
    "SERVER_PUBLIC_KEY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_public.key")
)

# ── Language Registry ─────────────────────────────────────────────────────────
# Single source of truth for all supported languages.
# To add a new language, add ONE entry here. Zero code changes elsewhere.
LANGUAGE_REGISTRY = {
    "ta": {
        "name": "Tamil",
        "script_range": (0x0B80, 0x0BFF),
        "indictrans": "tam_Taml",
        "domain_seed": (
            "வணக்கம், இது தமிழில் பேச்சு. "
            "வருடங்கள், திரைப்படம், இயக்குநர், நிகழ்ச்சி, "
            "தொழில்நுட்பம், பொருளாதாரம், கல்வி, விளையாட்டு, "
            "விண்வெளி, அறிவியல் புனைகதை, ஸ்டான்லி குப்ரிக், "
            "ஆர்தர் சி கிளார்க், சென்டினல், இரண்டாயிரத்து ஒன்று."
        ),
    },
    "hi": {
        "name": "Hindi",
        "script_range": (0x0900, 0x097F),
        "indictrans": "hin_Deva",
        "domain_seed": "नमस्ते, यह हिंदी में बातचीत है। बैठक, कंपनी, उत्पादन, विपणन, बिक्री, प्रबंधन।",
    },
    "te": {
        "name": "Telugu",
        "script_range": (0x0C00, 0x0C7F),
        "indictrans": "tel_Telu",
        "domain_seed": "నమస్తే, ఇది తెలుగులో సంభాషణ. సమావేశం, ఉత్పత్తి, మార్కెటింగ్, నిర్వహణ.",
    },
    "ml": {
        "name": "Malayalam",
        "script_range": (0x0D00, 0x0D7F),
        "indictrans": "mal_Mlym",
        "domain_seed": "നമസ്കാരം, ഇത് മലയാളത്തിലുള്ള സംഭാഷണമാണ്. മീറ്റിംഗ്, ഉൽപ്പാദനം, വിപണനം, മാനേജ്മെന്റ്.",
    },
    "kn": {
        "name": "Kannada",
        "script_range": (0x0C80, 0x0CFF),
        "indictrans": "kan_Knda",
        "domain_seed": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡದಲ್ಲಿ ಮಾತುಕತೆ. ಸಭೆ, ಉತ್ಪಾದನೆ, ಮಾರುಕಟ್ಟೆ, ನಿರ್ವಹಣೆ.",
    },
    "en": {
        "name": "English",
        "script_range": None,   # No validation needed for English
        "indictrans": "eng_Latn",
        "domain_seed": "",
    },
}

# ── Production safety check ───────────────────────────────────────────────────
# When ENVIRONMENT=production, crash on startup if critical secrets are missing.
# This ensures the company server never runs with dev fallback credentials.
if ENVIRONMENT == "production":
    _REQUIRED_ENV = [
        "MSSQL_PASSWORD", "MSSQL_USER",
        "DEFAULT_DEPARTMENT_ID", "SESSION_TOKEN",
    ]
    for _var in _REQUIRED_ENV:
        if not os.environ.get(_var):
            raise RuntimeError(
                f"[config] ENVIRONMENT=production but required variable "
                f"'{_var}' is not set. Set it before starting the server."
            )
