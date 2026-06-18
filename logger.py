# logger.py
"""
Centralized logging configuration for RealTimeSpeechTranslator.

Why replace print() with logging?
───────────────────────────────────
• print() output vanishes when the process is managed by systemd/supervisord
  (stdout is not captured by default). logging writes to a file that persists
  across restarts.
• logging has levels (DEBUG / INFO / WARNING / ERROR / CRITICAL) so you can
  suppress debug noise in production with a single config line.
• logging is thread-safe out of the box. print() under concurrent threads
  can interleave mid-line in the terminal.
• Structured log lines (timestamp + level + logger name + message) make
  log files grep-able and ingest-able by tools like Loki / Datadog.

Structure
─────────
  ispeak                 root logger  (controls overall level)
  ispeak.db              database queries
  ispeak.pipeline        RouterService pipeline steps
  ispeak.ws              WebSocket connection events
  ispeak.vad             VAD decisions
  ispeak.stt             STT results
  ispeak.translation     Translation results
  ispeak.tts             TTS generation
  ispeak.gc              Garbage collector

Usage (in any module)
─────────────────────
  import logging
  logger = logging.getLogger("ispeak.pipeline")
  logger.info("STT result: %s", text)          # lazy formatting — fast
  logger.debug("VAD segments: %s", segments)   # only logged when DEBUG enabled
  logger.error("Translation failed", exc_info=True)  # includes traceback

Call setup_logging() ONCE, at the very top of main.py before any imports
that themselves call logging.getLogger().
"""

import logging
import logging.handlers
import os
import sys


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = "logs/ispeak.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,               # keep 5 rotated files → 50 MB total
) -> None:
    """
    Configure the 'ispeak' logger hierarchy.

    Parameters
    ──────────
    log_level:    "DEBUG" | "INFO" | "WARNING" | "ERROR"
                  Override with LOG_LEVEL env var in production.
    log_file:     Path to the rotating log file.
                  None → log to stdout only (useful for Docker).
    max_bytes:    Rotate the file when it exceeds this size.
    backup_count: Number of rotated files to keep before deleting oldest.

    Log format
    ──────────
    2025-01-15 14:32:01,234 | INFO     | ispeak.pipeline | STT result: Hello
    """

    # Allow env var override so you can set LOG_LEVEL=DEBUG without
    # touching source code.
    level_str = os.environ.get("LOG_LEVEL", log_level).upper()
    level = getattr(logging, level_str, logging.INFO)

    # ── Format ────────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Root 'ispeak' logger ──────────────────────────────────────────────────
    root = logging.getLogger("ispeak")
    root.setLevel(level)

    # Prevent duplicate log lines if setup_logging() is called more than once
    if root.handlers:
        return

    # ── Console handler (always on) ───────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── Rotating file handler (optional) ─────────────────────────────────────
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.info(
        "[Logger] Logging initialised — level=%s, file=%s",
        level_str,
        log_file or "stdout only",
    )

    # ── Silence noisy third-party loggers ─────────────────────────────────────
    # These libraries log at DEBUG/INFO constantly and pollute your output.
    for noisy in ("uvicorn.access", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
