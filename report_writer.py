# report_writer.py
"""
Thread-safe, non-blocking feedback report writer.

The problem with open(file, "a") under concurrency
────────────────────────────────────────────────────
`/api/report` is a FastAPI endpoint. FastAPI runs endpoints on a thread pool
by default (unless you use `async def`). Even with `async def`, the
`loop.run_in_executor()` calls in main.py mean multiple threads can call
`open(file, "a")` simultaneously.

Python's GIL does NOT protect file writes. Two threads writing at the same
time will interleave bytes mid-line, producing malformed JSON like:

  {"utterance_id": "abc", "src_la{"utterance_id": "def", ...}

This silently corrupts your feedback log and breaks any downstream parsing.

The fix: a single background writer thread
──────────────────────────────────────────
All callers enqueue a dict onto a thread-safe queue.Queue. A single
background daemon thread is the only entity that ever writes to the file,
so writes are always serialised. Callers return immediately — no blocking.

The SENTINEL object signals the writer to flush and exit cleanly on shutdown.

Usage
─────
  from report_writer import enqueue_report, start_report_writer, stop_report_writer

  # In startup_event:
  start_report_writer(path="feedback_reports.jsonl")

  # In /api/report endpoint:
  enqueue_report(report_data_dict)

  # In shutdown_event:
  stop_report_writer()
"""

import json
import logging
import queue
import threading
import os

logger = logging.getLogger("ispeak.report_writer")

_report_queue: queue.Queue = queue.Queue()
_writer_thread: threading.Thread | None = None
_SENTINEL = object()


def start_report_writer(path: str = "feedback_reports.jsonl") -> None:
    """
    Start the background writer thread.
    Call once in FastAPI startup_event.
    """
    global _writer_thread

    if _writer_thread is not None and _writer_thread.is_alive():
        logger.warning("[ReportWriter] Already running — ignoring duplicate start.")
        return

    # Ensure the directory exists
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)

    def _writer_loop():
        logger.info(f"[ReportWriter] Writer thread started → {path}")
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            # buffering=1 → line-buffered: each write() flushes immediately.
            # This means the file is always readable up to the last completed
            # line, even if the process crashes.
            while True:
                item = _report_queue.get()
                if item is _SENTINEL:
                    logger.info("[ReportWriter] Sentinel received — shutting down.")
                    _report_queue.task_done()
                    break
                try:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception:
                    logger.exception("[ReportWriter] Failed to write report: %s", item)
                finally:
                    _report_queue.task_done()

    _writer_thread = threading.Thread(
        target=_writer_loop,
        name="ReportWriter",
        daemon=True,   # Won't block process exit if stop_report_writer() is skipped
    )
    _writer_thread.start()


def stop_report_writer(timeout: float = 5.0) -> None:
    """
    Flush all pending reports and stop the writer thread.
    Call in FastAPI shutdown_event.
    """
    global _writer_thread
    if _writer_thread is None or not _writer_thread.is_alive():
        return

    _report_queue.put(_SENTINEL)
    _writer_thread.join(timeout=timeout)

    if _writer_thread.is_alive():
        logger.warning(
            "[ReportWriter] Writer thread did not stop within %.1fs — "
            "some reports may be lost.", timeout
        )
    else:
        logger.info("[ReportWriter] Writer thread stopped cleanly.")

    _writer_thread = None


def enqueue_report(report_data: dict) -> None:
    """
    Non-blocking: place a report dict on the queue.
    The background thread writes it to disk. Returns immediately.
    """
    _report_queue.put(report_data)
