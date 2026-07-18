"""
runtime.py
==========

Starts and manages the RTST FastAPI backend.

Responsibilities
----------------
1. Start backend/main.py
2. Monitor startup
3. Stop backend cleanly
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests


class RuntimeManager:

    def __init__(self):

        self.process = None
        self.project_root = self.find_project_root()

    # ---------------------------------------------------------

    def find_project_root(self):

        current = Path.cwd()

        while current != current.parent:

            if (current / "backend").exists():
                return current

            current = current.parent

        raise FileNotFoundError("RTST project not found.")

    # ---------------------------------------------------------

    def start(self):

        if self.process is not None:
            print("Backend already running.")
            return

        backend_file = self.project_root / "backend" / "main.py"

        print("Starting RTST Backend...")

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=self.project_root,
        )

    # ---------------------------------------------------------

    def wait_until_ready(self, timeout=180):

        print("Waiting for FastAPI...")

        start = time.time()

        while time.time() - start < timeout:

            try:

                response = requests.get(
                    "http://127.0.0.1:8000/health",
                    timeout=2,
                )

                if response.status_code == 200:

                    print("RTST Backend Ready.")

                    return True

            except Exception:
                pass

            time.sleep(2)

        raise RuntimeError("Backend failed to start.")

    # ---------------------------------------------------------

    def stop(self):

        if self.process is None:
            return

        print("Stopping Backend...")

        self.process.terminate()

        self.process.wait()

        self.process = None

        print("Stopped.")

    # ---------------------------------------------------------

    def is_running(self):

        if self.process is None:
            return False

        return self.process.poll() is None