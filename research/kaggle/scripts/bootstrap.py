"""
bootstrap.py
============

Entry point for the RTST Kaggle Development Platform.

Responsibilities
----------------
1. Verify execution environment.
2. Locate the RTST project.
3. Display environment information.
4. Prepare for runtime startup.

This file intentionally does NOT start FastAPI yet.
That responsibility belongs to runtime.py.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


class Bootstrap:

    def __init__(self):

        self.workspace = Path.cwd()
        self.project_root = None

    # -------------------------------------------------------------

    def print_banner(self):

        print("=" * 60)
        print(" Real-Time Speech Translator")
        print(" Kaggle Development Platform")
        print("=" * 60)

    # -------------------------------------------------------------

    def check_python(self):

        print(f"Python Version : {platform.python_version()}")

    # -------------------------------------------------------------

    def check_os(self):

        print(f"Operating System : {platform.system()} {platform.release()}")

    # -------------------------------------------------------------

    def detect_kaggle(self):

        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            print("Environment : Kaggle")
            return True

        print("Environment : Local Machine")
        return False

    # -------------------------------------------------------------

    def locate_project(self):

        current = Path.cwd()

        while current != current.parent:

            if (current / "backend").exists():

                self.project_root = current
                break

            current = current.parent

        if self.project_root is None:

            raise FileNotFoundError(
                "Unable to locate RTST project root."
            )

        print(f"Project Root : {self.project_root}")

    # -------------------------------------------------------------

    def run(self):

        self.print_banner()

        self.check_python()

        self.check_os()

        self.detect_kaggle()

        self.locate_project()

        print("\nBootstrap completed successfully.\n")


def bootstrap():

    Bootstrap().run()


if __name__ == "__main__":

    bootstrap()