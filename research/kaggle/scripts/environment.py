"""
environment.py

Collects information about the current execution environment.

This module ONLY inspects the system.
It never installs packages, downloads models, or modifies anything.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass
class PythonInfo:
    """
    Information about the current Python runtime.
    """

    version: str
    executable: str
    implementation: str


class EnvironmentInspector:
    """
    Collects information about the execution environment.

    This class is responsible ONLY for inspection.
    It does not modify the system.
    """

    def get_python_info(self) -> PythonInfo:
        """
        Return information about the current Python runtime.
        """

        return PythonInfo(
            version=sys.version.split()[0],
            executable=sys.executable,
            implementation=platform.python_implementation(),
        )


def main() -> None:
    """
    Simple test entry point.
    """

    inspector = EnvironmentInspector()

    python_info = inspector.get_python_info()

    print("\n========== Python Information ==========")
    print(f"Version        : {python_info.version}")
    print(f"Executable     : {python_info.executable}")
    print(f"Implementation : {python_info.implementation}")
    print("========================================\n")


if __name__ == "__main__":
    main()