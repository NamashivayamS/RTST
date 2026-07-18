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


# ==========================================================
# Data Classes
# ==========================================================

@dataclass
class PythonInfo:
    """Information about the current Python runtime."""

    version: str
    executable: str
    implementation: str


@dataclass
class OSInfo:
    """Information about the operating system."""

    system: str
    release: str
    version: str
    machine: str
    processor: str


# ==========================================================
# Environment Inspector
# ==========================================================

class EnvironmentInspector:
    """
    Collects information about the execution environment.

    This class ONLY inspects the environment.
    It never changes anything on the system.
    """

    def get_python_info(self) -> PythonInfo:
        """Return Python runtime information."""

        return PythonInfo(
            version=sys.version.split()[0],
            executable=sys.executable,
            implementation=platform.python_implementation(),
        )

    def get_os_info(self) -> OSInfo:
        """Return operating system information."""

        return OSInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
        )


# ==========================================================
# Main (Testing)
# ==========================================================

def main() -> None:
    inspector = EnvironmentInspector()

    python_info = inspector.get_python_info()
    os_info = inspector.get_os_info()

    print("\n================ Environment Report ================\n")

    print("Python")
    print("--------------------------------------------")
    print(f"Version         : {python_info.version}")
    print(f"Executable      : {python_info.executable}")
    print(f"Implementation  : {python_info.implementation}")

    print("\nOperating System")
    print("--------------------------------------------")
    print(f"System          : {os_info.system}")
    print(f"Release         : {os_info.release}")
    print(f"Version         : {os_info.version}")
    print(f"Machine         : {os_info.machine}")
    print(f"Processor       : {os_info.processor}")

    print("\n====================================================")


if __name__ == "__main__":
    main()