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

try:
    import torch
except ImportError:
    torch = None


# ==========================================================
# Data Classes
# ==========================================================

@dataclass
class PythonInfo:
    version: str
    executable: str
    implementation: str


@dataclass
class OSInfo:
    system: str
    release: str
    version: str
    machine: str
    processor: str


@dataclass
class TorchInfo:
    installed: bool
    version: str
    cuda_available: bool
    cuda_version: str
    cudnn_enabled: bool


# ==========================================================
# Environment Inspector
# ==========================================================

class EnvironmentInspector:

    def get_python_info(self) -> PythonInfo:

        return PythonInfo(
            version=sys.version.split()[0],
            executable=sys.executable,
            implementation=platform.python_implementation(),
        )

    def get_os_info(self) -> OSInfo:

        return OSInfo(
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
        )

    def get_torch_info(self) -> TorchInfo:

        if torch is None:
            return TorchInfo(
                installed=False,
                version="Not Installed",
                cuda_available=False,
                cuda_version="N/A",
                cudnn_enabled=False,
            )

        return TorchInfo(
            installed=True,
            version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda or "CPU Only",
            cudnn_enabled=torch.backends.cudnn.is_available(),
        )


# ==========================================================
# Main (Testing)
# ==========================================================

def main():

    inspector = EnvironmentInspector()

    python_info = inspector.get_python_info()
    os_info = inspector.get_os_info()
    torch_info = inspector.get_torch_info()

    print("\n================ RTST Environment Report ================\n")

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

    print("\nPyTorch")
    print("--------------------------------------------")
    print(f"Installed       : {torch_info.installed}")
    print(f"Version         : {torch_info.version}")
    print(f"CUDA Available  : {torch_info.cuda_available}")
    print(f"CUDA Version    : {torch_info.cuda_version}")
    print(f"cuDNN Enabled   : {torch_info.cudnn_enabled}")

    print("\n=========================================================")


if __name__ == "__main__":
    main()