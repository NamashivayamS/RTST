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


@dataclass
class GPUInfo:
    available: bool
    gpu_count: int
    gpu_name: str
    total_memory_gb: float
    compute_capability: str
    allocated_memory_gb: float
    reserved_memory_gb: float


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

    def get_gpu_info(self) -> GPUInfo:

        if (
            torch is None
            or not torch.cuda.is_available()
            or torch.cuda.device_count() == 0
        ):
            return GPUInfo(
                available=False,
                gpu_count=0,
                gpu_name="N/A",
                total_memory_gb=0.0,
                compute_capability="N/A",
                allocated_memory_gb=0.0,
                reserved_memory_gb=0.0,
            )

        device = 0

        properties = torch.cuda.get_device_properties(device)

        total_memory = properties.total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)

        capability = f"{properties.major}.{properties.minor}"

        return GPUInfo(
            available=True,
            gpu_count=torch.cuda.device_count(),
            gpu_name=properties.name,
            total_memory_gb=round(total_memory, 2),
            compute_capability=capability,
            allocated_memory_gb=round(allocated_memory, 2),
            reserved_memory_gb=round(reserved_memory, 2),
        )


# ==========================================================
# Main
# ==========================================================

def main():

    inspector = EnvironmentInspector()

    python_info = inspector.get_python_info()
    os_info = inspector.get_os_info()
    torch_info = inspector.get_torch_info()
    gpu_info = inspector.get_gpu_info()

    print("\n================ RTST Environment Report ================\n")

    print("Python")
    print("-" * 55)
    print(f"Version              : {python_info.version}")
    print(f"Executable           : {python_info.executable}")
    print(f"Implementation       : {python_info.implementation}")

    print("\nOperating System")
    print("-" * 55)
    print(f"System               : {os_info.system}")
    print(f"Release              : {os_info.release}")
    print(f"Version              : {os_info.version}")
    print(f"Machine              : {os_info.machine}")
    print(f"Processor            : {os_info.processor}")

    print("\nPyTorch")
    print("-" * 55)
    print(f"Installed            : {torch_info.installed}")
    print(f"Version              : {torch_info.version}")
    print(f"CUDA Available       : {torch_info.cuda_available}")
    print(f"CUDA Version         : {torch_info.cuda_version}")
    print(f"cuDNN Enabled        : {torch_info.cudnn_enabled}")

    print("\nGPU")
    print("-" * 55)
    print(f"GPU Available        : {gpu_info.available}")
    print(f"GPU Count            : {gpu_info.gpu_count}")
    print(f"GPU Name             : {gpu_info.gpu_name}")
    print(f"Total Memory (GB)    : {gpu_info.total_memory_gb}")
    print(f"Compute Capability   : {gpu_info.compute_capability}")
    print(f"Allocated Memory     : {gpu_info.allocated_memory_gb} GB")
    print(f"Reserved Memory      : {gpu_info.reserved_memory_gb} GB")

    print("\n=========================================================\n")


if __name__ == "__main__":
    main()