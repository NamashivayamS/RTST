import time
import torch
import numpy as np
import os
import sys

# Add root folder to sys.path so we can import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_pool
from services.speaker_id_service import SpeakerIDService

def profile_device(device):
    print(f"\n==================================================")
    print(f"Profiling SpeakerIDService on device: {device.upper()}")
    print(f"==================================================")
    
    try:
        service = SpeakerIDService(device=device)
    except Exception as e:
        print(f"Failed to initialize service on {device}: {e}")
        return None

    # We will test durations: 1s, 3s, 5s, 10s, 15s, 20s
    durations = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    sr = 16000
    
    results = {}
    
    # Warmup
    print("Warming up model...")
    warmup_audio = np.random.randn(sr * 3).astype(np.float32)
    for _ in range(3):
        service.get_embedding(warmup_audio, sr)
        
    print(f"{'Duration (s)':<15}{'Run 1 (ms)':<12}{'Run 2 (ms)':<12}{'Run 3 (ms)':<12}{'Avg (ms)':<12}")
    print("-" * 65)
    
    for dur in durations:
        audio = np.random.randn(int(sr * dur)).astype(np.float32)
        runs = []
        for i in range(3):
            t0 = time.perf_counter()
            service.get_embedding(audio, sr)
            dt_ms = (time.perf_counter() - t0) * 1000
            runs.append(dt_ms)
            
        avg = sum(runs) / len(runs)
        results[dur] = {
            "runs": runs,
            "avg": avg
        }
        print(f"{dur:<15.1f}{runs[0]:<12.1f}{runs[1]:<12.1f}{runs[2]:<12.1f}{avg:<12.1f}")
        
    return results

if __name__ == "__main__":
    # Initialize DB pool
    print("Initializing DB Pool...")
    init_pool()
    
    cpu_results = profile_device("cpu")
    
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        gpu_results = profile_device("cuda")
        
        # Compare
        if cpu_results and gpu_results:
            print("\n==================================================")
            print("COMPARISON: CPU vs GPU Average Latencies")
            print("==================================================")
            print(f"{'Duration (s)':<15}{'CPU Avg (ms)':<15}{'GPU Avg (ms)':<15}{'Speedup':<10}")
            print("-" * 60)
            for dur in cpu_results.keys():
                cpu_avg = cpu_results[dur]["avg"]
                gpu_avg = gpu_results[dur]["avg"]
                speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
                print(f"{dur:<15.1f}{cpu_avg:<15.1f}{gpu_avg:<15.1f}{speedup:<10.1f}x")
    else:
        print("CUDA is not available on this system. GPU profiling skipped.")
