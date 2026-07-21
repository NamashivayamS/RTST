#!/usr/bin/env python3
# research/kaggle/launch.py
"""
Kaggle Production-Quality Launcher for RTST Backend
===================================================
Starts the FastAPI/Uvicorn backend, downloads/launches Cloudflared Quick Tunnel,
and streams logs from both processes. Displays the public URL for runtime testing.
"""

import os
import sys
import time
import socket
import shutil
import signal
import re
import urllib.request
import platform
import subprocess
import threading

# Global flag to track shutdown status across threads
shutdown_initiated = False
uvicorn_proc = None
cloudflared_proc = None

def wait_for_port(host="127.0.0.1", port=8000, timeout=300):
    """
    Attempts to connect to the uvicorn port to ensure the backend is fully
    loaded and ready to receive traffic before starting the tunnel.
    """
    start_time = time.time()
    print(f"[*] Waiting for Uvicorn to start on {host}:{port}...", flush=True)
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print("[*] Backend is open and accepting connections.", flush=True)
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            if time.time() - start_time > timeout:
                print("[!] Timed out waiting for backend to start.", flush=True)
                return False
            time.sleep(1)

def setup_cloudflared():
    """
    Checks if cloudflared is installed globally. If not, detects the operating
    system and downloads the appropriate executable to the script's directory.
    """
    system = platform.system().lower()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Check if cloudflared is already in the system PATH
    bin_path = shutil.which("cloudflared")
    if bin_path:
        print(f"[*] Found system cloudflared at: {bin_path}", flush=True)
        return bin_path

    # 2. Determine local filename and download URL based on OS
    if system == "linux":
        download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        local_bin = os.path.join(script_dir, "cloudflared")
    elif system == "windows":
        download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        local_bin = os.path.join(script_dir, "cloudflared.exe")
    elif system == "darwin":
        download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
        local_bin = os.path.join(script_dir, "cloudflared")
    else:
        print(f"[!] Unsupported operating system: {system}", flush=True)
        sys.exit(1)

    # 3. Download if not present locally
    if not os.path.exists(local_bin):
        print(f"[*] cloudflared not found. Downloading for {system}...", flush=True)
        print(f"[*] URL: {download_url}", flush=True)
        try:
            # Set a modern user-agent to bypass any robot blocking
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req) as response, open(local_bin, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            
            # Make binary executable on Unix environments
            if system in ("linux", "darwin"):
                os.chmod(local_bin, 0o755)
            
            print(f"[*] Download completed successfully: {local_bin}", flush=True)
        except Exception as e:
            print(f"[!] Failed to download cloudflared: {e}", flush=True)
            sys.exit(1)
    else:
        print(f"[*] Using local cloudflared binary at: {local_bin}", flush=True)

    return local_bin

def print_banner(url):
    """Prints a prominent visual banner showing the public tunnel address."""
    banner = f"""
================================================================================
  CLOUDFLARE QUICK TUNNEL CREATED SUCCESSFULLY!
  
  Public HTTPS URL: \033[1;32;40m{url}\033[0m
  
  Open this link in your browser to validate the active workspace backend.
================================================================================
"""
    print(banner, flush=True)

def stream_output(pipe, prefix, is_cloudflared=False):
    """
    Helper thread target that reads lines from a subprocess's combined
    stdout/stderr pipe, prefixes them, and prints them to console.
    Also watches the cloudflared output stream to extract the quick tunnel URL.
    """
    url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    url_found = False

    for line in iter(pipe.readline, b""):
        decoded = line.decode("utf-8", errors="replace").rstrip()
        
        # Parse and display the Cloudflared Quick Tunnel URL if found
        if is_cloudflared and not url_found:
            match = url_regex.search(decoded)
            if match:
                url_found = True
                print_banner(match.group(0))
                
        print(f"[{prefix}] {decoded}", flush=True)

def signal_handler(signum, frame):
    """Handles SIGINT/SIGTERM to shut down child subprocesses cleanly."""
    global shutdown_initiated, uvicorn_proc, cloudflared_proc
    if shutdown_initiated:
        return
    shutdown_initiated = True
    
    print("\n[*] Ctrl+C / Termination signal detected. Cleaning up...", flush=True)
    
    # Gracefully terminate Cloudflared
    if cloudflared_proc and cloudflared_proc.poll() is None:
        print("[*] Stopping cloudflared...", flush=True)
        cloudflared_proc.terminate()
        try:
            cloudflared_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cloudflared_proc.kill()
            
    # Gracefully terminate Uvicorn
    if uvicorn_proc and uvicorn_proc.poll() is None:
        print("[*] Stopping uvicorn...", flush=True)
        uvicorn_proc.terminate()
        try:
            uvicorn_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            uvicorn_proc.kill()

    print("[*] Cleanup finished. Exiting.", flush=True)
    sys.exit(0)

def main():
    global uvicorn_proc, cloudflared_proc
    
    # Register termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Calculate workspaces paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    print("[*] Starting RTST Kaggle Launcher...", flush=True)

    # ── Automatically symlink the mounted Kaggle dataset if present ──
    # Note: We support linking to BOTH root/whisper-tamil-medium-ct2 and models/whisper-tamil-medium-ct2 
    # to ensure compatibility whichever path is active in whisper_model.py.
    kaggle_input_dir = "/kaggle/input/whisper-tamil-medium-ct2"
    if os.path.exists(kaggle_input_dir):
        # We will symlink to both potential target paths so that whichever configuration is used, it works.
        target_paths = [
            os.path.join(root_dir, "whisper-tamil-medium-ct2"),
            os.path.join(root_dir, "models", "whisper-tamil-medium-ct2")
        ]
        for target_link_dir in target_paths:
            if not os.path.exists(target_link_dir) and not os.path.islink(target_link_dir):
                print(f"[*] Kaggle dataset detected. Creating symbolic link to {target_link_dir}...", flush=True)
                try:
                    os.makedirs(os.path.dirname(target_link_dir), exist_ok=True)
                    os.symlink(kaggle_input_dir, target_link_dir)
                    print(f"[*] Symbolic link created successfully at {target_link_dir}.", flush=True)
                except Exception as sym_err:
                    print(f"[!] Failed to create symbolic link to {target_link_dir}: {sym_err}", flush=True)

    # 1. Set up cloudflared binary
    cloudflared_bin = setup_cloudflared()

    # 2. Launch Uvicorn backend process
    # Run uvicorn as a module using the active python interpreter to preserve env
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    
    try:
        uvicorn_proc = subprocess.Popen(
            uvicorn_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=root_dir
        )
    except Exception as e:
        print(f"[!] Failed to launch Uvicorn subprocess: {e}", flush=True)
        sys.exit(1)

    # Start Uvicorn log streaming thread
    t_uvicorn = threading.Thread(
        target=stream_output, 
        args=(uvicorn_proc.stdout, "uvicorn"), 
        daemon=True
    )
    t_uvicorn.start()

    # 3. Wait for backend port to open
    if not wait_for_port(host="127.0.0.1", port=8000, timeout=300):
        print("[!] Backend failed to start. Terminating.", flush=True)
        if uvicorn_proc.poll() is None:
            uvicorn_proc.kill()
        sys.exit(1)

    # 4. Launch Cloudflare Tunnel
    cloudflared_cmd = [cloudflared_bin, "tunnel", "--url", "http://127.0.0.1:8000"]
    try:
        cloudflared_proc = subprocess.Popen(
            cloudflared_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=root_dir
        )
    except Exception as e:
        print(f"[!] Failed to launch Cloudflared subprocess: {e}", flush=True)
        if uvicorn_proc.poll() is None:
            uvicorn_proc.kill()
        sys.exit(1)

    # Start Cloudflared log streaming thread
    t_cloudflared = threading.Thread(
        target=stream_output, 
        args=(cloudflared_proc.stdout, "cloudflared", True), 
        daemon=True
    )
    t_cloudflared.start()

    # 5. Monitor subprocesses in a polling loop
    while True:
        uv_status = uvicorn_proc.poll()
        cf_status = cloudflared_proc.poll()
        
        # If either child process terminates unexpectedly, fail fast and clean up
        if uv_status is not None and not shutdown_initiated:
            print(f"\n[!] ERROR: Uvicorn process died unexpectedly (Exit Code: {uv_status}).", flush=True)
            signal_handler(None, None)
            break
            
        if cf_status is not None and not shutdown_initiated:
            print(f"\n[!] ERROR: Cloudflared process died unexpectedly (Exit Code: {cf_status}).", flush=True)
            signal_handler(None, None)
            break

        time.sleep(1)

if __name__ == "__main__":
    main()
