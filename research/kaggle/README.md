# Kaggle GPU Validation Guide

This directory contains the configurations and launcher script to run and validate the Real-Time Speech Translator (RTST) backend on a Kaggle Linux GPU instance.

## Workflow

To test and validate the application on Kaggle:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NamashivayamS/RTST.git
   cd RTST
   ```

2. **Install Dependencies with Constraints**:
   Run the package installation using the Kaggle requirements manifest restricted by the preinstalled environment constraints:
   ```bash
   pip install -r research/kaggle/requirements-kaggle.txt -c research/kaggle/constraints-kaggle.txt
   ```

3. **Start the Launcher**:
   Start the automated runner which downloads `cloudflared` (if not present), launches the Uvicorn ASGI application, opens a Cloudflare Quick Tunnel, and displays the public testing URL:
   ```bash
   python research/kaggle/launch.py
   ```

4. **Verify and Terminate**:
   - Open the printed public HTTPS URL in your browser to interact with the backend service.
   - Once validation is complete, press `Ctrl+C` in the terminal to gracefully stop both Uvicorn and Cloudflared.
