# Real-Time Speech Translator (Indic Languages ↔ English)

This project provides a bidirectional streaming pipeline that translates spoken Indic languages (Tamil, Hindi, Telugu, Kannada, Malayalam) to English, and English back to Indic languages in real-time. It uses an asynchronous FastAPI WebSocket backend and a low-latency web interface.

## Key Features

* **Real-Time Bidirectional Streaming**: Translates speech in both directions with minimal delay.
* **Advanced AI Pipeline**:
  * **Silero VAD**: Filters out background noise and silence before processing.
  * **Faster-Whisper (Medium)**: Handles Speech-to-Text (STT) optimized for Tamil, including built-in Hallucination filtering and Tanglish (code-switching) detection.
  * **DeepMultilingualPunctuation**: Restores missing punctuation in transcriptions.
  * **IndicTrans2**: Provides neural machine translation between English and multiple Indic languages (Tamil, Hindi, Telugu, Kannada, Malayalam).
  * **IndicF5**: High-fidelity zero-shot Text-to-Speech (TTS) for natural Indic and English audio synthesis.
* **Zero-Copy Audio Streaming**: Uses a custom binary protocol (JSON metadata + PCM raw float32 arrays) over WebSockets to bypass Base64 encoding overhead.
* **Latency Masking & Chunking**: Chunks text based on punctuation to feed the TTS engine in small 3-5 word bursts, reducing perceived latency.
* **VRAM Optimized**: Uses bidirectional lazy loading for large models (IndicTrans2, IndicF5) so they only consume GPU memory when actively translating in a specific direction. It runs stably on consumer hardware like an RTX 3050.

## Architecture

1. **Client Audio Capture**: The frontend captures raw microphone PCM data via the Web Audio API and streams it via WebSockets.
2. **Router Service**: Audio is queued and managed by a multi-threaded FastAPI backend without blocking the event loop.
3. **VAD & STT**: Silero VAD drops silence, and Faster-Whisper transcribes speech into text.
4. **Punctuation & Chunking**: The text is punctuated and divided into small semantic chunks.
5. **Translation**: IndicTrans2 translates the chunked text.
6. **TTS Synthesis**: IndicF5 synthesizes the translated text back into audio.
7. **Client Playback**: Audio chunks are streamed back via binary WebSocket frames for continuous playback.

## Tech Stack

* **Backend**: Python, FastAPI, WebSockets, PyTorch, asyncio
* **Frontend**: Vanilla HTML/JS/CSS, Web Audio API
* **AI Models**: Faster-Whisper, IndicTrans2, IndicF5, Silero VAD, DeepMultilingualPunctuation

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NamashivayamS/RealTimeSpeechTranslator.git
cd RealTimeSpeechTranslator
```

### 2. Install Dependencies
Ensure you have Python 3.10+ and the CUDA toolkit installed for GPU acceleration.
```bash
python -m venv venv
venv\Scripts\activate  # On Windows (use `source venv/bin/activate` on Mac/Linux)
pip install -r requirements.txt
```

### 3. Run the Backend
```bash
python run_router.py
```
*The server will start and wait for WebSocket connections on `ws://localhost:8000/ws/translate`.*

### 4. Launch the Frontend
Open `frontend/index.html` in a web browser or serve it using a simple HTTP server like VSCode Live Server.

## Project Structure

```text
RealTimeSpeechTranslator/
├── backend/          # FastAPI app and WebSocket handlers
├── frontend/         # UI, Web Audio API integration
├── models/           # AI Model wrappers (Whisper, IndicTrans, TTS)
├── services/         # Core business logic (STT, VAD, Chunking, Routing)
├── utils/            # Helper functions, formatters, audio conversions
├── run_router.py     # Main application entry point
└── requirements.txt  # Python dependencies
```

## Notes for Windows/Nvidia Users
PyTorch CUDA context crashes on Windows can occur if CPU and GPU modules aren't imported in a specific order. This project includes safeguards and lazy-loading mechanisms to handle initialization properly and prevent memory leaks and VRAM crashes.

## NOTE
This project is under Development.
