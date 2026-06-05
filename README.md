# 🎙️ Real-Time Speech Translator (Tamil <-> English)

A high-performance, bidirectional streaming pipeline designed to translate spoken Tamil to English, and English to Tamil in **real-time**. Built with a fully asynchronous FastAPI WebSocket backend and a dynamic, low-latency web UI.

## ✨ Key Features

* **⚡ Real-Time Bidirectional Streaming**: Seamlessly translate speech in both directions with near-instant playback.
* **🧠 Advanced AI Pipeline**:
  * **Silero VAD**: Automatically filters out background noise and silence before processing.
  * **Faster-Whisper (Medium)**: Highly accurate Speech-to-Text (STT) optimized for Tamil consonant clusters, featuring built-in Hallucination filtering and Tanglish (code-switching) detection.
  * **DeepMultilingualPunctuation**: Restores missing punctuation on the fly.
  * **IndicTrans2**: State-of-the-art neural machine translation between English and Tamil.
  * **IndicF5**: High-fidelity zero-shot Text-to-Speech (TTS) for natural Tamil/English audio synthesis.
* **🚀 Zero-Copy Audio Streaming**: Uses a custom Two-Frame binary protocol (JSON metadata + PCM raw float32 arrays) over WebSockets, bypassing slow Base64 encoding completely.
* **⏱️ Latency Masking & Chunking**: Intelligently chunks text based on punctuation to feed the TTS engine in small 3-5 word bursts, effectively masking generation latency.
* **🛠️ VRAM Optimized**: Employs bidirectional lazy loading for massive models (IndicTrans2, IndicF5) so they only consume GPU memory when actively translating in a specific direction. Tested and stable on consumer hardware (e.g., RTX 3050).

## 🏗️ Architecture

1. **Client Audio Capture**: Frontend captures raw microphone PCM data via the Web Audio API and streams it via WebSockets.
2. **Router Service (Producer/Consumer)**: Audio is queued and managed by a multi-threaded FastAPI backend without blocking the event loop.
3. **VAD & STT**: Silero VAD drops silence ➔ Faster-Whisper transcribes speech into text.
4. **Punctuation & Chunking**: Text is punctuated and intelligently divided into small semantic chunks.
5. **Translation**: IndicTrans2 translates the chunked text.
6. **TTS Synthesis**: IndicF5 synthesizes the translated text back into audio.
7. **Client Playback**: Audio chunks are streamed back via binary WebSocket frames for smooth, continuous playback.

## 💻 Tech Stack

* **Backend**: Python, FastAPI, WebSockets, PyTorch, asyncio
* **Frontend**: Vanilla HTML/JS/CSS, Web Audio API
* **AI Models**: Faster-Whisper, IndicTrans2, IndicF5, Silero VAD, DeepMultilingualPunctuation

## 🚀 Quick Start

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
Simply open `frontend/index.html` in your favorite web browser (or serve it using a simple HTTP server like VSCode Live Server).

## 📁 Project Structure

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

## ⚠️ Notes for Windows/Nvidia Users
PyTorch CUDA context crashes on Windows can be avoided by maintaining strict import ordering between CPU/GPU modules. This project includes specific safeguards and lazy-loading mechanisms to initialize dependencies in the correct order to prevent memory leaks and VRAM crashes.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License
This project is open-source and available under standard open-source licenses.
