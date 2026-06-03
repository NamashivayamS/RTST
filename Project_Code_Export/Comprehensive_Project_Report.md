# Real-Time Speech Translator: Project Report

## 1. Project Overview
The **Real-Time Speech Translator** is a highly optimized, bidirectional streaming pipeline designed to translate spoken Tamil to English, and English to Tamil in real-time. It features a fully asynchronous FastAPI WebSocket backend and a dynamic React-based frontend UI.

## 2. What We Have Done Till Now
* **Pipeline Orchestration (`router_service.py`)**: Built a fully multi-threaded, Producer-Consumer architecture that handles streaming audio without blocking the event loop.
* **Latency Masking & Chunking (`chunking_service.py`)**: Implemented intelligent text chunking based on punctuation to feed the TTS engine in small 3-5 word bursts, masking generation latency.
* **VRAM Optimization**: Solved PyTorch CUDA context crashes on Windows by strictly ordering CPU vs GPU imports. Implemented bidirectional lazy loading for massive models like IndicTrans2 and IndicF5 so they only consume memory when actively translating in a specific direction.
* **FastAPI WebSocket Backend (`backend/`)**: Designed a custom Two-Frame binary protocol (JSON metadata + PCM raw float32 arrays) to achieve zero-copy audio streaming, entirely bypassing slow Base64 encoding.
* **AI Models Integrated**:
    1. **Silero VAD**: Filters out background noise and silence.
    2. **Faster-Whisper (Medium)**: Performs Speech-to-Text with high accuracy for Tamil consonant clusters. Includes Hallucination filtering and Tanglish detection.
    3. **DeepMultilingualPunctuation**: Restores missing English punctuation.
    4. **IndicTrans2**: Translates perfectly between English and Tamil.
    5. **IndicF5**: High-fidelity zero-shot TTS model used for voice cloning and final Tamil/English audio synthesis.
* **Frontend UI (`frontend/index.html`)**: Developed a stunning, dark-mode real-time UI utilizing the Web Audio API to record microphone input, visualize waveforms dynamically, and playback streamed audio queue chunks seamlessly.

## 3. Project File Structure
```text
RealTimeSpeechTranslator/
├── .gitignore
├── Project_Code_Export/
│   ├── Comprehensive_Project_Report.md
│   ├── backend.py
│   ├── frontend.html
│   ├── models.py
│   ├── services.py
│   ├── tests.py
│   └── utils.py
├── backend/
│   ├── __init__.py
│   ├── connection_manager.py
│   └── main.py
├── benchmark_tts.py
├── config.py
├── export_by_folder.py
├── frontend/
│   └── index.html
├── models/
│   ├── __init__.py
│   ├── indic_f5_model.py
│   ├── indictrans_model.py
│   ├── model_manager.py
│   ├── parler_tts_model.py
│   ├── punctuation_model.py
│   └── whisper_model.py
├── project_analysis.md
├── requirements.txt
├── router_log.txt
├── run_router.py
├── scratch_debug_router.py
├── services/
│   ├── __init__.py
│   ├── chunking_service.py
│   ├── correction_service.py
│   ├── punctuation_service.py
│   ├── refinement_service.py
│   ├── router_service.py
│   ├── stt_service.py
│   ├── translation_service.py
│   ├── tts_service.py
│   └── vad_service.py
├── temp/
├── tests/
│   ├── audio_samples/
│   ├── backend/
│   │   ├── test_fastapi_import.py
│   │   ├── test_uvicorn.py
│   │   ├── test_websocket.py
│   │   ├── test_websocket_client.py
│   │   └── test_websocket_server.py
│   ├── benchmark_all.py
│   ├── pipeline/
│   │   ├── test_phase1_stt.py
│   │   ├── test_phase2_cleanup.py
│   │   ├── test_phase3_punctuation.py
│   │   ├── test_phase4_translation.py
│   │   └── test_phase5_indic_parler_tts.py
│   ├── stt/
│   │   └── test_whisper.py
│   ├── translation/
│   │   ├── test_muril.py
│   │   ├── test_punctuation.py
│   │   └── test_translation.py
│   ├── tts/
│   │   ├── test_indic_f5_tts.py
│   │   ├── test_indic_parler_tts.py
│   │   └── test_xtts.py
│   └── vad/
│       └── test_vad.py
├── tests_all_code.md
├── transcribe_tamil.py
└── utils/
    ├── __init__.py
    ├── corrections/
    │   ├── __init__.py
    │   ├── correction_engine.py
    │   ├── tamil_corrections.py
    │   └── tanglish_corrections.py
    └── translation_refinement/
        ├── __init__.py
        ├── phrase_dictionary.py
        └── translation_refiner.py

```

## 4. Current Status
**Status:** Backend completely finalized; Frontend fully integrated.
The pipeline is currently stable on Windows environments with NVIDIA GPUs (tested on RTX 3050). The system successfully handles bidirectional streaming audio chunks and masks generation latency via concurrent TTS queuing.
