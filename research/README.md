# RTST Research and Development Workspace

This directory is isolated for research, cloud notebook execution, custom model experimentation, and benchmarking. 

## Folder Structure

```text
research/
│
├── README.md                   # This workspace documentation
│
├── kaggle/
│   ├── notebooks/              # Minimal Jupyter notebooks for running trials
│   │   └── 00_Setup.ipynb      # One-click environment bootstrap notebook
│   │
│   ├── scripts/                # Modular Python modules containing setup logic
│   │   ├── __init__.py
│   │   ├── bootstrap.py        # Central setup coordinator
│   │   ├── environment.py      # Hardware (GPU, CUDA, PyTorch) checking utilities
│   │   ├── installer.py        # Kaggle-specific dependency management
│   │   ├── git_sync.py         # Automated Git synchronization
│   │   └── logger.py           # Experiment logging and telemetry
│   │
│   └── configs/                # Configuration and environment variables
│       ├── requirements_kaggle.txt
│       └── kaggle.env
│
├── cache/                      # Cached assets (models, weights, intermediate data)
├── experiments/                # JSON/CSV outputs tracking metric logs, WER, latency
├── outputs/                    # Temporary output files (audio files, visualizations)
└── logs/                       # System and runtime logs
```

## Core Principles

1. **Never Develop in Notebooks**: Notebooks are used strictly for triggering execution and visualizing results. All primary logic, model enhancements, and helper functions must be built in standard Python modules (`.py` files) in the repository.
2. **Git as the Single Source of Truth**: The notebook environment is synchronized via Git. Always push changes to GitHub from your local workspace first, and then run/pull them inside Kaggle.
3. **Keep Production Isolated**: Experimental pipelines, training routines, and performance tests reside exclusively inside `research/` to keep root production services clean.

## Workflow

```text
[Local Machine]                 [GitHub]                    [Kaggle GPU Environment]
    │                              │                                  │
    │ 1. Modify/Enhance Code       │                                  │
    │ 2. Push changes  ────────────>                                  │
    │                              │ 3. Fetch/Pull code via Git  ─────>
    │                              │                                  │ 4. Run setup & evaluation
```

## How to Start

1. Open a new notebook session on Kaggle.
2. Import the setup notebook `research/kaggle/notebooks/00_Setup.ipynb`.
3. Run the setup cell. This automatically:
   - Detects GPU capabilities and resource limits.
   - Installs missing dependencies from `requirements_kaggle.txt`.
   - Clones or updates the project repository to sync your latest commits.
   - Restores the cached model files.

## Future Experiments

- **STT Models**: Benchmarking Whisper Tiny, Base, Small, and Medium models for Tamil ASR transcription (Latency vs. Word Error Rate).
- **Translation Models**: Comparative evaluation of NLLB-200 and IndicTrans2 for Tamil-English and English-Tamil translations.
- **Speaker Identification**: Profiling the speed and classification accuracy of PyAnnote/custom speaker embedding clustering.
- **Text-to-Speech (TTS)**: Testing latency and quality of F5-TTS, MeloTTS, and other fast TTS pipelines.
- **Memory & Latency Profiling**: Logging benchmark results to `experiments/history.csv` across various commits to prevent regression.
