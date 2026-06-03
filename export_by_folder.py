import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(PROJECT_ROOT, "Project_Code_Export")

# Folders we care about
TARGET_FOLDERS = [
    "models",
    "services",
    "backend",
    "frontend",
    "utils",
    "tests",
]

# Exclusions
EXCLUDE_DIRS = {"__pycache__", "venv", ".git", "requirements", "audio"}
EXCLUDE_EXTS = {".wav", ".pt", ".bin", ".pyc", ".json"}

def generate_tree(dir_path, prefix=""):
    tree_str = ""
    try:
        items = sorted(os.listdir(dir_path))
    except Exception:
        return ""
    
    # Filter out excluded dirs
    items = [i for i in items if i not in EXCLUDE_DIRS]
    
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        if os.path.isdir(path):
            tree_str += f"{prefix}{connector}{item}/\n"
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix=prefix + extension)
        else:
            if not any(item.endswith(ext) for ext in EXCLUDE_EXTS):
                tree_str += f"{prefix}{connector}{item}\n"
    return tree_str

def export_code():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    print("Exporting folder-wise code...")
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(PROJECT_ROOT, folder)
        if not os.path.exists(folder_path):
            continue
            
        export_file = os.path.join(EXPORT_DIR, f"{folder}.py")
        if folder == "frontend":
            export_file = os.path.join(EXPORT_DIR, f"{folder}.html")
            
        with open(export_file, "w", encoding="utf-8") as f:
            for root, _, files in os.walk(folder_path):
                # Skip excluded dirs
                if any(ex in root for ex in EXCLUDE_DIRS):
                    continue
                    
                for file in files:
                    if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                        continue
                        
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    
                    f.write(f"\n{'='*60}\n")
                    f.write(f"FILE: {rel_path}\n")
                    f.write(f"{'='*60}\n\n")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as source_file:
                            f.write(source_file.read())
                    except Exception as e:
                        f.write(f"Error reading file: {e}\n")
                    f.write("\n")
        print(f"Created: {export_file}")

def generate_report():
    print("Generating comprehensive report...")
    report_path = os.path.join(EXPORT_DIR, "Comprehensive_Project_Report.md")
    
    tree = f"RealTimeSpeechTranslator/\n{generate_tree(PROJECT_ROOT)}"
    
    report_content = f"""# Real-Time Speech Translator: Project Report

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
{tree}
```

## 4. Current Status
**Status:** Backend completely finalized; Frontend fully integrated.
The pipeline is currently stable on Windows environments with NVIDIA GPUs (tested on RTX 3050). The system successfully handles bidirectional streaming audio chunks and masks generation latency via concurrent TTS queuing.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Created: {report_path}")

if __name__ == "__main__":
    export_code()
    generate_report()
    print("\nExport completed successfully! Check the 'Project_Code_Export' folder.")
