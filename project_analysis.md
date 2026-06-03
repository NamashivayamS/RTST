# Real-Time Speech Translator - Project Analysis

## 📁 Directory Structure
```text
RealTimeSpeechTranslator/
    ├── benchmark_tts.py
    ├── run_router.py
    ├── scratch_debug_router.py
    ├── transcribe_tamil.py
    ├── backend/
    ├── frontend/
    ├── models/
        ├── indictrans_model.py
        ├── indic_f5_model.py
        ├── model_manager.py
        ├── parler_tts_model.py
        ├── punctuation_model.py
        ├── whisper_model.py
        ├── __init__.py
    ├── services/
        ├── chunking_service.py
        ├── correction_service.py
        ├── punctuation_service.py
        ├── refinement_service.py
        ├── router_service.py
        ├── stt_service.py
        ├── translation_service.py
        ├── tts_service.py
        ├── vad_service.py
        ├── __init__.py
    ├── utils/
        ├── __init__.py
        ├── corrections/
            ├── correction_engine.py
            ├── tamil_corrections.py
            ├── tanglish_corrections.py
            ├── __init__.py
        ├── translation_refinement/
            ├── phrase_dictionary.py
            ├── translation_refiner.py
            ├── __init__.py
```

---
## 💻 Codebase by Folder

### 📂 Folder: `root/`

#### File: `benchmark_tts.py`
```python
import time
import soundfile as sf
import sys
import numpy as np

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading Real-Time Speech Translator Models...")

# Import the integrated TTS model
from models.indic_f5_model import generate_tamil_speech
from transcribe_tamil import transcribe_tamil

# ==========================================
# 1. GPU WARMUP (Simulating Production State)
# ==========================================
print("\nPerforming tiny GPU warmup (to avoid first-run CUDA allocation overhead)...")
_ = generate_tamil_speech("வணக்கம்.")
print("Warmup complete. System is in steady-state production mode.")

# ==========================================
# 2. THE REAL BENCHMARK
# ==========================================
unseen_target_text = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."
print("\n" + "="*50)
print(f"BENCHMARK TARGET: '{unseen_target_text}'")
print("="*50)

# Start High-Precision Timer (only wrapping the TTS generation)
start_time = time.perf_counter()

# Generate the speech
audio_arr, sr = generate_tamil_speech(unseen_target_text)

# End Timer
end_time = time.perf_counter()
latency = end_time - start_time
audio_duration = len(audio_arr) / sr
real_time_factor = latency / audio_duration

print(f"\n⏱️  GENERATION LATENCY: {latency:.2f} seconds")
print(f"🔊  AUDIO DURATION: {audio_duration:.2f} seconds")
print(f"⚡  REAL-TIME FACTOR (RTF): {real_time_factor:.2f}x (Lower is better)")

if real_time_factor < 1.0:
    print("✅ Performance is faster than real-time! Excellent for live translation.")
else:
    print("⚠️ Performance is slower than real-time.")

# ==========================================
# 3. SAVE AND VERIFY
# ==========================================
output_path = "output_realworld_benchmark.wav"
sf.write(output_path, audio_arr, sr)
print(f"\nSaved Audio to: {output_path}")

print("\nVerifying Unseen Pronunciation with Whisper...")
transcription = transcribe_tamil(
    audio_path=output_path,
    model_size="medium",
    device="cuda",
    compute_type="float16"
)

print(f"\n🎯 WHISPER TRANSCRIPTION: '{transcription}'")
print("="*50)

```

#### File: `run_router.py`
```python
import time
import sys

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading core PyTorch libraries first to prevent Windows DLL conflicts...")
import torch
import soundfile as sf

print("Loading services...")
from services.router_service import RouterService

print("Initializing Router...")
router = RouterService()

# Simulate a translation arriving from the STT -> Translation pipeline
test_translation = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."

# Process it instantly (returns almost instantly)
start_t = time.time()
print("\n--- Sending Translation to Router ---")
router.process_translation(test_translation)
print(f"Router returned in {time.time() - start_t:.4f} seconds! (Frontend is unblocked)\n")

# Now simulate the playback system grabbing the audio sequentially
print("Simulating Audio Playback Handler...")
for i in range(3): # We expect 3 chunks based on chunking logic
    print(f"Waiting for audio chunk {i+1} from background thread...")
    
    # Wait for up to 60 seconds for the TTS engine to finish generating the chunk
    audio_payload = router.get_generated_audio(block=True, timeout=60)
    
    if audio_payload is None:
        print("TIMEOUT: Did not receive audio chunk!")
        break
        
    output_file = f"output_stream_chunk_{i+1}.wav"
    sf.write(output_file, audio_payload["audio"], audio_payload["sample_rate"])
    
    print(f"✅ Received Audio Chunk {i+1} for text: '{audio_payload['text']}'")
    print(f"   Saved to {output_file}")
    
print("\nSimulation Complete!")
router.shutdown()

```

#### File: `scratch_debug_router.py`
```python
import sys
import traceback

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    print("Attempting to import RouterService...")
    from services.router_service import RouterService
    print("Import successful!")
except Exception as e:
    print(f"IMPORT ERROR CAUGHT: {e}")
    traceback.print_exc(file=sys.stdout)

```

#### File: `transcribe_tamil.py`
```python
#!/usr/bin/env python3
"""
Tamil Audio Transcription Script using Faster-Whisper.
This script transcribes Tamil audio files to text using the Whisper model.
It is self-contained and does not modify any existing files or dependencies.
"""

import os
import sys
import time
import argparse
import torch
from faster_whisper import WhisperModel

# Ensure stdout and stderr support UTF-8 encoding (especially critical on Windows terminals)
# to avoid UnicodeEncodeError when printing Tamil characters or emojis.
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS,mmm or MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_tamil(
    audio_path: str,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    language: str = "ta",
    output_txt_path: str = None
):
    """
    Transcribes the given audio file using Faster-Whisper.
    """
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    print("=" * 60)
    print("🎙️  TAMIL AUDIO TRANSCRIPTION SYSTEM (WHISPER)")
    print("=" * 60)
    print(f"📁 Input Audio File : {audio_path}")
    print(f"🤖 Whisper Model    : {model_size}")
    print(f"🗣️  Target Language  : {language if language else 'Auto-detect'}")
    print(f"🎯 Beam Size        : {beam_size}")

    # Determine Device
    if device == "auto":
        determined_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        determined_device = device
    
    # Determine Compute Type
    if compute_type == "auto":
        # float16 is recommended for CUDA, int8 or float32 for CPU
        determined_compute = "float16" if determined_device == "cuda" else "int8"
    else:
        determined_compute = compute_type

    print(f"💻 Running on       : {determined_device.upper()} (Compute: {determined_compute})")
    print("-" * 60)

    # Load model
    print("⏳ Loading Whisper model into memory... (This may take a moment on the first run)")
    start_load = time.time()
    try:
        model = WhisperModel(
            model_size,
            device=determined_device,
            compute_type=determined_compute
        )
        print(f"✅ Model loaded successfully in {time.time() - start_load:.2f} seconds!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Suggestion: If CUDA out of memory, try a smaller model like 'small' or 'base', or run on CPU.")
        sys.exit(1)

    print("-" * 60)
    print("🚀 Starting Transcription...")
    start_transcribe = time.time()

    try:
        # Transcribe
        segments, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language
        )

        # info is a named tuple with language, language_probability, duration, duration_after_vad, etc.
        audio_duration = info.duration
        detected_lang = info.language
        detected_prob = info.language_probability

        print(f"📝 Language Detected: '{detected_lang}' with confidence {detected_prob:.2%}")
        print(f"⏱️  Audio Duration   : {audio_duration:.2f} seconds")
        print("\n--- SEGMENTS TRANSCRIPTION ---")

        # We need to iterate over segments to perform transcription.
        # segments is a generator, so the actual transcription happens as we iterate.
        all_text = []
        for segment in segments:
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            segment_text = segment.text.strip()
            all_text.append(segment_text)
            print(f"[{start_str} -> {end_str}] {segment_text}")

        full_transcription = " ".join(all_text)
        end_transcribe = time.time()
        elapsed_time = end_transcribe - start_transcribe

        print("-" * 60)
        print("✅ Transcription Completed!")
        print(f"⏱️  Time Taken       : {elapsed_time:.2f} seconds")
        print(f"⚡ Speed Factor     : {audio_duration / elapsed_time:.2f}x Real-time")
        print("-" * 60)
        
        print("\n📋 FULL TRANSCRIPTION TEXT:")
        print(full_transcription)
        print("-" * 60)

        # Output to file
        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Audio File: {audio_path}\n")
                f.write(f"Model: {model_size}\n")
                f.write(f"Detected Language: {detected_lang} ({detected_prob:.2%})\n")
                f.write(f"Duration: {audio_duration:.2f}s\n")
                f.write("=" * 40 + "\n\n")
                f.write(full_transcription)
            print(f"💾 Saved transcription to: {output_txt_path}")
            print("-" * 60)

        return full_transcription

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe Tamil audio using Faster-Whisper.")
    parser.add_argument(
        "--audio", 
        type=str, 
        default=r"D:\NEED\Sem\Sem 7\Ramraj Intern\Tamilsound\11\tag_01469_01287667058.wav",
        help="Path to the audio file to transcribe."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        help="Whisper model size to use (default: medium)."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto)."
    )
    parser.add_argument(
        "--compute_type", 
        type=str, 
        default="auto",
        choices=["auto", "float16", "float32", "int8"],
        help="Compute type for model precision (default: auto)."
    )
    parser.add_argument(
        "--beam_size", 
        type=int, 
        default=5,
        help="Beam size for transcription search (default: 5)."
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="ta",
        help="Language code of the audio (default: ta for Tamil, set to '' for auto-detect)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save the transcription text file. (default: none)."
    )

    args = parser.parse_args()
    
    # Resolve relative audio path to absolute path for safety if needed,
    # or keep it as relative.
    audio_path = args.audio
    if not os.path.isabs(audio_path):
        audio_path = os.path.abspath(audio_path)

    output_path = args.output
    if output_path is None:
        # Save as the audio filename with .txt in the same directory, or root
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(os.getcwd(), f"{base_name}_transcription.txt")

    transcribe_tamil(
        audio_path=audio_path,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language if args.language else None,
        output_txt_path=output_path
    )

```

---
### 📂 Folder: `models/`

#### File: `indictrans_model.py`
```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from IndicTransToolkit.processor import (
    IndicProcessor
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading IndicTrans2 Model...")

MODEL_NAME = (
    "ai4bharat/"
    "indictrans2-en-indic-dist-200M"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

translation_model = (
    AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(DEVICE)
)

print("IndicTrans2 Model Loaded Successfully!")

print("\nLoading IndicProcessor...")

ip = IndicProcessor(
    inference=True
)

print("IndicProcessor Loaded Successfully!")
```

#### File: `indic_f5_model.py`
```python
import torch
from transformers import AutoModel

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading AI4Bharat IndicF5 Model...")

# This loads the actual Tamil/multilingual-Indian model
f5_tts = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True,
    remove_sil=False
).to(DEVICE)
f5_tts.config.remove_sil = False

print("IndicF5 Loaded Successfully!")

import os
import soundfile as sf
import numpy as np
from f5_tts.infer.utils_infer import infer_process

# Pre-load the reference audio once to save time on every generation call
# Using a path relative to this file to ensure it works from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "tests", "audio_samples", "ref_cropped.wav")
REF_TEXT = "தாமிரபரணி ஆற்றின் கரையுரங்களில் வசிக்கும்."

# Load raw reference audio into memory (bypassing silence trimming)
_ref_audio_arr, _ref_sr = sf.read(REF_AUDIO_PATH)


def generate_tamil_speech(target_text: str) -> tuple[np.ndarray, int]:
    """
    Generates Tamil speech from text using the perfectly tuned IndicF5 model.
    Returns: (audio_array: np.ndarray, sample_rate: int)
    """
    print(f"Generating Tamil speech for: '{target_text}'...")
    
    # We pass the pre-loaded raw audio array path directly (or rather, the file path)
    # Actually, infer_process accepts a file path string or a tuple of (array, sr).
    # Since we bypassed preprocessing, we can just pass the path.
    audio_arr, final_sample_rate, _ = infer_process(
        REF_AUDIO_PATH,
        REF_TEXT,
        target_text,
        f5_tts.ema_model,
        f5_tts.vocoder,
        mel_spec_type="vocos",
        speed=0.85, # Ideal speed for Tamil byte-density
        device=f5_tts.device,
        fix_duration=None # Let the model calculate perfect physical timing
    )
    
    # Ensure it's a numpy array on CPU
    if hasattr(audio_arr, "cpu"):
        audio_arr = audio_arr.cpu().numpy()
    else:
        audio_arr = np.array(audio_arr)
        
    return audio_arr, final_sample_rate

```

#### File: `model_manager.py`
```python

```

#### File: `parler_tts_model.py`
```python
import torch

from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor
)

from parler_tts import (
    ParlerTTSForConditionalGeneration
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading Indic Parler-TTS...")

TTS_MODEL_NAME = (
    "ai4bharat/indic-parler-tts"
)

tts_model = (
    ParlerTTSForConditionalGeneration
    .from_pretrained(TTS_MODEL_NAME)
    .to(DEVICE)
)

tts_tokenizer = (
    AutoTokenizer.from_pretrained(
        TTS_MODEL_NAME
    )
)

feature_extractor = (
    AutoFeatureExtractor
    .from_pretrained(
        TTS_MODEL_NAME
    )
)

print(
    "Indic Parler-TTS "
    "Loaded Successfully!"
)
```

#### File: `punctuation_model.py`
```python
from deepmultilingualpunctuation import (
    PunctuationModel
)

print("\nLoading Punctuation Model...")

punctuation_model = PunctuationModel()

print(
    "Punctuation Model "
    "Loaded Successfully!"
)
```

#### File: `whisper_model.py`
```python
from faster_whisper import WhisperModel
import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading Faster-Whisper Model...")

whisper_model = WhisperModel(
    "small",
    device=DEVICE,
    compute_type="float16"
)

print("Whisper Model Loaded Successfully!")
```

#### File: `__init__.py`
```python

```

---
### 📂 Folder: `services/`

#### File: `chunking_service.py`
```python
import re

class ChunkingService:
    def __init__(self, min_words: int = 3, max_words: int = 5):
        """
        Initializes the chunking service with safe word limits for the TTS engine.
        IndicF5 performs best on 3-5 word chunks.
        """
        self.min_words = min_words
        self.max_words = max_words

    def split_text_for_tts(self, text: str) -> list[str]:
        """
        Intelligently splits Tamil text into safe TTS chunks.
        It prioritizes splitting on punctuation (periods, commas) to maintain natural pauses.
        If no punctuation exists within the max_words limit, it forcefully splits at max_words.
        """
        if not text:
            return []

        # Split the text by any whitespace while preserving the words
        words = text.split()
        chunks = []
        current_chunk = []

        # Common Tamil sentence ending punctuation
        punctuation_marks = {'.', ',', '!', '?', ';', ':', '।'}

        for word in words:
            current_chunk.append(word)
            
            # Check if the word ends with a punctuation mark
            has_punctuation = any(word.endswith(p) for p in punctuation_marks)
            
            # Condition 1: We hit a natural pause (punctuation) and have at least enough words (or it's the end of a thought)
            # We allow chunks shorter than min_words if they end in punctuation because it's a natural breath.
            if has_punctuation and len(current_chunk) > 0:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                continue

            # Condition 2: We hit the maximum word limit, we MUST split to prevent TTS degradation
            if len(current_chunk) >= self.max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Add any remaining words as the final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

# Quick test when run directly
if __name__ == "__main__":
    service = ChunkingService(min_words=3, max_words=5)
    test_text = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."
    
    print(f"Original Text: {test_text}\n")
    print("Chunks:")
    chunks = service.split_text_for_tts(test_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: '{chunk}' (Words: {len(chunk.split())})")

```

#### File: `correction_service.py`
```python

```

#### File: `punctuation_service.py`
```python

```

#### File: `refinement_service.py`
```python

```

#### File: `router_service.py`
```python
import threading
import queue
import time
import sys
import os
from typing import Generator

# Ensure the root project directory is in the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.chunking_service import ChunkingService
from services.tts_service import TTSService

class RouterService:
    """
    The core orchestrator of the Real-Time Speech Translator backend.
    It receives translated text, streams subtitles instantly, chunks the text,
    and dispatches audio generation to a background worker thread to mask latency.
    """
    def __init__(self):
        self.chunking_service = ChunkingService(min_words=3, max_words=5)
        
        # Pre-load the TTS service in the MAIN thread to avoid CUDA context crashes!
        print("RouterService: Initializing TTS Service in main thread...")
        self.tts_service = TTSService()
        
        # Queues for orchestration
        self.tts_input_queue = queue.Queue()
        self.audio_output_queue = queue.Queue()
        
        # Start the background worker thread immediately (Non-daemon to prevent CUDA crashes on Windows)
        self.worker_thread = threading.Thread(target=self._tts_worker_loop, daemon=False)
        self.worker_thread.start()
        print("RouterService: Background TTS worker thread started.")

    def shutdown(self):
        """Cleanly shuts down the worker thread."""
        self.tts_input_queue.put("[SHUTDOWN]")
        self.worker_thread.join(timeout=10)
        print("RouterService: Shutdown complete.")

    def _tts_worker_loop(self):
        """
        The background thread that processes text chunks into audio gaplessly.
        """
        print("[TTS Worker] Ready and listening for chunks...")
        
        while True:
            try:
                # Block until a chunk is available
                chunk_text = self.tts_input_queue.get()
                
                # A special token to shut down the thread if needed
                if chunk_text == "[SHUTDOWN]":
                    break
                    
                print(f"[TTS Worker] Generating audio for chunk: '{chunk_text}'")
                audio_arr, sr = self.tts_service.generate_audio(chunk_text)
                
                # Push the generated audio to the output queue for playback/streaming
                self.audio_output_queue.put({
                    "text": chunk_text,
                    "audio": audio_arr,
                    "sample_rate": sr
                })
                
                # Mark the task as done
                self.tts_input_queue.task_done()
                
            except Exception as e:
                print(f"[TTS Worker Error]: {e}")

    def process_translation(self, translated_text: str):
        """
        Main entrypoint for the Translation Service.
        1. Immediately yields the text (simulating subtitle streaming).
        2. Chunks the text.
        3. Queues chunks for background audio generation.
        """
        print(f"\n--- Processing Translation ---")
        
        # 1. Immediate Subtitle Stream
        print(f"🖥️  SUBTITLE STREAM: {translated_text}")
        
        # 2. Chunking
        chunks = self.chunking_service.split_text_for_tts(translated_text)
        
        # 3. Queueing
        for chunk in chunks:
            print(f"📥 Queuing chunk for TTS: '{chunk}'")
            self.tts_input_queue.put(chunk)

    def get_generated_audio(self, block=True, timeout=None):
        """
        Retrieves the next available piece of generated audio from the background worker.
        Used by the playback/websocket handler to stream to the user.
        """
        try:
            return self.audio_output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

# Quick Simulation when run directly
if __name__ == "__main__":
    import soundfile as sf
    import traceback
    
    try:
        router = RouterService()
        
        # Simulate a translation arriving from the STT -> Translation pipeline
        test_translation = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."
        
        # Process it instantly (returns almost instantly)
        start_t = time.time()
        router.process_translation(test_translation)
        print(f"Router returned in {time.time() - start_t:.4f} seconds! (Frontend is unblocked)")
        
        # Now simulate the playback system grabbing the audio sequentially
        print("\nSimulating Audio Playback Handler...")
        for i in range(3): # We expect 3 chunks based on chunking logic
            print("Waiting for next audio chunk from background thread...")
            audio_payload = router.get_generated_audio(block=True, timeout=60)
            
            output_file = f"output_stream_chunk_{i+1}.wav"
            sf.write(output_file, audio_payload["audio"], audio_payload["sample_rate"])
            
            print(f"✅ Received Audio Chunk {i+1} for text: '{audio_payload['text']}'")
            print(f"   Saved to {output_file}")
    except Exception as e:
        print("CRITICAL ERROR IN ROUTER SIMULATION:")
        traceback.print_exc()

```

#### File: `stt_service.py`
```python

```

#### File: `translation_service.py`
```python

```

#### File: `tts_service.py`
```python
import numpy as np
from models.indic_f5_model import generate_tamil_speech

class TTSService:
    """
    Standardized wrapper service for the IndicF5 Text-To-Speech engine.
    This service is designed to be called asynchronously by the background worker thread.
    """
    def __init__(self):
        # Model is already pre-loaded into memory by indic_f5_model import
        print("TTSService initialized and ready.")

    def generate_audio(self, text: str) -> tuple[np.ndarray, int]:
        """
        Generates audio for a given text chunk.
        Returns:
            audio_data (np.ndarray): The raw PCM audio array.
            sample_rate (int): The sample rate (24000 for IndicF5).
        """
        if not text.strip():
            # Return empty audio for empty chunks to prevent model crash
            return np.array([]), 24000
            
        try:
            # Call our heavily optimized and tuned F5 generation function
            audio_arr, sr = generate_tamil_speech(text)
            return audio_arr, sr
        except Exception as e:
            print(f"[TTSService Error] Failed to generate audio for chunk '{text}': {e}")
            # Fallback to silence on failure to keep pipeline alive
            return np.zeros(24000, dtype=np.float32), 24000

# Quick test when run directly
if __name__ == "__main__":
    service = TTSService()
    arr, sr = service.generate_audio("சோதனை.")
    print(f"Generated {len(arr)} samples at {sr}Hz.")

```

#### File: `vad_service.py`
```python

```

#### File: `__init__.py`
```python

```

---
### 📂 Folder: `utils/`

#### File: `__init__.py`
```python

```

---
### 📂 Folder: `utils/corrections/`

#### File: `correction_engine.py`
```python
from utils.corrections.tamil_corrections import TAMIL_CORRECTIONS
from utils.corrections.tanglish_corrections import TANGLISH_CORRECTIONS

# Merge all correction dictionaries
ALL_CORRECTIONS = {
    **TAMIL_CORRECTIONS,
    **TANGLISH_CORRECTIONS
}


def apply_corrections(text):

    # Remove unwanted spaces/newlines
    text = text.strip()

    corrected_text = text

    for wrong, correct in ALL_CORRECTIONS.items():

        corrected_text = corrected_text.replace(
            wrong,
            correct
        )

    return corrected_text
```

#### File: `tamil_corrections.py`
```python
TAMIL_CORRECTIONS = {

    # ===== Common ASR mistakes =====
    "மண்ண": "பண்ண",
    "செஞ்சி": "செய்து",
    "செஞ்ச": "செய்த",
    "போய்ட்டு": "போயிட்டு",
    "வந்துட்டு": "வந்துவிட்டு",
    "இருக்கு": "இருக்கிறது",
    "இருக்க": "இருக்க",
    "கிடைக்கல": "கிடைக்கவில்லை",
    "தெரியல": "தெரியவில்லை",
    "புரியல": "புரியவில்லை",
    "வேணும்": "வேண்டும்",
    "வேணா": "வேண்டாம்",
    "பாத்து": "பார்த்து",
    "சாப்ப்டு": "சாப்பிட்டு",
    "கேக்குது": "கேட்கிறது",
    "சொல்ற": "சொல்கிற",
    "சொல்றேன்": "சொல்கிறேன்",
    "பாக்க": "பார்க்க",
    "பாக்கலாம்": "பார்க்கலாம்",
    "கேக்க": "கேட்க",
    "வச்சு": "வைத்து",
    "எடுத்துட்டு": "எடுத்துவிட்டு",
    "குடுத்து": "கொடுத்து",
    "குடு": "கொடு",
    "என்னடா": "என்ன டா",
    "என்னடி": "என்ன டி",

    # ===== Joined-word corrections =====
    "நடை வேறும்": "நடைபெறும்",
    "பார்த்து கொண்டு": "பார்த்துக்கொண்டு",
    "செய்து கொண்டு": "செய்துக்கொண்டு",
    "பேசி கொண்டு": "பேசிக்கொண்டு",
    "வந்து கொண்டு": "வந்துக்கொண்டு",
    "போய் கொண்டு": "போய்க்கொண்டு",
    "எடுத்து கொண்டு": "எடுத்துக்கொண்டு",
    "கொண்டு இருக்கு": "கொண்டிருக்கிறது",
    "சொல்லி இருந்தேன்": "சொல்லியிருந்தேன்",
    "பண்ணி இருந்தேன்": "பண்ணியிருந்தேன்",
    "கேட்டு கொண்டு": "கேட்டுக்கொண்டு",
    "உட்கார்ந்து கொண்டு": "உட்கார்ந்துக்கொண்டு",

    # ===== Pronunciation corrections =====
    "ஏட்டு": "எட்டு",
    "லச்சம்": "லட்சம்",
    "அவ்ளோ": "அவ்வளவு",
    "இவ்ளோ": "இவ்வளவு",
    "எவ்ளோ": "எவ்வளவு",
    "அப்டி": "அப்படி",
    "இப்டி": "இப்படி",
    "எப்டி": "எப்படி",
    "அதான்": "அதுதான்",
    "இதான்": "இதுதான்",
    "எதான்": "எதுதான்",
    "அங்க": "அங்கு",
    "இங்க": "இங்கு",
    "எங்க": "எங்கு",

    # ===== Spoken → Formal Tamil =====
    "பண்ணு": "செய்",
    "பண்ணிட்டேன்": "செய்துவிட்டேன்",
    "பண்ணிட்டான்": "செய்துவிட்டான்",
    "பண்ணிட்டாங்க": "செய்துவிட்டார்கள்",
    "போறேன்": "போகிறேன்",
    "வரேன்": "வருகிறேன்",
    "சாப்ப்டேன்": "சாப்பிட்டேன்",
    "படிச்சேன்": "படித்தேன்",
    "கேட்டேன்": "கேட்டேன்",
    "பார்த்தேன்": "பார்த்தேன்",
    "குடுத்தேன்": "கொடுத்தேன்",
    "எடுத்தேன்": "எடுத்தேன்",

    # ===== Common OCR / ASR confusions =====
    "னு": "என்று",
    "ன்னு": "என்று",
    "அப்றம்": "அப்புறம்",
    "இப்றம்": "இப்புறம்",
    "ஒண்ணு": "ஒன்று",
    "ரெண்டு": "இரண்டு",
    "மூணு": "மூன்று",
    "நாலு": "நான்கு",
    "அஞ்சு": "ஐந்து",
    "ஆறுu": "ஆறு",
    "ஏழுu": "ஏழு",

    # ===== Numbers pronunciation =====
    "ஒம்பது": "ஒன்பது",
    "பதினொன்னு": "பதினொன்று",
    "பன்னெண்டு": "பன்னிரண்டு",
    "இருவத்தி": "இருபத்து",
    "முப்பத்தி": "முப்பத்து",
    "நாப்பத்தி": "நாற்பத்து",
    "அம்பத்தி": "ஐம்பத்து",

    # ===== Common fillers =====
    "அம்மாா": "அம்மா",
    "சார்ர்": "சார்",
    "ஹலோோ": "ஹலோ",
    "ஓகேே": "ஓகே",
    "ஹ்ம்ம்": "ஹும்",

    # ===== Common sentence fixes =====
    "என்ன ஆச்சு": "என்ன ஆயிற்று",
    "எப்படி இருக்கு": "எப்படி இருக்கிறது",
    "நல்லா இருக்கு": "நன்றாக இருக்கிறது",
    "சரியா இருக்கு": "சரியாக இருக்கிறது",
    "பிரச்சனை இல்ல": "பிரச்சனை இல்லை",
    "ஒன்னும் இல்ல": "ஒன்றும் இல்லை",

    # ===== Time / Date corrections =====
    "நேத்து": "நேற்று",
    "இன்னிக்கு": "இன்று",
    "நாளைக்கு": "நாளை",
    "மறுநாளைக்கு": "மறுநாள்",
    "காலையில": "காலையில்",
    "மாலையில": "மாலையில்",
    "ராத்திரி": "இரவு",

    # ===== Misc =====
    "கிட்ட": "அருகில்",
    "வீட்ல": "வீட்டில்",
    "ஸ்கூல்ல": "பள்ளியில்",
    "கல்லூரில": "கல்லூரியில்",
    "ரோட்ல": "சாலையில்",
    "இங்கிருந்து": "இங்கிருந்து",
    "அங்கிருந்து": "அங்கிருந்து"
}

```

#### File: `tanglish_corrections.py`
```python
TANGLISH_CORRECTIONS = {

    # ===== Stable English loanwords in Tamil script =====
    # Safe because they are usually standalone complete words
    "சோ": "so",
    "மீட்டிங்": "meeting",
    "மீட்டின்": "meeting",
    "ஜாய்ன்": "join",
    "ஜாயின்": "join",
    "ஆபீஸ்": "office",
    "டீம்": "team",
    "மேனேஜர்": "manager",
    "டாஸ்க்": "task",
    "ப்ராஜெக்ட்": "project",
    "இன்டர்வியூ": "interview",
    "ரெஸ்யூம்": "resume",
    "சாலரி": "salary",

    # ===== Tech =====
    "போன்": "phone",
    "மொபைல்": "mobile",
    "லாப்டாப்": "laptop",
    "கம்ப்யூட்டர்": "computer",
    "சிஸ்டம்": "system",
    "சர்வர்": "server",
    "டேட்டா": "data",
    "நெட்வொர்க்": "network",
    "வைஃபை": "wifi",
    "ஹாட்ஸ்பாட்": "hotspot",
    "ப்ளூடூத்": "bluetooth",
    "ரூட்டர்": "router",

    # ===== Programming =====
    "கோடு": "code",
    "கோடிங்": "coding",
    "ப்ரோக்ராம்": "program",
    "ப்ரோக்ராமிங்": "programming",
    "டெவலப்பர்": "developer",
    "டீபக்": "debug",
    "பக்": "bug",
    "எரர்": "error",
    "எக்ஸெப்ஷன்": "exception",
    "கம்பைல்": "compile",
    "பில்ட்": "build",
    "டிப்ளாய்": "deploy",
    "ஃப்ரேம்வொர்க்": "framework",
    "லைப்ரரி": "library",
    "மொட்யூல்": "module",
    "பேக்கேஜ்": "package",
    "டேட்டாபேஸ்": "database",
    "ஏபிஐ": "api",

    # ===== File / Software =====
    "பைல்": "file",
    "போல்டர்": "folder",
    "டாக்குமென்ட்": "document",
    "பிடிஎப்": "pdf",
    "எக்செல்": "excel",
    "வேர்ட்": "word",
    "பிபிடி": "ppt",
    "ப்ரௌசர்": "browser",
    "டெர்மினல்": "terminal",
    "கமாண்ட்": "command",

    # ===== Internet =====
    "குரோம்": "chrome",
    "கூகுள்": "google",
    "யூடியூப்": "youtube",
    "ஜிமெயில்": "gmail",
    "வாட்ஸ்அப்": "whatsapp",
    "டெலிகிராம்": "telegram",
    "இன்ஸ்டாகிராம்": "instagram",
    "லிங்க்டின்": "linkedin",
    "ட்விட்டர்": "twitter",
    "ஃபேஸ்புக்": "facebook",

    # ===== Communication =====
    "கால்": "call",
    "மெசேஜ்": "message",
    "டெக்ஸ்ட்": "text",
    "சாட்": "chat",
    "ரிப்ளை": "reply",
    "மெயில்": "mail",
    "ஈமெயில்": "email",
    "லிங்க்": "link",
    "ஜூம்": "zoom",

    # ===== Actions (safe standalone forms only) =====
    "அப்டேட்": "update",
    "டவுன்லோடு": "download",
    "அப்லோடு": "upload",
    "இன்ஸ்டால்": "install",
    "அன்இன்ஸ்டால்": "uninstall",
    "லாகின்": "login",
    "லாக்அவுட்": "logout",
    "கனெக்ட்": "connect",
    "டிஸ்கனெக்ட்": "disconnect",

    # ===== AI / Data =====
    "ஏஐ": "ai",
    "என்எல்பி": "nlp",
    "சாட்பாட்": "chatbot",
    "மாடல்": "model",
    "டேட்டாசெட்": "dataset",

    # ===== Misc stable words =====
    "டிக்கெட்": "ticket",
    "பேமென்ட்": "payment",
    "பில்": "bill",
    "கேஷ்": "cash",
    "பேங்க்": "bank",
    "அக்கவுண்ட்": "account",
    "பாஸ்வேர்ட்": "password",
    "யூசர்": "user"
}


```

#### File: `__init__.py`
```python

```

---
### 📂 Folder: `utils/translation_refinement/`

#### File: `phrase_dictionary.py`
```python
PHRASE_REPLACEMENTS = {

    "குட் மார்னிங்": "காலை வணக்கம்",

    "ஹலோ": "வணக்கம்",

    "தாங்க் யூ": "நன்றி"
}
```

#### File: `translation_refiner.py`
```python
from utils.translation_refinement.phrase_dictionary import (
    PHRASE_REPLACEMENTS
)

def refine_translation(text):

    refined_text = text

    for wrong, correct in PHRASE_REPLACEMENTS.items():

        refined_text = refined_text.replace(
            wrong,
            correct
        )

    return refined_text
```

#### File: `__init__.py`
```python

```

---
