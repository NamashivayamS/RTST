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
