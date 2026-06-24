import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from config import VAD_THRESHOLD

class VADService:
    """
    Voice Activity Detection service using Silero VAD.
    Detects speech segments in raw audio, filtering out silence.
    Runs on CPU intentionally to preserve GPU VRAM for Whisper and TTS.
    """

    def __init__(self, sampling_rate: int = 16000):
        print("VADService: Loading Silero VAD model...")
        self.model = load_silero_vad()
        self.sampling_rate = sampling_rate
        print("VADService: Silero VAD loaded successfully.")

    def get_speech_segments(
        self,
        audio: np.ndarray,
        return_seconds: bool = True
    ) -> list[dict]:
        """
        Detects speech segments in a numpy audio array.

        Args:
            audio:          1-D float32 numpy array at self.sampling_rate.
            return_seconds: If True, timestamps are in seconds (float).
                            If False, timestamps are in samples (int).

        Returns:
            List of dicts with 'start' and 'end' keys, e.g.:
            [{'start': 0.32, 'end': 2.88}, ...]
        """
        if audio.ndim != 1:
            raise ValueError(
                f"Expected 1-D audio array, got shape {audio.shape}."
            )

        wav_tensor = torch.tensor(audio, dtype=torch.float32)

        """segments = get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=return_seconds
        )"""
        segments = get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=return_seconds,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=200
        )

        return segments

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Returns True if any speech is detected in the audio chunk.
        Useful as a quick gate before sending audio to Whisper.
        """
        segments = self.get_speech_segments(audio)
        return len(segments) > 0

    def extract_speech_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Returns a new array containing only the speech portions,
        with silence stripped out. Useful for reducing Whisper input length.
        """
        # Get segments in samples (not seconds) for slicing
        segments = self.get_speech_segments(audio, return_seconds=False)

        if not segments:
            return np.array([], dtype=np.float32)

        parts = [audio[seg["start"]: seg["end"]] for seg in segments]
        return np.concatenate(parts)


# Quick test when run directly
if __name__ == "__main__":
    import soundfile as sf

    service = VADService()

    # Replace with any available .wav for a quick sanity check
    audio_path = "tests/audio_samples/ref_cropped.wav"
    audio, sr = sf.read(audio_path)

    print(f"Audio shape: {audio.shape}, sample rate: {sr}")
    segments = service.get_speech_segments(audio.astype(np.float32))

    print(f"\nDetected {len(segments)} speech segment(s):")
    for seg in segments:
        print(f"  {seg}")

    print(f"\nHas speech: {service.has_speech(audio.astype(np.float32))}")
