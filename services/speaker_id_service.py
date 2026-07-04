import os
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerIDService:
    """
    Speaker Identification and Verification Service using SpeechBrain's ECAPA-TDNN.
    Extracts speaker embeddings from audio signals and performs similarity matching.
    """

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = None):
        if device is None:
            self.device = "cpu"
        else:
            self.device = device
            
        print(f"SpeakerIDService: Loading model '{model_name}' on {self.device}...")
        
        # Load pre-trained ECAPA-TDNN speaker encoder model
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": self.device}
        )
        print("SpeakerIDService: Model loaded successfully.")
        
        # In-memory speaker profile store mapping: speaker_id -> embedding (numpy array)
        self.profiles = {}

    def get_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding (192-dimensional vector) from audio chunk.
        
        Args:
            audio: 1-D float32 numpy array.
            sample_rate: The sample rate of the input audio.
            
        Returns:
            1-D numpy array of size 192 representing the speaker embedding.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

        if len(audio) == 0:
            raise ValueError("Input audio is empty")

        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Resample to 16000 Hz if necessary (SpeechBrain models expect 16kHz)
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        # Ensure signal is 2D: [batch, time]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Send tensor to the appropriate device
        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            # Get embeddings from the classifier
            embeddings = self.classifier.encode_batch(audio_tensor)
            # Squeeze to get 1D vector and convert to numpy on CPU
            emb_numpy = embeddings.squeeze().cpu().numpy()

        return emb_numpy

    def enroll_speaker(self, speaker_id: str, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Extract embedding from the audio and register it for a speaker_id.
        If the speaker already exists, average the new embedding with the old one
        to create a more robust speaker profile.
        
        Args:
            speaker_id: A unique string identifier for the speaker.
            audio: 1-D float32 numpy array.
            sample_rate: The sample rate of the input audio.
        """
        try:
            new_emb = self.get_embedding(audio, sample_rate)
            
            if speaker_id in self.profiles:
                # Average embeddings for multiple enrollments
                old_emb = self.profiles[speaker_id]
                avg_emb = (old_emb + new_emb) / 2.0
                # Re-normalize to unit length
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    self.profiles[speaker_id] = avg_emb / norm
                else:
                    self.profiles[speaker_id] = avg_emb
                print(f"SpeakerIDService: Updated profile for speaker '{speaker_id}' (averaged embeddings).")
            else:
                # Normalize to unit length
                norm = np.linalg.norm(new_emb)
                if norm > 0:
                    self.profiles[speaker_id] = new_emb / norm
                else:
                    self.profiles[speaker_id] = new_emb
                print(f"SpeakerIDService: Enrolled speaker '{speaker_id}' successfully.")
            return True
        except Exception as e:
            print(f"SpeakerIDService Error: Failed to enroll speaker '{speaker_id}': {e}")
            return False

    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove a speaker profile."""
        if speaker_id in self.profiles:
            del self.profiles[speaker_id]
            print(f"SpeakerIDService: Removed speaker '{speaker_id}'.")
            return True
        return False

    def clear_profiles(self):
        """Clear all enrolled speaker profiles."""
        self.profiles.clear()
        print("SpeakerIDService: All speaker profiles cleared.")

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings."""
        dot_prod = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_prod / (norm1 * norm2))

    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.5) -> dict:
        """
        Identify the speaker from the audio segment by comparing it to all enrolled profiles.
        
        Args:
            audio: 1-D float32 numpy array.
            sample_rate: The sample rate of the input audio.
            threshold: Cosine similarity threshold for identifying a known speaker.
            
        Returns:
            dict containing:
                "speaker_id": The matched speaker ID or "unknown" if below threshold.
                "similarity": The cosine similarity score.
                "scores": A dict of scores for all enrolled speakers.
        """
        if not self.profiles:
            return {"speaker_id": "unknown", "similarity": 0.0, "scores": {}}

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            
            scores = {}
            for spk_id, ref_emb in self.profiles.items():
                sim = self.compute_similarity(test_emb, ref_emb)
                scores[spk_id] = sim

            # Find speaker with maximum similarity
            best_spk = max(scores, key=scores.get)
            best_score = scores[best_spk]

            if best_score >= threshold:
                matched_spk = best_spk
            else:
                matched_spk = "unknown"

            return {
                "speaker_id": matched_spk,
                "similarity": best_score,
                "scores": scores
            }
        except Exception as e:
            print(f"SpeakerIDService Error: Speaker identification failed: {e}")
            return {"speaker_id": "unknown", "similarity": 0.0, "scores": {}, "error": str(e)}

    def verify_speaker(self, speaker_id: str, audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.5) -> dict:
        """
        Verify whether the speaker of the audio segment matches a specific claimed speaker_id.
        
        Args:
            speaker_id: The claimed speaker ID.
            audio: 1-D float32 numpy array.
            sample_rate: The sample rate of the input audio.
            threshold: Cosine similarity threshold for verification.
        """
        if speaker_id not in self.profiles:
            return {"verified": False, "similarity": 0.0, "error": f"Speaker '{speaker_id}' is not enrolled"}

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            ref_emb = self.profiles[speaker_id]
            sim = self.compute_similarity(test_emb, ref_emb)
            
            return {
                "verified": sim >= threshold,
                "similarity": sim,
                "threshold": threshold
            }
        except Exception as e:
            print(f"SpeakerIDService Error: Speaker verification failed: {e}")
            return {"verified": False, "similarity": 0.0, "error": str(e)}


# Quick test when run directly
if __name__ == "__main__":
    import soundfile as sf

    # 1. Initialize Service
    service = SpeakerIDService()

    # 2. Find sample WAV file
    sample_paths = [
        "tests/audio_samples/ref_cropped.wav",
        "tests/audio_samples/reference.wav"
    ]
    
    selected_path = None
    for p in sample_paths:
        if os.path.exists(p):
            selected_path = p
            break

    if selected_path:
        print(f"\n--- Running tests using file: {selected_path} ---")
        audio, sr = sf.read(selected_path)
        print(f"Loaded audio shape: {audio.shape}, Sample rate: {sr}")
        
        # If stereo, convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Convert to float32
        audio = audio.astype(np.float32)

        # Split the audio into two halves to test verification
        midpoint = len(audio) // 2
        first_half = audio[:midpoint]
        second_half = audio[midpoint:]

        # Test Enrollment
        print("\n[Test 1] Enrolling speaker 'speaker_A' using first half of the audio...")
        enroll_ok = service.enroll_speaker("speaker_A", first_half, sample_rate=sr)
        
        if enroll_ok:
            # Test Identification (same speaker, second half)
            print("\n[Test 2] Identifying speaker of the second half...")
            result = service.identify_speaker(second_half, sample_rate=sr, threshold=0.25)
            print(f"Identification Result: {result}")

            # Test Verification
            print("\n[Test 3] Verifying speaker_A identity on second half...")
            verify_result = service.verify_speaker("speaker_A", second_half, sample_rate=sr, threshold=0.25)
            print(f"Verification Result: {verify_result}")

            # Test with dummy/noise array
            print("\n[Test 4] Identifying a random noise signal (should be unknown)...")
            noise = np.random.randn(sr * 3).astype(np.float32) * 0.01
            noise_result = service.identify_speaker(noise, sample_rate=sr, threshold=0.25)
            print(f"Noise Identification Result: {noise_result}")
    else:
        print("\nNo test wav files found. Running quick test with synthetic signals...")
        # Synthesize simple tone signals
        t = np.linspace(0, 3, 16000 * 3)
        signal_A1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        signal_A2 = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t) * 0.2).astype(np.float32)
        signal_B = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        print("\n[Test 1] Enrolling speaker 'tone_A'...")
        service.enroll_speaker("tone_A", signal_A1, sample_rate=16000)

        print("\n[Test 2] Identifying 'tone_A' variation...")
        result = service.identify_speaker(signal_A2, sample_rate=16000, threshold=0.20)
        print(f"Identification Result: {result}")

        print("\n[Test 3] Identifying a completely different tone 'tone_B'...")
        result_diff = service.identify_speaker(signal_B, sample_rate=16000, threshold=0.20)
        print(f"Different Tone Identification Result: {result_diff}")
