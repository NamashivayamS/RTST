import re
import threading
import torch
import numpy as np
import torchaudio.transforms as T
from speechbrain.inference.speaker import EncoderClassifier

# Self-introduction patterns — English + common Tamil/Tanglish phrasings.
# Conservative on purpose: a missed name just means no auto-enrollment for
# that utterance (safe — falls back to identify_speaker), but a wrong name
# baked into meeting minutes is worse than a missed one.
_NAME_PATTERNS = [
    r"\bi'?m\s+([A-Za-z]+)",
    r"\bi am\s+([A-Za-z]+)",
    r"\bmy name is\s+([A-Za-z]+)",
    r"\bmyself\s+([A-Za-z]+)",
    r"\bthis is\s+([A-Za-z]+)\s+speaking",
    # Tamil: only the explicit "என் பெயர்" (my name is) pattern is safe.
    # "நான்" (I) is too common — it appears in nearly every Tamil sentence
    # and would enroll random following words as fake speakers.
    r"என்\s*பெயர்\s*([A-Za-z\u0B80-\u0BFF]+)",
]
_NAME_RE = [re.compile(p, re.IGNORECASE) for p in _NAME_PATTERNS]

# Words that commonly follow "I'm" / "I am" / "myself" but are NOT names —
# without this guard, ordinary sentences like "I'm sure this will work" or
# "I'm happy to help" would auto-enroll a fake speaker named "Sure" or "Happy".
# Same defensive-stopword pattern as correction_service.py's _EXCLUSIONS.
_NON_NAME_WORDS = {
    "sure", "happy", "sorry", "going", "trying", "done", "fine", "glad",
    "okay", "ok", "ready", "not", "still", "also", "just", "here", "there",
    "back", "afraid", "confused", "excited", "looking", "thinking",
    "wondering", "calling", "asking", "working", "leaving", "coming",
    "speaking", "talking", "saying", "sending", "sharing", "presenting",
    "hoping", "planning", "starting", "finishing", "waiting", "listening",
    "checking", "assuming", "guessing", "worried", "concerned", "curious",
    "interested", "certain", "positive", "confident", "aware", "on", "off",
}


class SpeakerIDService:
    """
    Speaker Identification and Verification Service using SpeechBrain's ECAPA-TDNN.

    IMPORTANT — meeting-scoped, not global:
    RouterService is a single shared instance across all WebSocket connections.
    Enrolled voiceprints are therefore stored per `meeting_id`, not in one flat
    dict — otherwise two simultaneous meetings would leak speaker identities
    into each other. Every public method takes `meeting_id` as a required arg.

    Call clear_meeting(meeting_id) when a meeting/connection ends, or enrolled
    profiles will accumulate in memory indefinitely across the server's uptime.
    """

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = None):
        # Default is CPU, not auto-detect — keeps GPU VRAM free for
        # Whisper-Tamil + IndicTrans2. Pass device="cuda" explicitly if you
        # ever want to benchmark GPU placement.
        self.device = device if device is not None else "cpu"

        print(f"SpeakerIDService: Loading model '{model_name}' on {self.device}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": self.device}
        )
        print("SpeakerIDService: Model loaded successfully.")

        # meeting_id -> {speaker_name: normalized embedding}
        self.meeting_profiles: dict[str, dict[str, np.ndarray]] = {}

        # Serializes all profile dict reads/writes. Needed because multiple
        # _run_pipeline tasks (even within the same meeting) can run
        # concurrently via run_in_executor — without this, two near-simultaneous
        # enrollments for the same name can race on the read-modify-write
        # averaging step and silently corrupt the stored embedding.
        # Mirrors the existing stt_lock/translation_lock pattern in RouterService.
        self.lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Embedding extraction
    # ──────────────────────────────────────────────────────────────────

    def get_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dim speaker embedding from a mono float32 audio array."""
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")
        if len(audio) == 0:
            raise ValueError("Input audio is empty")

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(audio_tensor)
            emb_numpy = embeddings.squeeze().cpu().numpy()

        return emb_numpy

    # ──────────────────────────────────────────────────────────────────
    # Meeting-scoped profile management
    # ──────────────────────────────────────────────────────────────────

    def _get_meeting_profiles(self, meeting_id: str) -> dict[str, np.ndarray]:
        if meeting_id not in self.meeting_profiles:
            self.meeting_profiles[meeting_id] = {}
        return self.meeting_profiles[meeting_id]

    def clear_meeting(self, meeting_id: str):
        """Call when a meeting/connection ends — frees enrolled voiceprints."""
        with self.lock:
            removed = self.meeting_profiles.pop(meeting_id, None)
        if removed is not None:
            print(f"SpeakerIDService: Cleared {len(removed)} profile(s) for meeting {meeting_id}.")

    def enroll_speaker(self, speaker_id: str, audio: np.ndarray, sample_rate: int, meeting_id: str) -> bool:
        """
        Extract embedding and register it for `speaker_id` within `meeting_id`.
        Repeated enrollments for the same name are averaged for robustness.
        """
        try:
            # Embedding extraction itself doesn't need the lock (no shared
            # state touched), but the whole read-modify-write on `profiles`
            # must be atomic to avoid two concurrent enrollments for the same
            # name racing on the averaging step.
            new_emb = self.get_embedding(audio, sample_rate)

            with self.lock:
                profiles = self._get_meeting_profiles(meeting_id)
                if speaker_id in profiles:
                    old_emb = profiles[speaker_id]
                    avg_emb = (old_emb + new_emb) / 2.0
                    norm = np.linalg.norm(avg_emb)
                    profiles[speaker_id] = avg_emb / norm if norm > 0 else avg_emb
                    print(f"SpeakerIDService: Updated '{speaker_id}' in meeting {meeting_id} (averaged).")
                else:
                    norm = np.linalg.norm(new_emb)
                    profiles[speaker_id] = new_emb / norm if norm > 0 else new_emb
                    print(f"SpeakerIDService: Enrolled '{speaker_id}' in meeting {meeting_id}.")
            return True
        except Exception as e:
            print(f"SpeakerIDService Error: Failed to enroll '{speaker_id}': {e}")
            return False

    def remove_speaker(self, speaker_id: str, meeting_id: str) -> bool:
        profiles = self.meeting_profiles.get(meeting_id, {})
        if speaker_id in profiles:
            del profiles[speaker_id]
            print(f"SpeakerIDService: Removed '{speaker_id}' from meeting {meeting_id}.")
            return True
        return False

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot_prod = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_prod / (norm1 * norm2))

    def identify_speaker(self, audio: np.ndarray, sample_rate: int, meeting_id: str, threshold: float = 0.5) -> dict:
        """
        Identify the speaker by comparing against profiles enrolled in THIS
        meeting only. Returns speaker_id="unknown" if no enrolled profile
        clears the threshold — including when nobody has been enrolled yet.
        """
        with self.lock:
            profiles = dict(self.meeting_profiles.get(meeting_id, {}))  # snapshot
        if not profiles:
            return {"speaker_id": "unknown", "similarity": 0.0, "scores": {}}

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            scores = {spk_id: self.compute_similarity(test_emb, ref_emb)
                      for spk_id, ref_emb in profiles.items()}

            best_spk = max(scores, key=scores.get)
            best_score = scores[best_spk]
            matched_spk = best_spk if best_score >= threshold else "unknown"

            return {"speaker_id": matched_spk, "similarity": best_score, "scores": scores}
        except Exception as e:
            print(f"SpeakerIDService Error: Identification failed: {e}")
            return {"speaker_id": "unknown", "similarity": 0.0, "scores": {}, "error": str(e)}

    def verify_speaker(self, speaker_id: str, audio: np.ndarray, sample_rate: int, meeting_id: str, threshold: float = 0.5) -> dict:
        with self.lock:
            profiles = dict(self.meeting_profiles.get(meeting_id, {}))  # snapshot
        if speaker_id not in profiles:
            return {"verified": False, "similarity": 0.0, "error": f"'{speaker_id}' not enrolled in this meeting"}

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            sim = self.compute_similarity(test_emb, profiles[speaker_id])
            return {"verified": sim >= threshold, "similarity": sim, "threshold": threshold}
        except Exception as e:
            print(f"SpeakerIDService Error: Verification failed: {e}")
            return {"verified": False, "similarity": 0.0, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────
    # Name extraction from STT text (drives auto-enrollment)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_name_from_text(text: str) -> str | None:
        """
        Looks for self-introduction patterns in STT output.
        Returns the extracted name (title-cased) or None if nothing matched.
        """
        if not text:
            return None
        for pattern in _NAME_RE:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                if len(name) >= 2 and name.lower() not in _NON_NAME_WORDS:
                    return name.title()
        return None
