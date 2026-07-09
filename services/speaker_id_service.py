import re
import threading
import torch
import numpy as np
import torchaudio.transforms as T
import uuid
import logging
from speechbrain.inference.speaker import EncoderClassifier
from database.queries import (
    load_global_speaker_profiles,
    create_global_speaker_profile,
    add_speaker_template,
    evict_speaker_template,
    update_global_speaker_name,
    delete_global_speaker_profile,
)
from config import (
    SPEAKER_ID_LOCAL_THRESHOLD,
    SPEAKER_ID_GLOBAL_THRESHOLD,
    SPEAKER_ID_INTRO_THRESHOLD,
    SPEAKER_ID_MIN_ENROLL_DURATION,
    SPEAKER_ID_DEVICE,
    SPEAKER_ID_MAX_TEMPLATES,
    SPEAKER_ID_TEMPLATE_ACCEPT_SIM,
)

logger = logging.getLogger("ispeak.speaker_id")

# ──────────────────────────────────────────────────────────────────
# SpaCy NER — lazy-loaded on first call to avoid startup penalty.
# The small English model (en_core_web_sm, ~12 MB) runs on CPU and
# processes a typical 15-word sentence in < 2 ms.
# ──────────────────────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    """Lazy-load SpaCy's small English pipeline on first use."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            print("SpeakerIDService: SpaCy NER model 'en_core_web_sm' loaded.")
        except Exception as e:
            print(f"SpeakerIDService: SpaCy NER unavailable ({e}). Falling back to regex-only.")
            _nlp = False  # sentinel — don't retry on every call
    return _nlp if _nlp is not False else None

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
    r"என்\s*(?:பெயர்|பேர்)\s*([A-Za-z\u0B80-\u0BFF]+)",
    r"\bமை\s*நேம்\s*(?:ஈஸ்|இஸ்|இசு)?\s*([A-Za-z\u0B80-\u0BFF]+)",
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
    # Local geographic/corporate terms prone to false-positive NER name extraction
    "rajasthan", "coimbatore", "tirupur", "chennai", "india", "tamil", "nadu",
    "avinas", "avinashi", "kongu", "cotton", "ramraj", "madurai",
}


class SpeakerIDService:
    """
    Speaker Identification and Verification Service using SpeechBrain's ECAPA-TDNN.
    Persists profiles globally in PostgreSQL, keyed by a synthetic UUID.

    NOTE (Known Limitations & Architectural Trade-offs):
    1. Enrollment-Time Blind Spot: When a speaker introduces themselves (e.g. "I am Namas"),
       `enroll_speaker` unconditionally generates a new profile UUID without checking if that
       physical speaker already has an existing profile from a past meeting. Passive matching
       via `identify_speaker` only takes effect for subsequent utterances after someone else has
       spoken and been matched.
    2. Concurrent Meeting Staleness: If two concurrent meetings load profiles from PostgreSQL
       at startup, changes to `self.global_profiles` (like rename or merge) in one meeting won't
       be dynamically synchronized to the other in-memory instance until the second meeting is
       restarted. This avoids aggressive database polling overhead.
    """

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = None):
        if device is not None:
            self.device = device
        elif SPEAKER_ID_DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = SPEAKER_ID_DEVICE

        print(f"SpeakerIDService: Loading model '{model_name}' on {self.device}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": self.device}
        )
        self.model_name = model_name
        print("SpeakerIDService: Model loaded successfully.")

        # Dynamically determine the embedding dimension
        dummy_audio = np.zeros(16000, dtype=np.float32)
        dummy_emb = self.get_embedding(dummy_audio, 16000)
        self.emb_dim = len(dummy_emb)
        print(f"SpeakerIDService: Detected embedding dimension = {self.emb_dim}")

        # meeting_id -> {profile_id_uuid: {"name": name, "embedding": np.ndarray}}
        # NOTE: local meeting profiles use a single embedding per speaker.
        # Global profiles use multi-template: {"name": str, "templates": [dict], "primary_index": int}
        self.meeting_profiles: dict[str, dict[str, dict]] = {}

        # Load global profiles from DB compatible with active model/dimension
        self.global_profiles = load_global_speaker_profiles(self.model_name, self.emb_dim)

        # Warm up/pre-load SpaCy NER pipeline at startup to avoid blocking the event loop on the first utterance
        _get_nlp()

        self.lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Embedding extraction
    # ──────────────────────────────────────────────────────────────────

    def get_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a speaker embedding from a mono float32 audio array."""
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

        import time
        t0 = time.perf_counter()
        with torch.no_grad():
            embeddings = self.classifier.encode_batch(audio_tensor)
            emb_numpy = embeddings.squeeze().cpu().numpy()
        inf_time_ms = (time.perf_counter() - t0) * 1000
        logger.info("[SpeakerID] Inner model inference time: %.1fms on %s", inf_time_ms, self.device)

        return emb_numpy

    # ──────────────────────────────────────────────────────────────────
    # Meeting-scoped profile management
    # ──────────────────────────────────────────────────────────────────

    def _get_meeting_profiles(self, meeting_id: str) -> dict[str, dict]:
        if meeting_id not in self.meeting_profiles:
            self.meeting_profiles[meeting_id] = {}
        return self.meeting_profiles[meeting_id]

    def _best_match(self, emb: np.ndarray, candidates: dict) -> tuple[str | None, str | None, float]:
        """Iterate over single-embedding candidate profiles (local meeting profiles only)."""
        best_id = None
        best_name = None
        best_sim = 0.0
        for p_id, data in candidates.items():
            sim = self.compute_similarity(emb, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = p_id
                best_name = data["name"]
        return best_id, best_name, best_sim

    def _best_match_multi(self, query_emb: np.ndarray, candidates: dict) -> tuple[str | None, str | None, float, int]:
        """
        Multi-template matching against global profiles.
        candidates: {profile_id: {"name": str, "templates": list[dict]}}
        Each template dict has {"template_id": str, "embedding": np.ndarray, "is_primary": bool}.
        Returns (best_id, best_name, best_sim, best_template_index).
        Vectorized: stacks all templates into one matrix, one matmul, finds per-candidate max.
        """
        if not candidates:
            return None, None, 0.0, -1

        all_embs = []
        owner_map = []  # parallel: (profile_id, name, template_index_within_profile)
        for pid, data in candidates.items():
            for t_idx, t in enumerate(data["templates"]):
                all_embs.append(t["embedding"])
                owner_map.append((pid, data["name"], t_idx))

        if not all_embs:
            return None, None, 0.0, -1

        matrix = np.stack(all_embs)          # shape (total_templates, emb_dim)
        sims = matrix @ query_emb            # assumes pre-normalized embeddings
        best_row = int(np.argmax(sims))
        best_pid, best_name, best_t_idx = owner_map[best_row]
        return best_pid, best_name, float(sims[best_row]), best_t_idx

    def _find_outlier_template(self, templates: list[dict], protect_index: int) -> int:
        """
        Returns the index (excluding protect_index) whose average similarity
        to all OTHER templates in the list is lowest — i.e., the one that looks
        least consistent with the rest of this speaker's stored voiceprints.
        """
        n = len(templates)
        avg_sims = []
        for i in range(n):
            if i == protect_index:
                avg_sims.append(float("inf"))  # never selectable
                continue
            sims = [
                self.compute_similarity(templates[i]["embedding"], templates[j]["embedding"])
                for j in range(n) if j != i
            ]
            avg_sims.append(sum(sims) / len(sims) if sims else 0.0)
        return int(np.argmin(avg_sims))

    def clear_meeting(self, meeting_id: str):
        """Call when a meeting/connection ends — frees enrolled voiceprints."""
        with self.lock:
            removed = self.meeting_profiles.pop(meeting_id, None)
        if removed is not None:
            print(f"SpeakerIDService: Cleared {len(removed)} profile(s) for meeting {meeting_id}.")

    def enroll_speaker(self, speaker_name: str, audio: np.ndarray, sample_rate: int, meeting_id: str, threshold: float = SPEAKER_ID_LOCAL_THRESHOLD) -> dict:
        """
        Extract embedding and register it for `speaker_name` within `meeting_id`.
        Before minting a fresh UUID, checks if the voice matches an existing local
        speaker or global speaker (with lenient intro threshold or strict passive threshold)
        to prevent duplicate enrollments.

        Global profiles use multi-template structure:
            {"name": str, "templates": [{"template_id": str, "embedding": np.ndarray, "is_primary": bool}], "primary_index": int}
        """
        try:
            new_emb = self.get_embedding(audio, sample_rate)
            if new_emb is None:
                return {}

            norm = np.linalg.norm(new_emb)
            normalized_emb = new_emb / norm if norm > 0 else new_emb

            promote_to_db = False
            update_in_db = False
            remap_local_key = None       # (old_temp_id, new_id) if a swap is needed
            db_embedding = None
            result_id = None
            result_name = None
            was_pre_existing_local = False
            pre_rename_name = None

            with self.lock:
                profiles = self._get_meeting_profiles(meeting_id)

                # ── Tier A: existing local match ─────────────────────────────
                best_local_id, best_local_name, best_local_sim = self._best_match(normalized_emb, profiles)

                # ── Tier A2: name-filtered global search (multi-template) ─────
                name_matched_candidates = {
                    pid: data for pid, data in self.global_profiles.items()
                    if data["name"].strip().lower() == speaker_name.strip().lower()
                }
                best_intro_id, best_intro_name, best_intro_sim, _ = self._best_match_multi(
                    normalized_emb, name_matched_candidates
                ) if name_matched_candidates else (None, None, 0.0, -1)

                # ── Decision Tree ────────────────────────────────────────────
                if best_local_sim >= threshold:
                    if best_local_name.startswith("Speaker ") and not speaker_name.startswith("Speaker "):
                        if best_intro_sim >= SPEAKER_ID_INTRO_THRESHOLD:
                            # Case 1: Name + voice agree on an existing global identity
                            remap_local_key = (best_local_id, best_intro_id)
                            profiles[best_intro_id] = {
                                "name": speaker_name,
                                "embedding": profiles[best_local_id]["embedding"]
                            }
                            del profiles[best_local_id]
                            result_id, result_name = best_intro_id, speaker_name

                            old_name = self.global_profiles[best_intro_id]["name"]
                            if old_name != speaker_name:
                                self.global_profiles[best_intro_id]["name"] = speaker_name
                                update_in_db = True
                        else:
                            # Case 2: Spoken name mismatch/nickname, check for strict global voice match
                            best_global_id, best_global_name, best_global_sim, _ = self._best_match_multi(
                                normalized_emb, self.global_profiles
                            )
                            if best_global_sim >= SPEAKER_ID_GLOBAL_THRESHOLD:
                                # Voice matches a global profile strictly, trust database name
                                remap_local_key = (best_local_id, best_global_id)
                                profiles[best_global_id] = {
                                    "name": best_global_name,
                                    "embedding": profiles[best_local_id]["embedding"]
                                }
                                del profiles[best_local_id]
                                result_id, result_name = best_global_id, best_global_name
                            else:
                                # Case 3: Genuinely new identity; rename and promote local temp profile
                                assert best_local_id not in self.global_profiles, (
                                    f"Temp ID {best_local_id} already exists in global profiles — "
                                    f"this violates UUID uniqueness"
                                )
                                was_pre_existing_local = True
                                pre_rename_name = profiles[best_local_id]["name"]

                                profiles[best_local_id]["name"] = speaker_name
                                result_id, result_name = best_local_id, speaker_name
                                promote_to_db = True
                                db_embedding = profiles[best_local_id]["embedding"]
                                self.global_profiles[best_local_id] = {
                                    "name": speaker_name,
                                    "templates": [{"template_id": None, "embedding": db_embedding, "is_primary": True}],
                                    "primary_index": 0
                                }
                    else:
                        # Already enrolled under a specific name. Log mismatch if names differ.
                        if best_local_name != speaker_name and not best_local_name.startswith("Speaker "):
                            logger.warning(
                                "[SpeakerID] Mismatch in enroll_speaker: voice matched local profile '%s' (sim=%.3f), "
                                "but spoken name was '%s'. Keeping existing name.",
                                best_local_name, best_local_sim, speaker_name
                            )
                        return {
                            "profile_id": best_local_id,
                            "name": best_local_name,
                            "was_merged": False,
                            "merged_from_id": None
                        }

                elif best_intro_sim >= SPEAKER_ID_INTRO_THRESHOLD:
                    # Matches global profile directly by name + voice (no local anonymous profile existed yet)
                    profiles[best_intro_id] = {"name": speaker_name, "embedding": normalized_emb}
                    result_id, result_name = best_intro_id, speaker_name
                    old_name = self.global_profiles[best_intro_id]["name"]
                    if old_name != speaker_name:
                        self.global_profiles[best_intro_id]["name"] = speaker_name
                        update_in_db = True

                else:
                    # Tier B: Strict, unfiltered global voice match (name-independent safety net)
                    best_global_id, best_global_name, best_global_sim, _ = self._best_match_multi(
                        normalized_emb, self.global_profiles
                    )
                    if best_global_sim >= SPEAKER_ID_GLOBAL_THRESHOLD:
                        # Voice matches a global profile strictly, trust database name
                        profiles[best_global_id] = {"name": best_global_name, "embedding": normalized_emb}
                        result_id, result_name = best_global_id, best_global_name
                    else:
                        # Tier C: Genuinely new speaker
                        provisional_id = str(uuid.uuid4())
                        profiles[provisional_id] = {"name": speaker_name, "embedding": normalized_emb}
                        self.global_profiles[provisional_id] = {
                            "name": speaker_name,
                            "templates": [{"template_id": None, "embedding": normalized_emb, "is_primary": True}],
                            "primary_index": 0
                        }
                        result_id, result_name = provisional_id, speaker_name
                        promote_to_db = True
                        db_embedding = normalized_emb

            # ── Database updates (Outside Lock) ─────────────────────────────
            if promote_to_db:
                # Save the input ID we are passing, which was result_id (either provisional_id or best_local_id)
                input_id = result_id
                try:
                    # Create new profile + primary template atomically in DB
                    new_profile_id = create_global_speaker_profile(
                        result_name, db_embedding, self.model_name, self.emb_dim,
                        profile_id=input_id
                    )
                except Exception:
                    logger.exception(
                        "[SpeakerID] DB persistence failed for provisional profile %s — rolling back in-memory reservation.",
                        input_id
                    )
                    with self.lock:
                        profiles = self._get_meeting_profiles(meeting_id)
                        self.global_profiles.pop(input_id, None)
                        if was_pre_existing_local:
                            if input_id in profiles:
                                profiles[input_id]["name"] = pre_rename_name
                        else:
                            profiles.pop(input_id, None)
                    return {}

                result_id = new_profile_id
                # Build in-memory multi-template structure
                primary_template = {
                    "template_id": None,  # DB-generated, not critical for in-memory matching
                    "embedding": db_embedding,
                    "is_primary": True
                }
                with self.lock:
                    profiles = self._get_meeting_profiles(meeting_id)
                    if new_profile_id != input_id:
                        profiles.pop(input_id, None)
                        self.global_profiles.pop(input_id, None)
                    profiles[new_profile_id] = {"name": result_name, "embedding": db_embedding}
                    self.global_profiles[new_profile_id] = {
                        "name": result_name,
                        "templates": [primary_template],
                        "primary_index": 0
                    }
            elif update_in_db:
                update_global_speaker_name(result_id, result_name)

            return {
                "profile_id": result_id,
                "name": result_name,
                "was_merged": remap_local_key is not None,
                "merged_from_id": remap_local_key[0] if remap_local_key else None
            }
        except Exception as e:
            logger.exception("[SpeakerID] Failed to enroll '%s': %s", speaker_name, e)
            return {}

    def remove_speaker(self, profile_id: str, meeting_id: str) -> bool:
        with self.lock:
            profiles = self.meeting_profiles.get(meeting_id, {})
            if profile_id in profiles:
                del profiles[profile_id]
                print(f"SpeakerIDService: Removed '{profile_id}' from meeting {meeting_id}.")
                return True
        return False

    def rename_speaker(self, profile_id: str, new_name: str, meeting_id: str) -> bool:
        """
        Strictly changes the name label of a specific profile_id.
        If it was a temporary local-only profile, promotes it to a global profile
        upon explicit rename.
        """
        embedding = None
        is_new_global = False
        with self.lock:
            local_profiles = self.meeting_profiles.get(meeting_id, {})
            if profile_id not in local_profiles:
                print(f"SpeakerIDService: Cannot rename — '{profile_id}' not found in local meeting {meeting_id}.")
                return False

            local_profiles[profile_id]["name"] = new_name
            embedding = local_profiles[profile_id]["embedding"]

            # If not already in global profiles, promote it
            if profile_id not in self.global_profiles:
                is_new_global = True
            else:
                self.global_profiles[profile_id]["name"] = new_name

            print(f"SpeakerIDService: Renamed speaker ID {profile_id} → '{new_name}' in meeting {meeting_id}.")

        if is_new_global:
            # Promote to database — creates identity + primary template atomically
            new_id = create_global_speaker_profile(
                new_name, embedding, self.model_name, self.emb_dim
            )
            # Build in-memory multi-template structure
            primary_template = {
                "template_id": None,
                "embedding": embedding,
                "is_primary": True
            }
            with self.lock:
                self.global_profiles[new_id] = {
                    "name": new_name,
                    "templates": [primary_template],
                    "primary_index": 0
                }
                # Remap the local profile to the DB-generated id if it differs
                if new_id != profile_id:
                    lp = self.meeting_profiles.get(meeting_id, {})
                    if profile_id in lp:
                        lp[new_id] = lp.pop(profile_id)
        else:
            # Update name in database
            update_global_speaker_name(profile_id, new_name)

        return True

    def merge_speakers(self, source_profile_id: str, target_profile_id: str, meeting_id: str) -> bool:
        """
        Merges two speaker profiles by taking the union of their voice templates,
        capped at SPEAKER_ID_MAX_TEMPLATES (evicting outliers if the union exceeds
        the cap, protecting the target's primary). Deletes the source profile.
        Explicitly triggered by user intent.
        """
        with self.lock:
            local_profiles = self.meeting_profiles.get(meeting_id, {})
            if source_profile_id not in self.global_profiles or target_profile_id not in self.global_profiles:
                return False

            source_data = self.global_profiles[source_profile_id]
            target_data = self.global_profiles[target_profile_id]
            target_name = target_data["name"]

            # 1. Union templates: target's templates first (preserving primary), then source's (all non-primary)
            merged_templates = list(target_data["templates"])
            for t in source_data.get("templates", []):
                merged_templates.append({
                    "template_id": t["template_id"],
                    "embedding": t["embedding"],
                    "is_primary": False  # source templates lose primary status
                })

            # 2. Evict outliers if union exceeds cap
            primary_idx = target_data["primary_index"]
            while len(merged_templates) > SPEAKER_ID_MAX_TEMPLATES:
                evict_idx = self._find_outlier_template(merged_templates, primary_idx)
                merged_templates.pop(evict_idx)
                # Recalculate primary_index after removal
                for i, t in enumerate(merged_templates):
                    if t["is_primary"]:
                        primary_idx = i
                        break

            # 3. Update target profile in memory
            self.global_profiles[target_profile_id]["templates"] = merged_templates
            self.global_profiles[target_profile_id]["primary_index"] = primary_idx

            # Update local meeting profile with target's primary embedding for local matching
            primary_emb = merged_templates[primary_idx]["embedding"]
            if target_profile_id in local_profiles:
                local_profiles[target_profile_id]["embedding"] = primary_emb

            # 4. Remove source profile from memory maps
            self.global_profiles.pop(source_profile_id, None)
            local_profiles.pop(source_profile_id, None)
            print(f"SpeakerIDService: Merged {source_profile_id} into {target_profile_id} ({target_name}).")

        # 5. Delete source profile from database (CASCADE removes its templates)
        delete_global_speaker_profile(source_profile_id)
        # Note: the merged templates in memory will be persisted incrementally
        # via the normal template addition flow on subsequent matches.
        return True

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot_prod = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_prod / (norm1 * norm2))

    def identify_speaker(self, audio: np.ndarray, sample_rate: int, meeting_id: str, threshold: float = SPEAKER_ID_LOCAL_THRESHOLD) -> dict:
        """
        Identify speaker using a single lock acquisition to copy local and global snapshots.
        If the speaker remains unidentified after local/global checks, they are auto-enrolled
        locally under an anonymous "Speaker N" profile only if the audio duration is long enough
        (to prevent fragmentation on short, noisy utterances).
        """
        with self.lock:
            local_snapshot = dict(self.meeting_profiles.get(meeting_id, {}))
            global_snapshot = dict(self.global_profiles)

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            if test_emb is None:
                return {"speaker_id": "unknown", "speaker_name": "Unknown Speaker", "similarity": 0.0}

            # 1. Normalize embedding to match enroll_speaker's format
            norm = np.linalg.norm(test_emb)
            normalized_emb = test_emb / norm if norm > 0 else test_emb

            # Step 1: Match against local session profiles
            best_local_id, best_local_name, best_local_sim = self._best_match(normalized_emb, local_snapshot)

            if best_local_sim >= threshold:
                return {"speaker_id": best_local_id, "speaker_name": best_local_name, "similarity": best_local_sim}

            # Step 2: Match against global database profiles (multi-template)
            best_global_id, best_global_name, best_global_sim, _ = self._best_match_multi(normalized_emb, global_snapshot)

            if best_global_sim >= SPEAKER_ID_GLOBAL_THRESHOLD:
                # Use the primary template's embedding for local session enrollment
                gp = global_snapshot[best_global_id]
                best_global_emb = gp["templates"][gp["primary_index"]]["embedding"]
                # Auto-enroll in the current local meeting session
                should_add_template = False
                with self.lock:
                    if meeting_id not in self.meeting_profiles:
                        self.meeting_profiles[meeting_id] = {}
                    self.meeting_profiles[meeting_id][best_global_id] = {
                        "name": best_global_name,
                        "embedding": best_global_emb
                    }
                    # Template acceptance: only on genuinely confident passive matches
                    if best_global_sim >= SPEAKER_ID_TEMPLATE_ACCEPT_SIM:
                        live_gp = self.global_profiles.get(best_global_id)
                        if live_gp:
                            templates = live_gp["templates"]
                            if len(templates) < SPEAKER_ID_MAX_TEMPLATES:
                                # Room for a new template — add it
                                new_t = {"template_id": None, "embedding": normalized_emb, "is_primary": False}
                                templates.append(new_t)
                                should_add_template = True
                            else:
                                # Evict the outlier and replace
                                evict_idx = self._find_outlier_template(templates, live_gp["primary_index"])
                                self._evict_info = (templates[evict_idx].get("template_id"), best_global_id, best_global_sim)
                                templates[evict_idx] = {"template_id": None, "embedding": normalized_emb, "is_primary": False}
                                should_add_template = True  # will handle evict+add outside lock

                # DB writes outside lock
                if should_add_template:
                    try:
                        evict_info = getattr(self, '_evict_info', None)
                        if evict_info:
                            evict_tid, evict_sid, evict_sim = evict_info
                            if evict_tid:  # only evict if we have a real DB id
                                evict_speaker_template(evict_tid, evict_sid, evict_sim)
                            del self._evict_info
                        add_speaker_template(best_global_id, normalized_emb, best_global_sim)
                    except Exception:
                        logger.exception("[SpeakerID] Template add/evict failed for speaker %s", best_global_id)

                return {"speaker_id": best_global_id, "speaker_name": best_global_name, "similarity": best_global_sim}

            # Log details about the best sub-threshold match
            if best_local_sim > 0.0 or best_global_sim > 0.0:
                logger.info(
                    "[SpeakerID] Match failed threshold. "
                    "Best local: '%s' (sim=%.3f, threshold=%.2f). "
                    "Best global: '%s' (sim=%.3f, threshold=%.2f).",
                    best_local_name, best_local_sim, threshold,
                    best_global_name, best_global_sim, SPEAKER_ID_GLOBAL_THRESHOLD
                )

            # Step 3: Local-only auto-enrollment as a new numbered speaker ("Speaker N")
            # We enforce a minimum duration gate to prevent noise/short clips from auto-enrolling new profiles.
            duration_sec = len(audio) / sample_rate
            if duration_sec < SPEAKER_ID_MIN_ENROLL_DURATION:
                logger.info(
                    "[SpeakerID] Audio duration %.2fs is below minimum auto-enroll "
                    "limit (%.2fs). Skipping registration of a new speaker.",
                    duration_sec, SPEAKER_ID_MIN_ENROLL_DURATION
                )
                return {"speaker_id": "unknown", "speaker_name": "Unknown Speaker", "similarity": max(best_local_sim, best_global_sim)}

            # We must re-verify against the live profiles map under the lock to prevent
            # race conditions from concurrent pipeline tasks.
            with self.lock:
                if meeting_id not in self.meeting_profiles:
                    self.meeting_profiles[meeting_id] = {}
                meeting_profiles = self.meeting_profiles[meeting_id]

                # Check if a concurrent thread enrolled this voice during our async window
                best_live_id, best_live_name, best_live_sim = self._best_match(normalized_emb, meeting_profiles)

                if best_live_sim >= threshold:
                    logger.info(
                        "[SpeakerID] Concurrent task already enrolled this voice as "
                        "'%s' (ID=%s, sim=%.3f). Matching instead of duplicate auto-enrolling.",
                        best_live_name, best_live_id, best_live_sim
                    )
                    return {"speaker_id": best_live_id, "speaker_name": best_live_name, "similarity": best_live_sim}

                # Count existing Speaker N names
                existing_numbers = []
                for data in meeting_profiles.values():
                    name = data["name"]
                    if name.startswith("Speaker "):
                        try:
                            num = int(name.split(" ")[1])
                            existing_numbers.append(num)
                        except (ValueError, IndexError):
                            pass
                
                next_number = 1
                if existing_numbers:
                    next_number = max(existing_numbers) + 1
                    
                speaker_name = f"Speaker {next_number}"
                profile_id = str(uuid.uuid4())
                
                # Register in local map only (NOT global, NOT database)
                meeting_profiles[profile_id] = {
                    "name": speaker_name,
                    "embedding": normalized_emb
                }
                logger.info("[SpeakerID] Auto-enrolled unidentified speaker locally as '%s' (ID=%s) for meeting %s.", speaker_name, profile_id, meeting_id)
                
            return {"speaker_id": profile_id, "speaker_name": speaker_name, "similarity": 0.0}
        except Exception as e:
            logger.exception("[SpeakerID] Identification failed: %s", e)
            return {"speaker_id": "unknown", "speaker_name": "Unknown Speaker", "similarity": 0.0, "error": str(e)}

    def verify_speaker(self, profile_id: str, audio: np.ndarray, sample_rate: int, meeting_id: str, threshold: float = 0.5) -> dict:
        with self.lock:
            profiles = dict(self.meeting_profiles.get(meeting_id, {}))  # snapshot
        if profile_id not in profiles:
            return {"verified": False, "similarity": 0.0, "error": f"'{profile_id}' not enrolled in this meeting"}

        try:
            test_emb = self.get_embedding(audio, sample_rate)
            sim = self.compute_similarity(test_emb, profiles[profile_id]["embedding"])
            return {"verified": sim >= threshold, "similarity": sim, "threshold": threshold}
        except Exception as e:
            print(f"SpeakerIDService Error: Verification failed: {e}")
            return {"verified": False, "similarity": 0.0, "error": str(e)}

    # ──────────────────────────────────────────────────────────────────
    # Name extraction from STT text (drives auto-enrollment)
    # ──────────────────────────────────────────────────────────────────

    # Introduction cues that must co-occur with a PERSON entity for
    # the NER path to accept the name.  Keeps NER conservative — a
    # mention of "Modi" in a news discussion won't auto-enroll.
    _INTRO_CUES = {
        "i am", "i'm", "my name", "this is", "myself",
        "here", "speaking", "joined",
        # Tamil cue (romanised by Whisper sometimes)
        "en peyar",
    }

    @staticmethod
    def _extract_name_regex(text: str) -> str | None:
        """
        Primary extraction path — fast regex patterns.
        Returns the extracted name (title-cased) or None.
        """
        for pattern in _NAME_RE:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()

                # Check capitalization for English names to prevent false
                # positive matching of verbs/adverbs/adjectives
                # (e.g. "I am currently...", "I'm studying...")
                if re.match(r'^[A-Za-z]+$', name):
                    start_idx = match.start(1)
                    if start_idx >= 0 and start_idx < len(text):
                        first_char = text[start_idx]
                        if not first_char.isupper():
                            continue

                if len(name) >= 2 and name.lower() not in _NON_NAME_WORDS:
                    return name.title()
        return None

    @classmethod
    def _extract_name_ner(cls, text: str) -> str | None:
        """
        Fallback extraction path — SpaCy NER.

        Scans the transcription for PERSON entities on a sentence-by-sentence basis,
        accepting a candidate PERSON name only if its enclosing sentence also contains
        an introduction cue (e.g., "I am", "I'm", "my name", "speaking", "here").
        This prevents false-positive association across different clauses/sentences
        in a single turn (e.g. "I'm doing great. How are you doing Ram?").
        """
        nlp = _get_nlp()
        if nlp is None:
            return None

        doc = nlp(text)
        
        # Process each sentence independently to ensure co-occurrence of cue and name
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check if this sentence contains at least one introduction cue
            has_cue = any(cue in sent_lower for cue in cls._INTRO_CUES)
            if not has_cue:
                continue

            # Identify PERSON entities residing inside this specific sentence
            sent_start = sent.start_char
            sent_end = sent.end_char
            
            for ent in doc.ents:
                if ent.label_ == "PERSON" and ent.start_char >= sent_start and ent.end_char <= sent_end:
                    name = ent.text.strip()
                    if len(name) < 2 or name.lower() in _NON_NAME_WORDS:
                        continue

                    # Strong cues can appear anywhere in the sentence
                    strong_cues = {"i am", "i'm", "my name", "this is", "myself", "en peyar"}
                    if any(cue in sent_lower for cue in strong_cues):
                        return name.title()

                    # Weak cues must be adjacent/linked to the name in the sentence
                    name_lower = name.lower()
                    weak_patterns = [
                        f"{name_lower} here",
                        f"{name_lower} is here",
                        f"{name_lower} speaking",
                        f"{name_lower} is speaking",
                        f"{name_lower} joined",
                        f"{name_lower} has joined"
                    ]
                    if any(pat in sent_lower for pat in weak_patterns):
                        return name.title()

        return None

    @classmethod
    def extract_name_from_text(cls, text: str) -> str | None:
        """
        Two-pass name extraction from STT output:
          1. Fast regex patterns (< 0.01 ms)
          2. SpaCy NER fallback  (< 2 ms on CPU)

        Returns the extracted name (title-cased) or None if nothing matched.
        """
        if not text:
            return None

        # Pass 1 — regex (fast, high-precision)
        name = cls._extract_name_regex(text)
        if name:
            return name

        # Pass 2 — NER fallback (catches unstructured intros like
        # "Namashivayam here, let's start" or "This is Gobinath speaking")
        name = cls._extract_name_ner(text)
        if name:
            print(f"SpeakerIDService: NER extracted name '{name}' (regex missed).")
        return name
