# database/mock_queries.py
"""
In-memory mock implementations of all database query functions.

Used when MOCK_DATABASE=1 (e.g. on Kaggle where no SQL Server is available).
Every function mirrors the signature and return type of its counterpart in
queries.py, but stores data in simple Python dicts/lists instead of SQL Server.

Data is ephemeral — it lives only for the duration of the running process.
"""

import logging
import uuid
from datetime import datetime

import numpy as np

logger = logging.getLogger("ispeak.db.mock")

# ── In-memory storage ─────────────────────────────────────────────────────────

_meetings = {}          # {meeting_id: {title, department_id, created_at}}
_utterances = []        # [{meeting_id, source_text, translated_text, ...}]
_speaker_profiles = {}  # {profile_id: {speaker_name, model_version, embedding_dim, templates: [...]}}
_template_events = []   # [{speaker_id, action, similarity, created_at}]


# ── Meeting operations ────────────────────────────────────────────────────────

def create_meeting(
    title: str = "Live Translation Session",
    department_id: str = "default",
) -> str:
    meeting_id = str(uuid.uuid4()).lower()
    _meetings[meeting_id] = {
        "title": title,
        "department_id": department_id,
        "created_at": datetime.utcnow(),
    }
    logger.info(f"[Mock DB] Meeting created: {meeting_id}")
    return meeting_id


def get_meeting_details(meeting_id: str) -> dict | None:
    meeting = _meetings.get(meeting_id)
    if meeting is None:
        return None
    return {
        "title": meeting["title"],
        "created_at": meeting["created_at"],
    }


# ── Utterance operations ─────────────────────────────────────────────────────

def save_utterance(
    meeting_id: str,
    source_text: str,
    translated_text: str,
    source_language: str,
    target_language: str,
    total_latency_ms: int,
    speaker_label: str = "unknown",
    speaker_id: str = "unknown",
) -> str:
    utterance_id = str(uuid.uuid4()).lower()
    _utterances.append({
        "id": utterance_id,
        "meeting_id": meeting_id,
        "source_text": source_text,
        "translated_text": translated_text,
        "source_language": source_language,
        "target_language": target_language,
        "total_latency_ms": total_latency_ms,
        "speaker_label": speaker_label,
        "speaker_id": speaker_id,
        "utterance_time": datetime.utcnow(),
    })
    logger.debug(f"[Mock DB] Utterance saved: {utterance_id} (meeting={meeting_id})")
    return utterance_id


def get_meeting_transcript(meeting_id: str) -> list[dict]:
    return [
        {
            "speaker_label": u["speaker_label"],
            "source_text": u["source_text"],
            "translated_text": u["translated_text"],
            "source_language": u["source_language"],
            "target_language": u["target_language"],
            "utterance_time": str(u["utterance_time"]),
        }
        for u in _utterances
        if u["meeting_id"] == meeting_id
    ]


def rename_speaker_label(
    meeting_id: str,
    speaker_id: str,
    new_label: str,
) -> int:
    count = 0
    for u in _utterances:
        if u["meeting_id"] == meeting_id and u["speaker_id"] == speaker_id:
            u["speaker_label"] = new_label
            count += 1
    logger.info(
        "[Mock DB] Renamed speaker_id '%s' → '%s' in meeting %s (%d rows).",
        speaker_id, new_label, meeting_id, count,
    )
    return count


def merge_speaker_utterances(
    meeting_id: str,
    source_id: str,
    target_id: str,
    target_name: str,
) -> int:
    count = 0
    for u in _utterances:
        if u["meeting_id"] == meeting_id and u["speaker_id"] == source_id:
            u["speaker_id"] = target_id
            u["speaker_label"] = target_name
            count += 1
    logger.info(
        "[Mock DB] Merged speaker_id '%s' → '%s' (%s) in meeting %s (%d rows).",
        source_id, target_id, target_name, meeting_id, count,
    )
    return count


# ── Speaker profile operations ───────────────────────────────────────────────

def load_global_speaker_profiles(model_version: str, embedding_dim: int) -> dict:
    result = {}
    for pid, data in _speaker_profiles.items():
        if data["model_version"] == model_version and data["embedding_dim"] == embedding_dim:
            templates = []
            primary_index = 0
            for i, t in enumerate(data["templates"]):
                templates.append({
                    "template_id": t["template_id"],
                    "embedding": t["embedding"],
                    "is_primary": t["is_primary"],
                })
                if t["is_primary"]:
                    primary_index = i
            result[pid] = {
                "name": data["speaker_name"],
                "templates": templates,
                "primary_index": primary_index,
            }
    logger.info(f"[Mock DB] Loaded {len(result)} global speaker profiles.")
    return result


def create_global_speaker_profile(
    speaker_name: str,
    primary_embedding: np.ndarray,
    model_version: str,
    embedding_dim: int,
    profile_id: str = None,
) -> str:
    if profile_id is None:
        profile_id = str(uuid.uuid4()).lower()
    else:
        profile_id = profile_id.lower()

    template_id = str(uuid.uuid4()).lower()

    _speaker_profiles[profile_id] = {
        "speaker_name": speaker_name,
        "model_version": model_version,
        "embedding_dim": embedding_dim,
        "templates": [
            {
                "template_id": template_id,
                "embedding": primary_embedding.copy(),
                "is_primary": True,
            }
        ],
    }
    logger.info(
        "[Mock DB] Created global speaker profile: %s (%s), primary template: %s",
        profile_id, speaker_name, template_id,
    )
    return profile_id


def add_speaker_template(
    profile_id: str,
    embedding: np.ndarray,
    similarity: float,
) -> str:
    template_id = str(uuid.uuid4()).lower()

    if profile_id in _speaker_profiles:
        _speaker_profiles[profile_id]["templates"].append({
            "template_id": template_id,
            "embedding": embedding.copy(),
            "is_primary": False,
        })

    _template_events.append({
        "speaker_id": profile_id,
        "action": "added",
        "similarity": similarity,
        "created_at": datetime.utcnow(),
    })

    logger.info(
        "[Mock DB] Added secondary template %s for speaker %s (sim=%.3f)",
        template_id, profile_id, similarity,
    )
    return template_id


def evict_speaker_template(
    template_id: str,
    speaker_id: str,
    similarity: float,
) -> None:
    if speaker_id in _speaker_profiles:
        templates = _speaker_profiles[speaker_id]["templates"]
        _speaker_profiles[speaker_id]["templates"] = [
            t for t in templates
            if t["template_id"] != template_id or t["is_primary"]
        ]

    _template_events.append({
        "speaker_id": speaker_id,
        "action": "evicted",
        "similarity": similarity,
        "created_at": datetime.utcnow(),
    })

    logger.info(
        "[Mock DB] Evicted template %s from speaker %s (sim=%.3f)",
        template_id, speaker_id, similarity,
    )


def update_global_speaker_name(profile_id: str, new_name: str) -> None:
    if profile_id in _speaker_profiles:
        _speaker_profiles[profile_id]["speaker_name"] = new_name
    logger.info(f"[Mock DB] Updated name for global speaker profile: {profile_id} → {new_name}")


def delete_global_speaker_profile(profile_id: str) -> None:
    _speaker_profiles.pop(profile_id, None)
    logger.info(f"[Mock DB] Deleted global speaker profile: {profile_id}")
