# database/queries.py
"""
All PostgreSQL queries for RealTimeSpeechTranslator.

Connection discipline
─────────────────────
Every function borrows a connection from the pool via get_connection(),
does its work, commits (or rolls back on error), then releases the
connection back to the pool in the finally block.

NEVER call conn.close() here — that would destroy a pool connection
rather than returning it. Always call release_connection(conn).

Thread safety
─────────────
These functions run on background threads via loop.run_in_executor().
psycopg2.ThreadedConnectionPool is thread-safe for borrow/return operations,
and individual connections are never shared across threads, so this is safe.
"""

import logging
import numpy as np
import psycopg2
from database.connection import get_connection, release_connection
from config import DEFAULT_DEPARTMENT_ID

logger = logging.getLogger("ispeak.db")


def create_meeting(
    title: str = "Live Translation Session",
    department_id: str = DEFAULT_DEPARTMENT_ID,
) -> str:
    """
    Inserts a new meeting row and returns its UUID.
    Called once per WebSocket connection, at connect time.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO meetings (title, department_id)
            VALUES (%s, %s)
            RETURNING id;
            """,
            (title, department_id),
        )
        meeting_id = cur.fetchone()["id"]
        conn.commit()
        logger.info(f"[DB] Meeting created: {meeting_id}")
        return meeting_id

    except Exception:
        # Roll back so the connection is clean before it goes back to the pool.
        # Without this, a failed transaction leaves the connection in an error
        # state that causes every subsequent query on that connection to fail
        # with "InFailedSqlTransaction".
        if conn:
            conn.rollback()
        logger.exception("[DB] create_meeting failed — rolled back.")
        raise

    finally:
        # Always release — even if an exception was raised above.
        release_connection(conn)


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
    """
    Inserts one utterance row and returns its UUID.
    Called from a background thread for every successfully translated sentence.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO utterances
            (
                meeting_id,
                utterance_time,
                source_language,
                target_language,
                source_text,
                translated_text,
                total_latency_ms,
                speaker_label,
                speaker_id
            )
            VALUES
            (
                %s,
                CURRENT_TIMESTAMP,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            RETURNING id;
            """,
            (
                meeting_id,
                source_language,
                target_language,
                source_text,
                translated_text,
                total_latency_ms,
                speaker_label,
                speaker_id,
            ),
        )
        utterance_id = cur.fetchone()["id"]
        conn.commit()
        logger.debug(f"[DB] Utterance saved: {utterance_id} (meeting={meeting_id})")
        return utterance_id

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] save_utterance failed — rolled back.")
        raise

    finally:
        release_connection(conn)


def rename_speaker_label(
    meeting_id: str,
    speaker_id: str,
    new_label: str,
) -> int:
    """
    Renames all utterances in a meeting for a specific speaker_id to `new_label`.
    Called from a background thread when the user corrects a speaker name.
    Returns the number of rows updated.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE utterances
            SET speaker_label = %s
            WHERE meeting_id = %s
              AND speaker_id = %s;
            """,
            (new_label, meeting_id, speaker_id),
        )
        count = cur.rowcount
        conn.commit()
        logger.info(
            "[DB] Renamed speaker_id '%s' → '%s' in meeting %s (%d rows).",
            speaker_id, new_label, meeting_id, count,
        )
        return count

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] rename_speaker_label failed — rolled back.")
        raise

    finally:
        release_connection(conn)


def merge_speaker_utterances(
    meeting_id: str,
    source_id: str,
    target_id: str,
    target_name: str,
) -> int:
    """
    Merges all utterances for source_id to target_id and target_name.
    Called from a background thread when merging speaker profiles.
    Returns the number of rows updated.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE utterances
            SET speaker_id = %s, speaker_label = %s
            WHERE meeting_id = %s
              AND speaker_id = %s;
            """,
            (target_id, target_name, meeting_id, source_id),
        )
        count = cur.rowcount
        conn.commit()
        logger.info(
            "[DB] Merged speaker_id '%s' → '%s' (%s) in meeting %s (%d rows).",
            source_id, target_id, target_name, meeting_id, count,
        )
        return count

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] merge_speaker_utterances failed — rolled back.")
        raise

    finally:
        release_connection(conn)


def load_global_speaker_profiles(model_version: str, embedding_dim: int) -> dict:
    """
    Loads all global speaker profiles and their voice templates compatible
    with the active model.

    Returns:
        dict: {profile_id_str: {
            "name": str,
            "templates": [{"template_id": str, "embedding": np.ndarray, "is_primary": bool}, ...],
            "primary_index": int   # index into "templates" where is_primary=True
        }}
    """
    conn = None
    profiles = {}
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.id AS profile_id, p.speaker_name,
                   t.id AS template_id, t.embedding, t.is_primary
            FROM global_speaker_profiles p
            JOIN speaker_voice_templates t ON t.speaker_id = p.id
            WHERE p.model_version = %s AND p.embedding_dim = %s
            ORDER BY p.id, t.is_primary DESC, t.created_at ASC;
            """,
            (model_version, embedding_dim)
        )
        rows = cur.fetchall()
        for row in rows:
            pid = str(row["profile_id"])
            if pid not in profiles:
                profiles[pid] = {
                    "name": row["speaker_name"],
                    "templates": [],
                    "primary_index": 0
                }
            emb_bytes = bytes(row["embedding"])
            profiles[pid]["templates"].append({
                "template_id": str(row["template_id"]),
                "embedding": np.frombuffer(emb_bytes, dtype=np.float32),
                "is_primary": row["is_primary"]
            })

        # Fix up primary_index for each profile
        for pid, data in profiles.items():
            for i, t in enumerate(data["templates"]):
                if t["is_primary"]:
                    data["primary_index"] = i
                    break

        logger.info(f"[DB] Loaded {len(profiles)} global speaker profiles (multi-template).")
        return profiles
    except Exception:
        logger.exception("[DB] Failed to load global speaker profiles.")
        return {}
    finally:
        if conn:
            release_connection(conn)


def create_global_speaker_profile(
    speaker_name: str,
    primary_embedding: np.ndarray,
    model_version: str,
    embedding_dim: int,
    profile_id: str = None,
) -> str:
    """
    Creates a new identity row in global_speaker_profiles AND its first
    (is_primary=True) row in speaker_voice_templates, in a single transaction.

    If profile_id is given, it is used explicitly as the new row's primary key —
    needed so enroll_speaker's Case 3 can preserve the local temp UUID's identity
    (the local meeting_profiles dict, any already-saved utterance rows, and the
    frontend's rendered speaker_id all stay valid without a remap/merge broadcast).
    If profile_id is None, Postgres generates a fresh UUID via the column default.

    Returns the new profile_id as a string either way.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 1. Insert identity row
        if profile_id is not None:
            cur.execute(
                """
                INSERT INTO global_speaker_profiles (id, speaker_name, model_version, embedding_dim)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (profile_id, speaker_name, model_version, embedding_dim),
            )
        else:
            cur.execute(
                """
                INSERT INTO global_speaker_profiles (speaker_name, model_version, embedding_dim)
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (speaker_name, model_version, embedding_dim),
            )
        result_id = str(cur.fetchone()["id"])

        # 2. Insert primary template
        emb_bytes = primary_embedding.tobytes()
        cur.execute(
            """
            INSERT INTO speaker_voice_templates (speaker_id, embedding, is_primary)
            VALUES (%s, %s, TRUE)
            RETURNING id;
            """,
            (result_id, psycopg2.Binary(emb_bytes)),
        )
        template_id = str(cur.fetchone()["id"])

        conn.commit()
        logger.info(
            "[DB] Created global speaker profile: %s (%s), primary template: %s",
            result_id, speaker_name, template_id,
        )
        return result_id

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to create global speaker profile.")
        raise

    finally:
        if conn:
            release_connection(conn)


def add_speaker_template(
    profile_id: str,
    embedding: np.ndarray,
    similarity: float,
) -> str:
    """
    Inserts a new non-primary template row for an existing speaker.
    Also inserts an 'added' row into speaker_template_events with the similarity score.
    Returns the new template row's id.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        emb_bytes = embedding.tobytes()
        cur.execute(
            """
            INSERT INTO speaker_voice_templates (speaker_id, embedding, is_primary)
            VALUES (%s, %s, FALSE)
            RETURNING id;
            """,
            (profile_id, psycopg2.Binary(emb_bytes)),
        )
        template_id = str(cur.fetchone()["id"])

        cur.execute(
            """
            INSERT INTO speaker_template_events (speaker_id, action, similarity)
            VALUES (%s, 'added', %s);
            """,
            (profile_id, similarity),
        )

        conn.commit()
        logger.info(
            "[DB] Added secondary template %s for speaker %s (sim=%.3f)",
            template_id, profile_id, similarity,
        )
        return template_id

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to add speaker template.")
        raise

    finally:
        if conn:
            release_connection(conn)


def evict_speaker_template(
    template_id: str,
    speaker_id: str,
    similarity: float,
) -> None:
    """
    Deletes one non-primary template row.
    Also inserts an 'evicted' row into speaker_template_events.
    Must never be called with a template_id where is_primary=True — asserts this.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Safety check: never evict the primary template
        cur.execute(
            "SELECT is_primary FROM speaker_voice_templates WHERE id = %s;",
            (template_id,),
        )
        row = cur.fetchone()
        assert row is not None, f"Template {template_id} not found in database"
        assert not row["is_primary"], (
            f"Attempted to evict primary template {template_id} for speaker {speaker_id} — this is a bug"
        )

        cur.execute(
            "DELETE FROM speaker_voice_templates WHERE id = %s;",
            (template_id,),
        )

        cur.execute(
            """
            INSERT INTO speaker_template_events (speaker_id, action, similarity)
            VALUES (%s, 'evicted', %s);
            """,
            (speaker_id, similarity),
        )

        conn.commit()
        logger.info(
            "[DB] Evicted template %s from speaker %s (sim=%.3f)",
            template_id, speaker_id, similarity,
        )

    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to evict speaker template.")
        raise

    finally:
        if conn:
            release_connection(conn)


def update_global_speaker_name(profile_id: str, new_name: str) -> None:
    """
    Updates only the name label of a global speaker profile.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE global_speaker_profiles
            SET speaker_name = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s;
            """,
            (new_name, profile_id),
        )
        conn.commit()
        logger.info(f"[DB] Updated name for global speaker profile: {profile_id} → {new_name}")
    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to update global speaker name.")
    finally:
        if conn:
            release_connection(conn)


def delete_global_speaker_profile(profile_id: str) -> None:
    """
    Deletes a global speaker profile.
    CASCADE will remove all associated templates automatically.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM global_speaker_profiles WHERE id = %s;", (profile_id,))
        conn.commit()
        logger.info(f"[DB] Deleted global speaker profile: {profile_id}")
    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to delete global speaker profile.")
    finally:
        if conn:
            release_connection(conn)


def get_meeting_details(meeting_id: str) -> dict | None:
    """
    Fetches the title and creation time of a meeting.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT title, created_at
            FROM meetings
            WHERE id = %s;
            """,
            (meeting_id,),
        )
        row = cur.fetchone()
        if row:
            return {
                "title": row["title"],
                "created_at": row["created_at"]
            }
        return None
    except Exception:
        logger.exception("[DB] get_meeting_details failed.")
        return None
    finally:
        if conn:
            release_connection(conn)


def get_meeting_transcript(meeting_id: str) -> list[dict]:
    """
    Fetches all utterances for a meeting, ordered chronologically.
    Used by the summarization endpoint to build the full transcript.

    Returns:
        list of dicts with keys: speaker_label, source_text, translated_text,
        source_language, target_language, utterance_time
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT speaker_label, source_text, translated_text,
                   source_language, target_language, utterance_time
            FROM utterances
            WHERE meeting_id = %s
            ORDER BY utterance_time ASC;
            """,
            (meeting_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "speaker_label": row["speaker_label"],
                "source_text": row["source_text"],
                "translated_text": row["translated_text"],
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "utterance_time": str(row["utterance_time"]),
            }
            for row in rows
        ]

    except Exception:
        logger.exception("[DB] get_meeting_transcript failed.")
        return []

    finally:
        if conn:
            release_connection(conn)