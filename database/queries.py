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
    Loads all global speaker profiles compatible with the active model.
    Returns:
        dict: {profile_id_str: {"name": str, "embedding": np.ndarray}}
    """
    conn = None
    profiles = {}
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, speaker_name, embedding 
            FROM global_speaker_profiles
            WHERE model_version = %s AND embedding_dim = %s;
            """,
            (model_version, embedding_dim)
        )
        rows = cur.fetchall()
        for row in rows:
            profile_id = str(row["id"])
            name = row["speaker_name"]
            emb_bytes = bytes(row["embedding"])  # cast memoryview to bytes
            profiles[profile_id] = {
                "name": name,
                "embedding": np.frombuffer(emb_bytes, dtype=np.float32)
            }
        logger.info(f"[DB] Loaded {len(profiles)} global speaker profiles.")
        return profiles
    except Exception:
        logger.exception("[DB] Failed to load global speaker profiles.")
        return {}
    finally:
        if conn:
            release_connection(conn)


def save_global_speaker_profile(profile_id: str, speaker_name: str, embedding: np.ndarray, model_version: str, embedding_dim: int) -> None:
    """
    Upserts a speaker profile in the database keyed on profile ID.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        emb_bytes = embedding.tobytes()
        cur.execute(
            """
            INSERT INTO global_speaker_profiles (id, speaker_name, embedding, model_version, embedding_dim, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id)
            DO UPDATE SET speaker_name = EXCLUDED.speaker_name, embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;
            """,
            (profile_id, speaker_name, psycopg2.Binary(emb_bytes), model_version, embedding_dim),
        )
        conn.commit()
        logger.info(f"[DB] Saved global speaker profile: {profile_id} ({speaker_name})")
    except Exception:
        if conn:
            conn.rollback()
        logger.exception("[DB] Failed to save global speaker profile.")
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