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
                speaker_label
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