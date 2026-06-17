from database.connection import get_connection
from config import DEFAULT_DEPARTMENT_ID

def create_meeting(title="Live Translation Session", department_id=DEFAULT_DEPARTMENT_ID):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO meetings (title, department_id)
            VALUES (%s, %s)
            RETURNING id;
        """, (title, department_id))
        meeting_id = cur.fetchone()["id"]
        conn.commit()
        return meeting_id
    finally:
        conn.close()

def save_utterance(
    meeting_id,
    source_text,
    translated_text,
    source_language,
    target_language,
    total_latency_ms
):
    conn = get_connection()

    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO utterances
            (
                meeting_id,
                utterance_time,

                source_language,
                target_language,

                source_text,
                translated_text,

                total_latency_ms
            )
            VALUES
            (
                %s,
                CURRENT_TIMESTAMP,

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

            total_latency_ms
        ))

        utterance_id = cur.fetchone()["id"]

        conn.commit()

        return utterance_id

    finally:
        conn.close()