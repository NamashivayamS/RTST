import sys
import os
import psycopg2

# Add project root to sys.path so we can import config
sys.path.append(r"d:\NEED\Sem\Sem 7\Ramraj Intern\RealTimeSpeechTranslator")
from config import POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

def run_migration():
    print(f"Connecting to database {POSTGRES_DB} on {POSTGRES_HOST}...")
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    try:
        print("Altering utterances table to add speaker_id column...")
        cur.execute("""
            ALTER TABLE utterances
            ADD COLUMN IF NOT EXISTS speaker_id VARCHAR(36) NOT NULL DEFAULT 'unknown';
        """)
        print("Migration applied successfully!")
    except Exception as e:
        print("Migration failed:", e)
        raise e
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_migration()
