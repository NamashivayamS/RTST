import sys
import os
import numpy as np
import uuid

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_pool, close_pool
from database.queries import get_connection, release_connection, delete_global_speaker_profile
from services.speaker_id_service import SpeakerIDService

def check_db_profile_name(profile_id: str) -> str | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT speaker_name FROM global_speaker_profiles WHERE id = %s;", (profile_id,))
        row = cur.fetchone()
        return row["speaker_name"] if row else None
    finally:
        release_connection(conn)

def run_test():
    print("Initialising database connection pool...")
    init_pool(minconn=2, maxconn=5)

    print("Initialising SpeakerIDService...")
    service = SpeakerIDService()

    # Generate unique random noise audio so it never matches any existing speaker
    sr = 16000
    print("Generating synthetic unique noise audio...")
    audio = np.random.randn(sr * 4).astype(np.float32)

    meeting_id = str(uuid.uuid4())
    print(f"Created dummy meeting ID: {meeting_id}")

    try:
        # Step 1: Identify voice locally (RAM-only)
        print("\n=== STEP 1: Identify voice locally (RAM-only) ===")
        res1 = service.identify_speaker(audio, sr, meeting_id)
        print("Identification 1 Result:", res1)
        profile_id = res1.get("speaker_id")
        name1 = res1.get("speaker_name")
        assert profile_id is not None, "Profile ID should not be None"
        assert name1 == "Speaker 1", f"Name should be Speaker 1, got {name1}"

        # Check DB - since it was enrolled locally, it should NOT be in the global database
        db_name = check_db_profile_name(profile_id)
        print(f"Database name check for ID {profile_id}: {db_name}")
        assert db_name is None, "Local-only Speaker 1 should NOT be persisted in global DB"

        # Step 2: Enroll same voice with specific name "Namashivayam" (Promotion case)
        print("\n=== STEP 2: Enroll same voice as 'Namashivayam' (Promotion) ===")
        res2 = service.enroll_speaker("Namashivayam", audio, sr, meeting_id)
        print("Enrollment 2 Result:", res2)
        assert res2.get("profile_id") == profile_id, f"Should reuse the existing profile ID {profile_id}, got {res2.get('profile_id')}"
        assert res2.get("name") == "Namashivayam", f"Name should be updated to Namashivayam, got {res2.get('name')}"

        # Check DB - it should now be persisted globally with the specific name
        db_name = check_db_profile_name(profile_id)
        print(f"Database name check for ID {profile_id} after promotion: {db_name}")
        assert db_name == "Namashivayam", f"Expected 'Namashivayam' in DB, got: {db_name}"

        print("\n=== Cross-method promotion handoff (identify -> enroll) tested successfully and verified against PostgreSQL! ===")
    
    finally:
        if 'profile_id' in locals() and profile_id:
            print(f"Cleaning up database speaker profile {profile_id}...")
            delete_global_speaker_profile(profile_id)
        close_pool()

if __name__ == "__main__":
    run_test()
