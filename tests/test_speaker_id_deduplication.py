import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock

# 1. Mock SpeechBrain modules BEFORE importing SpeakerIDService to prevent actual ML model loading/downloads
sys.modules['speechbrain'] = MagicMock()
sys.modules['speechbrain.inference'] = MagicMock()
sys.modules['speechbrain.inference.speaker'] = MagicMock()

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.speaker_id_service as service_module
# Mock the DB queries at the module level before instantiation to avoid database dependency
service_module.load_global_speaker_profiles = MagicMock(return_value={})
service_module.create_global_speaker_profile = MagicMock(return_value="new-db-uuid")
service_module.add_speaker_template = MagicMock(return_value="new-template-uuid")
service_module.evict_speaker_template = MagicMock()
service_module.update_global_speaker_name = MagicMock()
service_module.delete_global_speaker_profile = MagicMock()

from services.speaker_id_service import SpeakerIDService


def _make_global_profile(name, embedding):
    """Helper: build a multi-template global profile dict from a single embedding."""
    return {
        "name": name,
        "templates": [{"template_id": "mock-tid", "embedding": embedding, "is_primary": True}],
        "primary_index": 0
    }


class TestSpeakerIDDeduplication(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        service_module.create_global_speaker_profile.reset_mock()
        service_module.update_global_speaker_name.reset_mock()

        # Instantiate service (we mock the classifier setup so it doesn't load SpeechBrain models)
        self.service = SpeakerIDService()
        self.service.classifier = MagicMock()
        self.service.global_profiles = {}
        self.service.meeting_profiles = {}

        # Set up mock embeddings (unit vectors so cosine similarity is simple dot product)
        self.v_global = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.v_lenient = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        self.v_strict = np.array([0.85, np.sqrt(1.0 - 0.85**2), 0.0], dtype=np.float32)
        self.v_different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def test_case_1_lenient_match(self):
        # Case 1: Name + voice agree on pre-existing global identity (lenient 0.52+ match)
        meeting_id = "test_meeting"
        global_uuid = "global-namashivayam-uuid"

        # Seed global profiles with Namashivayam (multi-template structure)
        self.service.global_profiles[global_uuid] = _make_global_profile("Namashivayam", self.v_global)

        # Seed local session profiles with a generic anonymous "Speaker 1"
        temp_uuid = "temp-speaker-1"
        profiles = self.service._get_meeting_profiles(meeting_id)
        profiles[temp_uuid] = {
            "name": "Speaker 1",
            "embedding": self.v_lenient # This local speaker's voice is leniently close to Namashivayam
        }

        # Mock get_embedding to return the lenient voice profile for enroll_speaker
        self.service.get_embedding = MagicMock(return_value=self.v_lenient)

        # Enroll "Namashivayam" (the user introduces themselves)
        # Using the same lenient voice, the spoken name matches the global profile name
        res = self.service.enroll_speaker("Namashivayam", np.zeros(16000), 16000, meeting_id)

        # Verify results
        self.assertEqual(res.get("profile_id"), global_uuid)
        self.assertEqual(res.get("name"), "Namashivayam")
        self.assertTrue(res.get("was_merged"))
        self.assertEqual(res.get("merged_from_id"), temp_uuid)

        # Verify profiles state in memory
        self.assertNotIn(temp_uuid, profiles)
        self.assertIn(global_uuid, profiles)
        self.assertEqual(profiles[global_uuid]["name"], "Namashivayam")

    def test_case_2_strict_fallback_promotion(self):
        # Case 2: Spoken name doesn't match (nickname/typo "Shiva"), but voice matches global strictly (0.70+)
        meeting_id = "test_meeting"
        global_uuid = "global-namashivayam-uuid"

        # Seed global profiles (multi-template)
        self.service.global_profiles[global_uuid] = _make_global_profile("Namashivayam", self.v_global)

        # Seed local profiles with generic anonymous "Speaker 1"
        temp_uuid = "temp-speaker-1"
        profiles = self.service._get_meeting_profiles(meeting_id)
        profiles[temp_uuid] = {
            "name": "Speaker 1",
            "embedding": self.v_strict # Local speaker's voice is strictly matched to Namashivayam (0.85)
        }

        self.service.get_embedding = MagicMock(return_value=self.v_strict)

        # Enroll with nickname "Shiva". The name-filtered global check fails,
        # but the strict global voice match passes and merges temp-speaker-1 into global_uuid.
        res = self.service.enroll_speaker("Shiva", np.zeros(16000), 16000, meeting_id)

        # Verify results
        self.assertEqual(res.get("profile_id"), global_uuid)
        self.assertEqual(res.get("name"), "Namashivayam") # Trust database name over spoken nickname
        self.assertTrue(res.get("was_merged"))
        self.assertEqual(res.get("merged_from_id"), temp_uuid)

        # Verify profiles state
        self.assertNotIn(temp_uuid, profiles)
        self.assertIn(global_uuid, profiles)

    def test_case_3_new_identity_promotion(self):
        # Case 3: Genuinely new speaker says name "John". No global match.
        meeting_id = "test_meeting"

        # Seed local profiles with generic "Speaker 1"
        temp_uuid = "temp-speaker-1"
        profiles = self.service._get_meeting_profiles(meeting_id)
        profiles[temp_uuid] = {
            "name": "Speaker 1",
            "embedding": self.v_different
        }

        self.service.get_embedding = MagicMock(return_value=self.v_different)

        res = self.service.enroll_speaker("John", np.zeros(16000), 16000, meeting_id)

        # Verify results — profile_id is now DB-generated ("new-db-uuid" from mock)
        self.assertEqual(res.get("profile_id"), "new-db-uuid")
        self.assertEqual(res.get("name"), "John")
        self.assertFalse(res.get("was_merged"))

        # Verify DB create was called
        service_module.create_global_speaker_profile.assert_called_once()
        # Verify in-memory global profile has multi-template structure
        self.assertIn("new-db-uuid", self.service.global_profiles)
        gp = self.service.global_profiles["new-db-uuid"]
        self.assertEqual(len(gp["templates"]), 1)
        self.assertTrue(gp["templates"][0]["is_primary"])

    def test_tier_b_strict_fallback_fresh(self):
        # Tier B: No local profile exists yet, name doesn't match, but voice matches global strictly (0.70+)
        meeting_id = "test_meeting"
        global_uuid = "global-namashivayam-uuid"

        # Seed global profiles (multi-template)
        self.service.global_profiles[global_uuid] = _make_global_profile("Namashivayam", self.v_global)

        # Clear meeting profiles (no local profile exists yet)
        profiles = self.service._get_meeting_profiles(meeting_id)
        profiles.clear()

        self.service.get_embedding = MagicMock(return_value=self.v_strict)

        # Enroll with nickname "Shiva".
        res = self.service.enroll_speaker("Shiva", np.zeros(16000), 16000, meeting_id)

        # Verify results
        self.assertEqual(res.get("profile_id"), global_uuid)
        self.assertEqual(res.get("name"), "Namashivayam")
        self.assertFalse(res.get("was_merged"))

        # Verify local session profiles list has auto-enrolled the global profile
        self.assertIn(global_uuid, profiles)
        self.assertEqual(profiles[global_uuid]["name"], "Namashivayam")

    def test_already_enrolled_specific_name_mismatch(self):
        # Already enrolled under a specific name mismatch:
        # If the local profile already has a specific name like "Namashivayam",
        # and another self-introduction with matching voice tries to change it to a different specific name "Gobi",
        # the system should warn and preserve the existing locked name "Namashivayam".
        meeting_id = "test_meeting"
        profile_uuid = "global-namashivayam-uuid"

        # Seed local session profiles with specific name "Namashivayam"
        profiles = self.service._get_meeting_profiles(meeting_id)
        profiles[profile_uuid] = {
            "name": "Namashivayam",
            "embedding": self.v_global
        }

        # Mock voice to match this profile
        self.service.get_embedding = MagicMock(return_value=self.v_global)

        # Attempt to enroll with name "Gobi"
        res = self.service.enroll_speaker("Gobi", np.zeros(16000), 16000, meeting_id)

        # Verify results
        self.assertEqual(res.get("profile_id"), profile_uuid)
        self.assertEqual(res.get("name"), "Namashivayam") # Name was not overwritten
        self.assertFalse(res.get("was_merged"))
        self.assertIsNone(res.get("merged_from_id"))

        # Verify memory remains correct
        self.assertEqual(profiles[profile_uuid]["name"], "Namashivayam")

if __name__ == "__main__":
    unittest.main()
