"""
Unit tests for multi-template voice matching in SpeakerIDService.

Tests cover:
1. Fresh enrollment creates exactly one primary template.
2. Passive match >= 0.75 with < 5 templates adds a new secondary template.
3. Passive match >= 0.75 with exactly 5 templates evicts the correct outlier (never primary).
4. Passive match between 0.70 and 0.75 identifies but does NOT add/evict templates.
5. _best_match_multi returns best score across multiple templates, not just template[0].
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

# Mock SpeechBrain modules BEFORE importing SpeakerIDService
sys.modules['speechbrain'] = MagicMock()
sys.modules['speechbrain.inference'] = MagicMock()
sys.modules['speechbrain.inference.speaker'] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.speaker_id_service as service_module
service_module.load_global_speaker_profiles = MagicMock(return_value={})
def mock_create_profile(name, emb, model, dim, profile_id=None):
    return profile_id or "new-db-uuid"
service_module.create_global_speaker_profile = MagicMock(side_effect=mock_create_profile)
service_module.add_speaker_template = MagicMock(return_value="new-template-uuid")
service_module.evict_speaker_template = MagicMock()
service_module.update_global_speaker_name = MagicMock()
service_module.delete_global_speaker_profile = MagicMock()

from services.speaker_id_service import SpeakerIDService


def _unit_vec(*components):
    """Create a unit vector from components (auto-normalized)."""
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestMultiTemplateMatching(unittest.TestCase):
    def setUp(self):
        service_module.create_global_speaker_profile.reset_mock()
        service_module.add_speaker_template.reset_mock()
        service_module.evict_speaker_template.reset_mock()

        self.service = SpeakerIDService()
        self.service.classifier = MagicMock()
        self.service.global_profiles = {}
        self.service.meeting_profiles = {}

        # Embeddings: v1..v5 are clustered (high mutual similarity), v_outlier is distant
        self.v1 = _unit_vec(1.0, 0.0, 0.0)
        self.v2 = _unit_vec(0.98, 0.2, 0.0)
        self.v3 = _unit_vec(0.95, 0.31, 0.0)
        self.v4 = _unit_vec(0.97, 0.24, 0.0)
        self.v5 = _unit_vec(0.96, 0.28, 0.0)
        self.v_outlier = _unit_vec(0.0, 1.0, 0.0)  # orthogonal to v1..v5

    def test_1_fresh_enrollment_creates_primary_template(self):
        """Fresh enrollment via enroll_speaker (Tier C) creates exactly one primary template."""
        meeting_id = "test_meeting"
        self.service.get_embedding = MagicMock(return_value=self.v1)

        res = self.service.enroll_speaker("Alice", np.zeros(16000), 16000, meeting_id)

        self.assertIsNotNone(res.get("profile_id"))
        service_module.create_global_speaker_profile.assert_called_once()

        # Verify in-memory global profile has exactly one template marked primary
        pid = res["profile_id"]
        gp = self.service.global_profiles[pid]
        self.assertEqual(len(gp["templates"]), 1)
        self.assertTrue(gp["templates"][0]["is_primary"])
        self.assertEqual(gp["primary_index"], 0)

    def test_2_passive_match_adds_template_under_cap(self):
        """Passive match at >= 0.75 with fewer than 5 templates adds a new non-primary template."""
        meeting_id = "test_meeting"
        global_uuid = "global-alice-uuid"

        # Seed global profile with 1 primary template
        self.service.global_profiles[global_uuid] = {
            "name": "Alice",
            "templates": [
                {"template_id": "t1", "embedding": self.v1, "is_primary": True}
            ],
            "primary_index": 0
        }

        # New embedding very similar to v1 (sim > 0.75)
        new_emb = _unit_vec(0.99, 0.14, 0.0)
        self.service.get_embedding = MagicMock(return_value=new_emb)

        res = self.service.identify_speaker(np.zeros(16000), 16000, meeting_id)

        self.assertEqual(res["speaker_id"], global_uuid)
        # Verify template was added
        templates = self.service.global_profiles[global_uuid]["templates"]
        self.assertEqual(len(templates), 2)
        self.assertTrue(templates[0]["is_primary"])
        self.assertFalse(templates[1]["is_primary"])
        service_module.add_speaker_template.assert_called_once()

    def test_3_passive_match_evicts_outlier_at_cap(self):
        """Passive match at >= 0.75 with exactly 5 templates evicts the outlier, never primary."""
        meeting_id = "test_meeting"
        global_uuid = "global-alice-uuid"

        # Seed with 5 templates: primary + 3 similar + 1 outlier
        self.service.global_profiles[global_uuid] = {
            "name": "Alice",
            "templates": [
                {"template_id": "t1", "embedding": self.v1, "is_primary": True},
                {"template_id": "t2", "embedding": self.v2, "is_primary": False},
                {"template_id": "t3", "embedding": self.v3, "is_primary": False},
                {"template_id": "t4", "embedding": self.v4, "is_primary": False},
                {"template_id": "t_outlier", "embedding": self.v_outlier, "is_primary": False},
            ],
            "primary_index": 0
        }

        # New embedding very similar to cluster
        new_emb = _unit_vec(0.99, 0.14, 0.0)
        self.service.get_embedding = MagicMock(return_value=new_emb)

        res = self.service.identify_speaker(np.zeros(16000), 16000, meeting_id)

        self.assertEqual(res["speaker_id"], global_uuid)

        templates = self.service.global_profiles[global_uuid]["templates"]
        # Still 5 templates (evicted one, added one)
        self.assertEqual(len(templates), 5)
        # Primary must still be present
        self.assertTrue(templates[0]["is_primary"])
        # The outlier should have been evicted
        template_ids = [t["template_id"] for t in templates if t["template_id"] is not None]
        self.assertNotIn("t_outlier", template_ids, "Outlier template should have been evicted")
        # Evict was called with the outlier's template_id
        service_module.evict_speaker_template.assert_called_once()
        evict_call_args = service_module.evict_speaker_template.call_args
        self.assertEqual(evict_call_args[0][0], "t_outlier")

    def test_4_passive_match_between_thresholds_no_template_change(self):
        """Passive match between 0.70 and 0.75 identifies correctly but does NOT add/evict templates."""
        meeting_id = "test_meeting"
        global_uuid = "global-alice-uuid"

        # Seed with 1 template
        self.service.global_profiles[global_uuid] = {
            "name": "Alice",
            "templates": [
                {"template_id": "t1", "embedding": self.v1, "is_primary": True}
            ],
            "primary_index": 0
        }

        # Create embedding that gives similarity ~0.72 (between 0.70 and 0.75)
        # cos(theta) = 0.72 => sin(theta) ~= 0.694
        mid_emb = _unit_vec(0.72, 0.694, 0.0)
        self.service.get_embedding = MagicMock(return_value=mid_emb)

        res = self.service.identify_speaker(np.zeros(16000), 16000, meeting_id)

        self.assertEqual(res["speaker_id"], global_uuid)
        self.assertEqual(res["speaker_name"], "Alice")
        # Template count must NOT have changed
        templates = self.service.global_profiles[global_uuid]["templates"]
        self.assertEqual(len(templates), 1)
        service_module.add_speaker_template.assert_not_called()
        service_module.evict_speaker_template.assert_not_called()

    def test_5_best_match_multi_returns_best_across_templates(self):
        """_best_match_multi returns the best similarity across ALL templates, not just template[0]."""
        # Profile with 3 templates: template[2] is closest to the query
        query = _unit_vec(0.0, 0.0, 1.0)
        candidates = {
            "profile-A": {
                "name": "Alice",
                "templates": [
                    {"template_id": "t1", "embedding": _unit_vec(1.0, 0.0, 0.0), "is_primary": True},
                    {"template_id": "t2", "embedding": _unit_vec(0.0, 1.0, 0.0), "is_primary": False},
                    {"template_id": "t3", "embedding": _unit_vec(0.0, 0.0, 1.0), "is_primary": False},
                ],
                "primary_index": 0
            }
        }

        best_id, best_name, best_sim, best_t_idx = self.service._best_match_multi(query, candidates)

        self.assertEqual(best_id, "profile-A")
        self.assertEqual(best_name, "Alice")
        self.assertAlmostEqual(best_sim, 1.0, places=4)
        self.assertEqual(best_t_idx, 2, "Should match template[2], not template[0]")


    def test_6_concurrency_race_prevention(self):
        """Two back-to-back enroll_speaker calls for the same new voice return the same profile ID."""
        meeting_id = "test_meeting"
        self.service.get_embedding = MagicMock(return_value=self.v1)

        # First enrollment (Tier C: brand new speaker)
        res1 = self.service.enroll_speaker("Alice", np.zeros(16000), 16000, meeting_id)
        id1 = res1.get("profile_id")
        self.assertIsNotNone(id1)

        # Second enrollment for the same voice, before the first DB write has theoretically completed/synced,
        # but because it reserved the ID in memory, the second call matches it strictly
        res2 = self.service.enroll_speaker("Alice", np.zeros(16000), 16000, meeting_id)
        id2 = res2.get("profile_id")

        self.assertEqual(id1, id2, "Sequential enrollments of same voice must return the same profile ID")


    def test_7_db_failure_rollback(self):
        """If create_global_speaker_profile raises an exception, provisional in-memory reservation is rolled back."""
        meeting_id = "test_meeting"
        self.service.get_embedding = MagicMock(return_value=self.v1)

        # Mock DB failure
        with patch.object(service_module, 'create_global_speaker_profile', side_effect=RuntimeError("DB Connection pool exhausted")):
            res = self.service.enroll_speaker("Alice", np.zeros(16000), 16000, meeting_id)

        self.assertEqual(res, {})
        # Verify that self.global_profiles and self.meeting_profiles are empty (no orphan provisional entries left behind)
        self.assertEqual(len(self.service.global_profiles), 0)
        lp = self.service._get_meeting_profiles(meeting_id)
        self.assertEqual(len(lp), 0)


    def test_8_db_failure_case3_rollback(self):
        """If create_global_speaker_profile raises an exception during Case 3, local identity is restored to pre-rename name (not deleted)."""
        meeting_id = "test_meeting"
        temp_id = "temp-local-speaker-id"
        self.service.get_embedding = MagicMock(return_value=self.v1)

        # Pre-seed local anonymous profile
        self.service.meeting_profiles[meeting_id] = {
            temp_id: {"name": "Speaker 1", "embedding": self.v1}
        }

        # Mock DB failure
        with patch.object(service_module, 'create_global_speaker_profile', side_effect=RuntimeError("DB Write failure")):
            res = self.service.enroll_speaker("Alice", np.zeros(16000), 16000, meeting_id)

        self.assertEqual(res, {})
        # Verify that the local profile still exists under its original anonymous name
        lp = self.service._get_meeting_profiles(meeting_id)
        self.assertIn(temp_id, lp)
        self.assertEqual(lp[temp_id]["name"], "Speaker 1")
        # Verify that global profiles has no orphan entry for it
        self.assertNotIn(temp_id, self.service.global_profiles)


if __name__ == "__main__":
    unittest.main()
