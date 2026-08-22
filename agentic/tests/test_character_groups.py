from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.app.character_workflow import build_goal_payload_from_character_config
from agentic.skills.longvideo import _apply_selected_character_to_storyboard
from agentic.tools.context_services import CharacterGroupSelectionService
from character_workflow_helpers import make_character_workflow_request


class _FakeCursor:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.query = ""
        self.params: list[object] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: list[object]) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.cursor_instance = _FakeCursor(rows)

    def cursor(self) -> _FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        return None


class CharacterGroupSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.kirby_config = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_query_uses_current_schema_and_filters_active_positive_weights(self) -> None:
        rows = [
            {"role_name_en": "MetaKnight", "role_description": "masked knight", "keywords": "sword", "status": 1, "weight": 0.8},
            {"role_name_en": "Disabled", "role_description": "", "keywords": "", "status": 0, "weight": 4},
            {"role_name_en": "ZeroWeight", "role_description": "", "keywords": "", "status": 1, "weight": 0},
        ]
        connection = _FakeConnection(rows)
        service = CharacterGroupSelectionService()
        with patch.dict(
            os.environ,
            {
                "mysql_host": "localhost",
                "mysql_port": "3306",
                "mysql_user": "user",
                "mysql_password": "password",
                "mysql_db_name": "anime",
            },
            clear=False,
        ), patch("pymysql.connect", return_value=connection):
            selection = service.select_random_character("Kirby", rng=random.Random(0))

        self.assertEqual(selection.selected_character, "MetaKnight")
        self.assertEqual(selection.to_dict()["candidate_count"], 1)
        self.assertNotIn("workflow_name", connection.cursor_instance.query)
        self.assertEqual(connection.cursor_instance.params, ["Kirby"])

    def test_selected_group_character_replaces_config_identity_in_payload(self) -> None:
        selection = {
            "mode": "group",
            "group_name": "Kirby",
            "selected_character": "MetaKnight",
            "candidate_count": 2,
            "candidates": [
                {"name": "Kirby", "status": 1, "weight": 1.0},
                {"name": "MetaKnight", "status": 1, "weight": 0.8},
            ],
            "selected_profile": {
                "role_description": "a masked knight who protects the group",
                "keywords": "mask, sword, cape",
            },
            "selection_source": "group_weighted_random",
        }
        payload = build_goal_payload_from_character_config(
            make_character_workflow_request(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby protects one glowing mechanism",
                preferred_generation_type="native_h3_story",
                selected_character_name="MetaKnight",
                character_selection=selection,
                publish_after_generate=False,
            )
        )

        self.assertEqual(payload["character_name"], "MetaKnight")
        self.assertEqual(payload["character_selection"]["selected_character"], "MetaKnight")
        self.assertEqual(payload["constraints"]["character"], "MetaKnight")
        self.assertIn("MetaKnight", payload["constraints"]["native_h3_creative_brief"])
        self.assertNotIn("Kirby", payload["constraints"]["native_h3_creative_brief"])
        self.assertEqual(payload["character_config_summary"]["character_name"], "MetaKnight")

    def test_fixed_config_does_not_query_database_without_group_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "fixed.yaml"
            config.write_text(
                "character:\n  name: FixedHero\ngeneration:\n  generation_type_weights:\n    text2img: 1\n",
                encoding="utf-8",
            )
            with patch.object(CharacterGroupSelectionService, "select_random_character") as selector:
                payload = build_goal_payload_from_character_config(
                    make_character_workflow_request(
                        self.repo_root,
                        config,
                        preferred_generation_type="text2img",
                        publish_after_generate=False,
                    )
                )
            selector.assert_not_called()
            self.assertEqual(payload["character_name"], "FixedHero")
            self.assertEqual(payload["character_selection"]["selection_source"], "fixed_config")

    def test_native_storyboard_identity_uses_selected_role_profile(self) -> None:
        storyboard = {
            "character": "Kirby",
            "base_prompt": "Kirby identity contract with a round pink body.",
            "world": {"continuity_rules": ["Only Kirby appears."]},
            "native_shots": [{"action": "Kirby presses the mechanism."}],
        }
        resolved = _apply_selected_character_to_storyboard(
            storyboard,
            selected_character="MetaKnight",
            character_profile={
                "role_description": "a masked knight with a blue cape",
                "keywords": "mask, cape, sword",
            },
        )

        self.assertEqual(resolved["character"], "MetaKnight")
        self.assertIn("MetaKnight", resolved["world"]["continuity_rules"][0])
        self.assertIn("MetaKnight", resolved["base_prompt"])
        self.assertIn("masked knight", resolved["base_prompt"])
        self.assertIn("MetaKnight", resolved["native_shots"][0]["action"])


if __name__ == "__main__":
    unittest.main()
