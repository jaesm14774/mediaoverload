from __future__ import annotations

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentic.app.character_workflow import (
    build_goal_payload_from_character_config,
    resolve_character_selection,
)
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

    def test_group_name_cannot_be_used_as_prompt_identity_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "group-only.yaml"
            config.write_text(
                "character:\n  group_name: Kirby\ngeneration:\n"
                "  generation_type_weights:\n    text2img: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "group_name requires a resolved character selection"):
                build_goal_payload_from_character_config(
                    make_character_workflow_request(self.repo_root, config, publish_after_generate=False)
                )

    def test_pair_selection_allows_same_name_with_replacement(self) -> None:
        rows = [
            {
                "group_name": "Kirby",
                "role_name_en": "Kirby",
                "role_description": "round pink hero",
                "keywords": "pink, brave",
                "status": 1,
                "weight": 1.0,
            }
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
            selection = service.select_pair("Kirby", is_same_group=True, rng=random.Random(0))

        data = selection.to_dict()
        self.assertEqual([item["name"] for item in data["subjects"]], ["Kirby", "Kirby"])
        self.assertEqual(data["group_name"], "Kirby")
        self.assertTrue(data["is_same_group"])
        self.assertIn("pair_weighted_random_with_replacement", data["selection_source"])

    def test_pair_selection_without_same_group_uses_group_for_primary_and_global_for_secondary(self) -> None:
        rows = [
            {"group_name": "GroupA", "role_name_en": "Alpha", "status": 1, "weight": 1.0},
            {"group_name": "GroupB", "role_name_en": "Beta", "status": 1, "weight": 1.0},
        ]
        service = CharacterGroupSelectionService()
        group_candidates = service._parse_candidates([rows[0]])
        global_candidates = service._parse_candidates(rows)
        with patch.object(
            service,
            "get_candidates",
            return_value=group_candidates,
        ) as get_group_candidates, patch.object(
            service,
            "get_all_candidates",
            return_value=global_candidates,
        ) as get_global_candidates:
            selection = service.select_pair("GroupA", is_same_group=False, rng=random.Random(0))

        data = selection.to_dict()
        self.assertEqual(get_group_candidates.call_args.args, ("GroupA",))
        self.assertEqual(get_global_candidates.call_args.args, ())
        self.assertEqual(data["group_name"], "GroupA")
        self.assertEqual([item["name"] for item in data["subjects"]], ["Alpha", "Beta"])
        self.assertEqual(data["candidate_count"], 1)
        self.assertEqual(data["secondary_candidate_count"], 2)
        self.assertEqual(data["selection_source"], "pair_primary_group_plus_global_random")

    def test_resolve_pair_without_same_group_preserves_configured_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "mixed-pair.yaml"
            config.write_text(
                "character:\n"
                "  name: Alpha\n"
                "  group_name: GroupA\n"
                "generation:\n"
                "  subject_mode: two_character_interaction\n"
                "  two_character_interaction:\n"
                "    is_same_group: false\n",
                encoding="utf-8",
            )
            pair_result = Mock()
            pair_result.to_dict.return_value = {
                "mode": "two_character_interaction",
                "group_name": "GroupA",
                "is_same_group": False,
                "subjects": [
                    {"role": "primary", "name": "Alpha", "profile": {}},
                    {"role": "secondary", "name": "Beta", "profile": {}},
                ],
                "selection_source": "pair_primary_group_plus_global_random",
            }
            with patch.object(
                CharacterGroupSelectionService,
                "select_pair",
                return_value=pair_result,
            ) as select_pair:
                resolve_character_selection(
                    make_character_workflow_request(
                        self.repo_root,
                        config,
                        rng=random.Random(0),
                    )
                )

        pair_args, pair_kwargs = select_pair.call_args
        self.assertEqual(pair_args, ("GroupA",))
        self.assertFalse(pair_kwargs["is_same_group"])

    def test_random_subject_mode_uses_configured_weights_and_records_effective_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "random.yaml"
            config.write_text(
                "character:\n"
                "  name: Kirby\n"
                "  group_name: Kirby\n"
                "generation:\n"
                "  subject_mode: random\n"
                "  subject_mode_weights:\n"
                "    single: 3\n"
                "    two_character_interaction: 1\n"
                "  two_character_interaction:\n"
                "    is_same_group: true\n",
                encoding="utf-8",
            )
            single_result = Mock()
            single_result.to_dict.return_value = {
                "mode": "group",
                "group_name": "Kirby",
                "selected_character": "Kirby",
                "candidate_count": 1,
                "selected_profile": {},
                "selection_source": "group_weighted_random",
            }
            with patch.object(
                CharacterGroupSelectionService,
                "select_random_character",
                return_value=single_result,
            ) as select_single:
                resolved = resolve_character_selection(
                    make_character_workflow_request(
                        self.repo_root,
                        config,
                        rng=random.Random(1),
                    )
                )

        select_single.assert_called_once()
        self.assertEqual(resolved["subject_mode"], "single")
        self.assertEqual(resolved["configured_subject_mode"], "random")
        self.assertEqual(
            resolved["subject_mode_weights"],
            {"single": 3.0, "two_character_interaction": 1.0},
        )
        self.assertEqual(resolved["subject_mode_selection_source"], "weighted_random")

    def test_random_subject_mode_can_select_pair_without_distinct_name_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "random-pair.yaml"
            config.write_text(
                "character:\n"
                "  name: Kirby\n"
                "  group_name: Kirby\n"
                "generation:\n"
                "  subject_mode: random\n"
                "  subject_mode_weights:\n"
                "    single: 0\n"
                "    two_character_interaction: 1\n"
                "  two_character_interaction:\n"
                "    is_same_group: true\n",
                encoding="utf-8",
            )
            pair_result = Mock()
            pair_result.to_dict.return_value = {
                "mode": "two_character_interaction",
                "group_name": "Kirby",
                "is_same_group": True,
                "subjects": [
                    {"role": "primary", "name": "Kirby", "profile": {}},
                    {"role": "secondary", "name": "Kirby", "profile": {}},
                ],
                "selection_source": "pair_weighted_random_with_replacement",
            }
            with patch.object(
                CharacterGroupSelectionService,
                "select_pair",
                return_value=pair_result,
            ) as select_pair:
                resolved = resolve_character_selection(
                    make_character_workflow_request(
                        self.repo_root,
                        config,
                        rng=random.Random(0),
                    )
                )

        pair_args, pair_kwargs = select_pair.call_args
        self.assertEqual(pair_args, ("Kirby",))
        self.assertTrue(pair_kwargs["is_same_group"])
        self.assertIsNotNone(pair_kwargs["rng"])
        self.assertEqual(resolved["subject_mode"], "two_character_interaction")
        self.assertEqual([item["name"] for item in resolved["subjects"]], ["Kirby", "Kirby"])

    def test_random_subject_mode_rejects_invalid_weights_without_selection_fallback(self) -> None:
        invalid_weights = (
            {},
            {"single": -1, "two_character_interaction": 1},
            {"unknown": 1},
            {"single": "not-a-number"},
            {"single": 0, "two_character_interaction": 0},
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights), tempfile.TemporaryDirectory() as temp_dir:
                config = Path(temp_dir) / "invalid-random.yaml"
                config.write_text(
                    "character:\n  name: Kirby\n  group_name: Kirby\n"
                    "generation:\n  subject_mode: random\n"
                    f"  subject_mode_weights: {weights!r}\n",
                    encoding="utf-8",
                )
                with patch.object(CharacterGroupSelectionService, "select_random_character") as select_single:
                    with self.assertRaises(ValueError):
                        resolve_character_selection(
                            make_character_workflow_request(self.repo_root, config)
                        )
                select_single.assert_not_called()

    def test_random_subject_mode_payload_uses_resolved_pair_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "random-payload.yaml"
            config.write_text(
                "character:\n"
                "  name: Kirby\n"
                "  group_name: Kirby\n"
                "generation:\n"
                "  subject_mode: random\n"
                "  subject_mode_weights:\n"
                "    single: 1\n"
                "    two_character_interaction: 1\n"
                "  generation_type_weights:\n"
                "    text2img: 1\n",
                encoding="utf-8",
            )
            selection = {
                "mode": "two_character_interaction",
                "subject_mode": "two_character_interaction",
                "configured_subject_mode": "random",
                "group_name": "Kirby",
                "is_same_group": True,
                "subjects": [
                    {"role": "primary", "name": "Kirby", "profile": {}},
                    {"role": "secondary", "name": "MetaKnight", "profile": {}},
                ],
                "selection_source": "pair_weighted_random_with_replacement",
            }
            payload = build_goal_payload_from_character_config(
                make_character_workflow_request(
                    self.repo_root,
                    config,
                    prompt="The pair protects one glowing mechanism",
                    preferred_generation_type="text2img",
                    character_selection=selection,
                    publish_after_generate=False,
                )
            )

        self.assertEqual(payload["constraints"]["subject_mode"], "two_character_interaction")
        self.assertEqual(payload["constraints"]["configured_subject_mode"], "random")
        self.assertTrue(payload["constraints"]["subject_context"]["interaction_contract"]["required"])

    def test_random_subject_mode_payload_requires_resolved_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "random-unresolved.yaml"
            config.write_text(
                "character:\n  name: Kirby\ngeneration:\n"
                "  subject_mode: random\n"
                "  subject_mode_weights:\n"
                "    single: 1\n"
                "    two_character_interaction: 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires a resolved selection"):
                build_goal_payload_from_character_config(
                    make_character_workflow_request(self.repo_root, config)
                )

    def test_two_character_payload_preserves_two_subject_slots_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "pair.yaml"
            config.write_text(
                "character:\n"
                "  name: Kirby\n"
                "  group_name: Kirby\n"
                "generation:\n"
                "  subject_mode: two_character_interaction\n"
                "  two_character_interaction:\n"
                "    is_same_group: true\n"
                "  generation_type_weights:\n"
                "    text2img: 1\n",
                encoding="utf-8",
            )
            selection = {
                "mode": "two_character_interaction",
                "subject_mode": "two_character_interaction",
                "group_name": "Kirby",
                "is_same_group": True,
                "candidate_count": 1,
                "subjects": [
                    {
                        "role": "primary",
                        "name": "Kirby",
                        "profile": {"role_description": "round pink hero", "keywords": "pink"},
                    },
                    {
                        "role": "secondary",
                        "name": "Kirby",
                        "profile": {"role_description": "round pink hero", "keywords": "pink"},
                    },
                ],
                "selection_source": "pair_weighted_random_with_replacement",
            }
            payload = build_goal_payload_from_character_config(
                make_character_workflow_request(
                    self.repo_root,
                    config,
                    prompt="The pair protects one glowing mechanism",
                    preferred_generation_type="text2img",
                    character_selection=selection,
                    publish_after_generate=False,
                )
            )

        context = payload["constraints"]["subject_context"]
        self.assertEqual(payload["constraints"]["subject_mode"], "two_character_interaction")
        self.assertEqual([item["name"] for item in context["subjects"]], ["Kirby", "Kirby"])
        self.assertTrue(context["interaction_contract"]["required"])
        self.assertEqual(payload["character_config_summary"]["subject_context"], context)
        self.assertIn("two required subjects", payload["constraints"]["native_h3_creative_brief"].lower())

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

    def test_native_storyboard_pair_removes_single_subject_negative_constraints(self) -> None:
        storyboard = {
            "character": "Kirby",
            "base_prompt": "Kirby identity contract.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, watermark, text",
            "world": {"continuity_rules": ["Kirby remains the only protagonist."]},
        }
        resolved = _apply_selected_character_to_storyboard(
            storyboard,
            selected_character="Kirby",
            subject_context={
                "subjects": [
                    {"role": "primary", "name": "Kirby", "profile": {}},
                    {"role": "secondary", "name": "MetaKnight", "profile": {}},
                ],
                "interaction_contract": {"required": True, "same_frame": True},
            },
        )

        self.assertEqual(resolved["characters"], ["Kirby", "MetaKnight"])
        self.assertNotIn("duplicate Kirby", resolved["negative_prompt"])
        self.assertNotIn("extra characters", resolved["negative_prompt"])
        self.assertIn("unrequested third subject", resolved["negative_prompt"])


if __name__ == "__main__":
    unittest.main()
