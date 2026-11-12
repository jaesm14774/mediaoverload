from __future__ import annotations

import argparse
import subprocess
import unittest
import os
import sys
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pymysql
from dotenv import dotenv_values
from pymysql.cursors import DictCursor

from agentic.app.cli import parse_args
from agentic.app.main import (
    _allow_runtime_output_for_visual_evidence,
    _prepare_serial_execution_plan,
    _record_serial_character_review,
    _record_serial_episode_outline_review,
    _record_serial_story_review,
    _resolve_serial_character_reference_paths,
    _serial_first_episode_argv,
    _serial_completion_result,
    main as app_main,
)
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import (
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
    RunState,
    SkillContext,
    SkillResult,
    WorkflowRunResult,
)
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.serial_video import SerialVideoStore
from agentic.skills.agent_primitives import AgentMediaSkills, AgentPlanningSkills
from agentic.skills.agent_social import register_agent_social_skills
from agentic.tools.media_services import register_media_service_tools
from agentic.tools.context_services import HumanReviewDecision


def _workspace_temporary_directory() -> TemporaryDirectory:
    temp_root = Path(__file__).resolve().parents[2] / ".tmp-tests"
    temp_root.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(dir=temp_root)


def _write_video(path: Path, color: str, duration_seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x180:r=24:d={duration_seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_seconds}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _write_reference(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _seed_accepted_existing_episode(
    store: SerialVideoStore,
    root: Path,
    series_id: str,
    *,
    approve_for_continuation: bool = False,
) -> dict[str, object]:
    identity = _write_reference(root / f"{series_id}-identity.png", b"identity")
    context = store.prepare_series_context(series_id, "Hero", [identity])
    episode = store.start_episode(context, "An existing story", topic="A new turn")
    store.record_plan(
        episode["episode_id"],
        segment_count=1,
        plan_payload={"goal": {"duration_seconds": 10}},
    )
    render_id = store.begin_render(
        episode["episode_id"],
        1,
        workflow_name="minimax_h3_ref2va",
        h3_mode="ref2va",
        conditioning_paths=[identity],
    )
    segment_video = _write_video(root / f"{series_id}-segment.mp4", "red")
    store.complete_render(render_id, segment_video)
    store.record_technical_qa(
        episode["episode_id"], 1, segment_video, True, {"passed": True}
    )
    store.record_review(episode["episode_id"], 1, "accepted", segment_video)
    ending = _write_reference(root / f"{series_id}-ending.png", b"accepted ending")
    store.record_tail(episode["episode_id"], 1, segment_video, ending, episode_end=True)
    final_video = _write_video(root / f"{series_id}-final.mp4", "blue")
    store.complete_episode(episode["episode_id"], "success", final_video)
    if approve_for_continuation:
        publication = store.begin_publication_attempt(
            episode["episode_id"],
            platforms=["threads"],
            episode_topic="A new turn",
            caption="An existing story continues.",
            hashtags="#Hero",
            media_path=final_video,
        )
        store.record_publication_review(
            episode["episode_id"], publication["attempt_number"], decision="accepted"
        )
    return episode


class _FixedRandomSource:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value

    def choice(self, values):
        return values[0]


class SerialVideoAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        values = dotenv_values(project_root / "media_overload.env")
        self.mysql_options = {
            "host": str(values.get("mysql_host") or ""),
            "port": int(values.get("mysql_port") or 3306),
            "user": str(values.get("mysql_user") or ""),
            "password": str(values.get("mysql_password") or ""),
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "connect_timeout": 5,
        }
        if not all(self.mysql_options[key] for key in ("host", "user", "password")):
            self.fail("Serial behavior tests require the configured real MySQL service.")
        self.test_database_name = ""
        env_keys = (
            "mysql_host",
            "mysql_port",
            "serial_video_mysql_database",
            "serial_video_mysql_user",
            "serial_video_mysql_password",
        )
        self.original_mysql_environment = {key: os.environ.get(key) for key in env_keys}

    def tearDown(self) -> None:
        for key, value in self.original_mysql_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _new_store(self) -> SerialVideoStore:
        if not self.test_database_name:
            self.test_database_name = f"mediaoverload_serial_test_{uuid.uuid4().hex}"
            connection = pymysql.connect(**self.mysql_options)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE `{self.test_database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
            finally:
                connection.close()
            self.addCleanup(self._drop_test_database)
            os.environ["mysql_host"] = str(self.mysql_options["host"])
            os.environ["mysql_port"] = str(self.mysql_options["port"])
            os.environ["serial_video_mysql_database"] = self.test_database_name
            os.environ["serial_video_mysql_user"] = str(self.mysql_options["user"])
            os.environ["serial_video_mysql_password"] = str(self.mysql_options["password"])
        return SerialVideoStore(self.test_database_name, mysql_options=self.mysql_options)

    def _drop_test_database(self) -> None:
        if not self.test_database_name:
            return
        if not self.test_database_name.startswith("mediaoverload_serial_test_"):
            raise AssertionError("Refusing to drop a database outside the strict test prefix.")
        connection = pymysql.connect(**self.mysql_options, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE `{self.test_database_name}`")
        finally:
            connection.close()

    def _mysql_one(self, store: SerialVideoStore, query: str, params: tuple = ()) -> dict:
        connection = pymysql.connect(**self.mysql_options, database=store.database_name)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query.replace("?", "%s"), params)
                return cursor.fetchone()
        finally:
            connection.close()

    def _mysql_all(self, store: SerialVideoStore, query: str, params: tuple = ()) -> list[dict]:
        connection = pymysql.connect(**self.mysql_options, database=store.database_name)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query.replace("?", "%s"), params)
                return cursor.fetchall()
        finally:
            connection.close()

    def test_user_given_a_new_series_story_when_the_creator_approves_an_edited_draft_then_mysql_keeps_the_exact_approved_version_before_series_activation(self) -> None:
        """User Given a complete story draft for a series with no character yet When the creator edits and approves it Then MySQL stores the exact approved text, SHA, source provenance, and review decision without activating a production series."""
        store = self._new_store()
        series_id = f"story-first-{uuid.uuid4().hex[:12]}"
        source = {
            "title": "Community repairs an old bridge",
            "source": "Example Daily",
            "url": "https://news.example.test/bridge",
            "created_at": "2026-10-24 09:00:00",
            "selected_at": "2026-10-24T10:00:00+08:00",
        }
        draft = store.save_series_story_draft(
            series_id,
            "# Draft\n\nThe lighthouse keeper meets a cloud.",
            user_idea="A lighthouse keeper helps a cloud.",
            inspiration=source,
            prompt_sha256="a" * 64,
            model_id="contract-test-model",
            artifact_path="output/serial_series/story.md",
        )
        reviewed_text = "# Approved\n\nThe keeper guides a cloud home and restores the harbor lights."

        approved = store.record_series_story_decision(
            draft["story_plan_id"],
            "accepted",
            reviewed_treatment=reviewed_text,
            reviewer="creator",
            notes="Ending is clear.",
        )

        self.assertEqual(draft["status"], "pending_review")
        self.assertEqual(draft["version"], 1)
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["approved_treatment"], reviewed_text)
        self.assertEqual(approved["approved_sha256"], hashlib.sha256(reviewed_text.encode("utf-8")).hexdigest())
        self.assertEqual(approved["inspiration"]["url"], source["url"])
        self.assertEqual(approved["reviewer"], "creator")
        self.assertEqual(approved["review_notes"], "Ending is clear.")
        self.assertIsNone(
            self._mysql_one(store, "SELECT series_id FROM series WHERE series_id = ?", (series_id,))
        )

        revised = store.save_series_story_draft(
            series_id,
            "# New draft\n\nA changed story.",
            user_idea="A revised idea.",
            inspiration={"mode": "none"},
            prompt_sha256="b" * 64,
            artifact_path="output/serial_series/story-v2.md",
        )
        rejected = store.record_series_story_decision(
            revised["story_plan_id"],
            "rejected",
            reviewer="creator",
            notes="The ending is unresolved.",
        )

        self.assertEqual(revised["version"], 2)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(
            store.get_latest_approved_series_story(series_id)["story_plan_id"],
            approved["story_plan_id"],
        )

    def test_user_given_a_pending_story_plan_when_creator_uses_story_review_cli_then_exact_approved_file_is_recorded(self) -> None:
        """User Given a pending series story has an editable registered Markdown file When the creator accepts it through the CLI Then MySQL records the exact reviewed text as approved."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            story_path = root / "story.md"
            reviewed_text = "# Approved story\n\nThe lighthouse keeper guides the lost cloud home."
            story_path.write_text(reviewed_text, encoding="utf-8")
            draft = store.save_series_story_draft(
                "story-review-cli",
                "# Draft\n\nThe lighthouse keeper meets a cloud.",
                user_idea="A lighthouse keeper helps a cloud.",
                prompt_sha256="a" * 64,
                model_id="contract-test-model",
                artifact_path=str(story_path),
            )

            app_main(
                [
                    "--serial-story-review-plan-id",
                    str(draft["story_plan_id"]),
                    "--serial-story-review-decision",
                    "accepted",
                    "--serial-story-reviewed-file",
                    str(story_path),
                ]
            )

            approved = store.get_latest_approved_series_story("story-review-cli")
            self.assertEqual(approved["status"], "approved")
            self.assertEqual(approved["approved_treatment"], reviewed_text)

    def test_user_given_a_creator_rejects_a_serial_story_in_discord_when_onboarding_records_the_decision_then_later_stages_are_blocked(self) -> None:
        """User Given a pending story draft When its Discord decision is rejected Then MySQL records rejection and onboarding returns a blocked story stage."""
        store = self._new_store()
        series_id = "rejected-onboarding-story"
        draft = store.save_series_story_draft(series_id, "A keeper guides a cloud home.")
        decision = HumanReviewDecision(
            status="rejected",
            selected_paths=[],
            edited_text="",
            reviewer="creator",
            review_mode="discord",
        )

        blocked = _record_serial_story_review(
            store, series_id, str(draft["story_plan_id"]), "A keeper guides a cloud home.", decision
        )

        self.assertEqual(blocked["onboarding_stage"], "story")
        self.assertEqual(store.get_series_story_plan(str(draft["story_plan_id"]))["status"], "rejected")

    def test_user_given_a_creator_accepts_a_character_with_a_selected_reference_when_onboarding_records_the_decision_then_identity_is_locked(self) -> None:
        """User Given an approved story and pending character card When Discord accepts the card and selects an image Then MySQL locks the reviewed card and selected reference."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = "approved-onboarding-character"
            story = store.save_series_story_draft(series_id, "A keeper guides a cloud home.")
            approved_story = store.record_series_story_decision(
                str(story["story_plan_id"]),
                "accepted",
                reviewed_treatment="A keeper guides a cloud home.",
            )
            card = store.save_series_character_card_draft(
                series_id, str(approved_story["story_plan_id"]), "Blue-gray seal post carrier."
            )
            reference = _write_reference(root / "identity.png", b"identity")
            decision = HumanReviewDecision(
                status="approved",
                selected_paths=[str(reference)],
                edited_text="",
                reviewer="creator",
                review_mode="discord",
            )

            blocked = _record_serial_character_review(
                store,
                series_id,
                str(card["character_card_id"]),
                "Tide",
                "Tide is a blue-gray seal post carrier.",
                decision,
            )

            approved_card = store.get_latest_approved_character_card(series_id)
            self.assertIsNone(blocked)
            self.assertEqual(approved_card["character_name"], "Tide")
            self.assertEqual(approved_card["reference_assets"][0]["path"], str(reference.resolve()))

    def test_user_given_a_creator_accepts_an_episode_outline_when_onboarding_records_the_decision_then_outline_is_approved(self) -> None:
        """User Given an approved story, character, and pending episode outline When Discord accepts the outline Then MySQL records the exact reviewed outline before production."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = "approved-onboarding-outline"
            story = store.save_series_story_draft(series_id, "A keeper guides a cloud home.")
            approved_story = store.record_series_story_decision(
                str(story["story_plan_id"]),
                "accepted",
                reviewed_treatment="A keeper guides a cloud home.",
            )
            card = store.save_series_character_card_draft(
                series_id, str(approved_story["story_plan_id"]), "Blue-gray seal post carrier."
            )
            store.record_series_character_card_decision(
                str(card["character_card_id"]),
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a blue-gray seal post carrier.",
                reference_paths=[_write_reference(root / "identity.png", b"identity")],
            )
            outline = store.save_series_episode_outline_draft(
                series_id, 1, "The lighthouse goes dark", "Tide clears salt from the lantern."
            )
            reviewed_text = "Episode 1: The lighthouse goes dark\n\nTide restores the harbor signal."
            decision = HumanReviewDecision(
                status="approved",
                selected_paths=[],
                edited_text="",
                reviewer="creator",
                review_mode="discord",
            )

            blocked = _record_serial_episode_outline_review(
                store,
                series_id,
                str(outline["outline_id"]),
                reviewed_text,
                decision,
            )

            approved_outline = store.get_latest_approved_series_episode_outline(series_id, 1)
            self.assertIsNone(blocked)
            self.assertEqual(approved_outline["approved_outline"], reviewed_text)

    def test_user_given_existing_character_reference_paths_when_onboarding_resolves_candidates_then_it_keeps_the_selected_files(self) -> None:
        """User Given the creator supplied fixed character reference images When onboarding resolves candidate files Then it returns those exact resolved paths without requesting another generation."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            supplied = _write_reference(root / "identity.png", b"identity")

            paths, failure = _resolve_serial_character_reference_paths(
                argparse.Namespace(serial_character_references=[str(supplied)]),
                output_root=root / "output",
                series_id="existing-reference",
                character_name="Hero",
                card_text="Approved character card",
            )

            self.assertEqual(paths, [str(supplied.resolve())])
            self.assertEqual(failure, "")

    def test_user_given_no_character_references_and_no_production_profile_when_onboarding_prepares_candidates_then_generation_is_blocked(self) -> None:
        """User Given no character reference images or production profile exist When onboarding prepares reference candidates Then it reports the missing profile before starting image generation."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            paths, failure = _resolve_serial_character_reference_paths(
                argparse.Namespace(
                    serial_character_references=[],
                    character_config=str(root / "missing-character.yaml"),
                ),
                output_root=root / "output",
                series_id="missing-profile",
                character_name="Hero",
                card_text="Approved character card",
            )

            self.assertEqual(paths, [])
            self.assertIn("Character production profile does not exist", failure)

    def test_user_given_onboarding_generation_arguments_when_the_first_episode_is_dispatched_then_auto_next_and_unreviewed_references_are_removed(self) -> None:
        """User Given CLI arguments were built for automatic serial continuation When onboarding dispatches its first episode Then it removes the auto-next flag and unreviewed reference pair while preserving the approved outline."""
        argv = [
            "--goal",
            "Approved first-episode outline",
            "--serial-auto-next",
            "--serial-character-reference",
            "generated-candidate.png",
            "--duration-seconds",
            "60",
        ]

        first_episode_argv = _serial_first_episode_argv(argv)

        self.assertEqual(
            first_episode_argv,
            ["--goal", "Approved first-episode outline", "--duration-seconds", "60"],
        )

    def test_user_given_a_character_proposal_and_selected_reference_when_approved_then_mysql_locks_the_reviewed_card_and_image_hash(self) -> None:
        """User Given an approved series story and a reviewed character card with one selected image When the creator approves the character Then MySQL stores the card version and exact reference image SHA without creating the production series."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"character-first-{uuid.uuid4().hex[:12]}"
            story_draft = store.save_series_story_draft(
                series_id,
                "A keeper helps a lost cloud return to the sea.",
            )
            story = store.record_series_story_decision(
                story_draft["story_plan_id"],
                "accepted",
                reviewed_treatment="A keeper helps a lost cloud return to the sea.",
            )
            character_draft = store.save_series_character_card_draft(
                series_id,
                story["story_plan_id"],
                "A blue-gray seal post carrier who is brave but absent-minded.",
                user_preferences="A young seal post carrier.",
                artifact_path=str(root / "character.md"),
            )
            reviewed_card = "Name: Tide\nAppearance: Blue-gray young seal in a red mail satchel."
            (root / "character.md").write_text(reviewed_card, encoding="utf-8")
            reference = _write_reference(root / "reference.png", b"approved character image")

            approved = store.record_series_character_card_decision(
                character_draft["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card=reviewed_card,
                reference_paths=[reference],
                reviewer="creator",
            )

            self.assertEqual(approved["status"], "approved")
            self.assertEqual(approved["character_name"], "Tide")
            self.assertEqual(approved["approved_card"], reviewed_card)
            self.assertEqual(approved["approved_sha256"], hashlib.sha256(reviewed_card.encode("utf-8")).hexdigest())
            self.assertEqual(len(approved["reference_assets"]), 1)
            self.assertEqual(approved["reference_assets"][0]["path"], str(reference.resolve()))
            self.assertEqual(approved["reference_assets"][0]["sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertEqual(store.get_latest_approved_character_card(series_id)["character_card_id"], character_draft["character_card_id"])
            self.assertIsNone(
                self._mysql_one(store, "SELECT series_id FROM series WHERE series_id = ?", (series_id,))
            )

    def test_user_given_a_story_draft_that_is_not_approved_when_character_card_creation_is_requested_then_the_store_refuses_the_character_stage(self) -> None:
        """User Given a pending story treatment When a character card is requested Then the store requires the story approval first."""
        store = self._new_store()
        series_id = f"character-needs-story-{uuid.uuid4().hex[:12]}"
        story = store.save_series_story_draft(series_id, "A story still awaiting review.")

        with self.assertRaisesRegex(ValueError, "approved series story"):
            store.save_series_character_card_draft(
                series_id,
                story["story_plan_id"],
                "A proposed character card.",
            )

    def test_user_given_approved_story_and_character_when_an_episode_outline_is_approved_then_the_outline_keeps_both_source_versions_and_waits_before_video(self) -> None:
        """User Given an approved story and character card When an episode outline is drafted and approved Then MySQL stores the exact readable outline with its source version IDs and hashes before any production episode exists."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"outline-first-{uuid.uuid4().hex[:12]}"
            story_draft = store.save_series_story_draft(
                series_id,
                "A keeper guides a lost cloud home and restores the harbor lights.",
            )
            story = store.record_series_story_decision(
                story_draft["story_plan_id"],
                "accepted",
                reviewed_treatment="A keeper guides a lost cloud home and restores the harbor lights.",
            )
            card_draft = store.save_series_character_card_draft(
                series_id,
                story["story_plan_id"],
                "Tide is a blue-gray seal post carrier.",
            )
            card = store.record_series_character_card_decision(
                card_draft["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a blue-gray seal post carrier.",
                reference_paths=[_write_reference(root / "tide.png", b"approved Tide reference")],
            )
            draft = store.save_series_episode_outline_draft(
                series_id,
                1,
                "The lighthouse goes dark",
                "Beat 1: Tide finds salt crystals blocking the lantern.\nShot 1: ...",
                episode_direction="Show why the harbor signal disappears.",
            )
            reviewed = "Episode 1: The lighthouse goes dark\n\nShot 1: Tide finds the salt crystals."
            outline = store.record_series_episode_outline_decision(
                draft["outline_id"],
                "accepted",
                reviewed_outline=reviewed,
                reviewer="creator",
            )

            self.assertEqual(outline["status"], "approved")
            self.assertEqual(outline["approved_outline"], reviewed)
            self.assertEqual(outline["approved_sha256"], hashlib.sha256(reviewed.encode("utf-8")).hexdigest())
            self.assertEqual(outline["story_plan_id"], story["story_plan_id"])
            self.assertEqual(outline["story_sha256"], story["approved_sha256"])
            self.assertEqual(outline["character_card_id"], card["character_card_id"])
            self.assertEqual(outline["character_sha256"], card["approved_sha256"])
            self.assertEqual(store.get_latest_approved_series_episode_outline(series_id, 1)["outline_id"], draft["outline_id"])
            self.assertIsNone(
                self._mysql_one(store, "SELECT series_id FROM series WHERE series_id = ?", (series_id,))
            )

    def test_user_given_a_story_change_after_an_episode_outline_draft_when_the_outline_is_approved_then_stale_lineage_is_rejected(self) -> None:
        """User Given an episode outline based on an approved story When a newer story version is approved before outline review Then the old outline cannot become approved."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"stale-outline-{uuid.uuid4().hex[:12]}"
            first_story = store.save_series_story_draft(series_id, "Original ending.")
            approved_first = store.record_series_story_decision(
                first_story["story_plan_id"], "accepted", reviewed_treatment="Original ending."
            )
            card_draft = store.save_series_character_card_draft(
                series_id, approved_first["story_plan_id"], "Tide is a seal."
            )
            store.record_series_character_card_decision(
                card_draft["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a seal.",
                reference_paths=[_write_reference(root / "tide.png", b"reference")],
            )
            outline = store.save_series_episode_outline_draft(
                series_id, 1, "First episode", "First outline."
            )
            revised_story = store.save_series_story_draft(series_id, "A new ending.")
            store.record_series_story_decision(
                revised_story["story_plan_id"], "accepted", reviewed_treatment="A new ending."
            )

            self.assertEqual(store.get_series_episode_outline(outline["outline_id"])["status"], "superseded")
            with self.assertRaisesRegex(ValueError, "pending episode outline"):
                store.record_series_episode_outline_decision(
                    outline["outline_id"], "accepted", reviewed_outline="Reviewed old outline."
                )

    def test_user_given_a_registered_identity_image_when_visual_evidence_roots_are_configured_then_its_parent_is_allowed_without_dropping_existing_roots(self) -> None:
        """User Given an explicitly registered serial identity image When visual evidence roots are configured Then its parent and existing caller roots remain allowed."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            reference_root = root / "references"
            reference_root.mkdir()
            reference = _write_reference(reference_root / "identity.png", b"identity")
            original = os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS")
            try:
                os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = str(root / "caller-images")

                _allow_runtime_output_for_visual_evidence(
                    output_root, additional_roots=[reference.parent]
                )

                allowed = {
                    Path(item).resolve()
                    for item in os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"].split(",")
                }
                self.assertIn((root / "caller-images").resolve(), allowed)
                self.assertIn(output_root.resolve(), allowed)
                self.assertIn(reference.parent.resolve(), allowed)
            finally:
                if original is None:
                    os.environ.pop("AGENTIC_ALLOWED_IMAGE_ROOTS", None)
                else:
                    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = original

    def test_user_given_no_serial_flags_when_daily_cli_is_parsed_then_serial_mode_stays_off(self) -> None:
        """User Given a normal daily request When its CLI arguments are parsed Then the existing non-serial route remains selected."""
        args = parse_args(["--goal", "make today's character video"])

        self.assertFalse(args.serial_mode)
        self.assertIsNone(args.series_id)
        self.assertEqual(args.media_type, "image")

    def test_user_given_a_story_first_serial_request_without_a_character_when_cli_is_parsed_then_news_is_selected_by_default(self) -> None:
        """User Given a new series story request with no fixed character When the CLI parses it Then story planning is allowed before character setup and random news inspiration is the default."""
        args = parse_args(["--serial-story-draft", "--goal", "A lighthouse keeper helps a cloud."])

        self.assertTrue(args.serial_story_draft)
        self.assertIsNone(args.character)
        self.assertEqual(args.serial_news_inspiration, "random")

    def test_user_given_a_custom_story_without_news_inspiration_when_cli_is_parsed_then_the_choice_is_explicit(self) -> None:
        """User Given a creator supplies a custom story and explicitly disables news When the CLI parses it Then the request keeps news inspiration disabled."""
        args = parse_args(
            [
                "--serial-story-draft",
                "--serial-news-inspiration", "none",
                "--goal", "A lighthouse keeper helps a cloud.",
            ]
        )

        self.assertEqual(args.serial_news_inspiration, "none")

    def test_user_given_an_approved_story_when_cli_parses_character_draft_without_preferences_then_character_creation_is_allowed(self) -> None:
        """User Given an approved story and no character preference When the creator starts character drafting Then the stage accepts no character image or character name yet."""
        args = parse_args(["--serial-character-draft", "--series-id", "story-2026-01"])

        self.assertTrue(args.serial_character_draft)
        self.assertEqual(args.serial_character_preferences, "")
        self.assertEqual(args.series_id, "story-2026-01")

    def test_user_given_a_character_card_and_image_when_cli_parses_approval_then_the_human_selects_the_name_and_reference(self) -> None:
        """User Given a pending character card When the creator approves it Then CLI approval requires a name, exact reviewed card file and selected identity image."""
        args = parse_args(
            [
                "--series-id", "story-2026-01",
                "--serial-character-card-review-id", "a" * 32,
                "--serial-character-card-review-decision", "accepted",
                "--serial-character-card-name", "Tide",
                "--serial-character-card-reviewed-file", "character.md",
                "--serial-character-reference", "tide.png",
            ]
        )

        self.assertEqual(args.serial_character_card_review_id, "a" * 32)
        self.assertEqual(args.serial_character_references, ["tide.png"])

    def test_user_given_approved_story_and_character_when_cli_parses_episode_outline_draft_then_it_requires_series_episode_and_topic_without_video_mode(self) -> None:
        """User Given approved story and character stages When an episode outline is drafted Then the CLI accepts episode number and topic without enabling serial video generation."""
        args = parse_args(
            [
                "--serial-episode-plan-draft",
                "--series-id", "story-2026-01",
                "--serial-episode-number", "1",
                "--episode-topic", "The lighthouse goes dark",
            ]
        )

        self.assertTrue(args.serial_episode_plan_draft)
        self.assertFalse(args.serial_mode)
        self.assertEqual(args.serial_episode_number, 1)
        self.assertEqual(args.episode_topic, "The lighthouse goes dark")

    def test_user_given_auto_next_without_a_series_id_when_cli_is_parsed_then_the_runtime_can_choose_the_series(self) -> None:
        """User Given auto-next mode without a chosen series When the CLI arguments are parsed Then the runtime is responsible for selecting a new or continuing series."""
        args = parse_args(
            [
                "--media-type", "long_video",
                "--serial-mode",
                "--serial-auto-next",
                "--character", "Hero",
            ]
        )

        self.assertTrue(args.serial_auto_next)
        self.assertIsNone(args.series_id)
        self.assertIsNone(args.goal)

    def test_user_given_a_series_with_approved_onboarding_when_cli_parses_video_generation_then_character_and_episode_outline_can_load_from_mysql(self) -> None:
        """User Given the creator selected an onboarded series When the CLI parses video generation Then it can load the approved character and episode outline from MySQL without resubmitting their text."""
        args = parse_args(
            [
                "--media-type", "long_video",
                "--serial-mode",
                "--series-id", "onboarded-series",
                "--character-config", "configs/characters/kirby.yaml",
            ]
        )

        self.assertTrue(args.serial_mode)
        self.assertEqual(args.series_id, "onboarded-series")
        self.assertIsNone(args.character)
        self.assertIsNone(args.goal)
        self.assertIsNone(args.episode_topic)

    def test_user_given_a_blocked_episode_in_a_completed_series_when_auto_next_selects_work_then_it_retries_before_random_selection(self) -> None:
        """User Given a failed episode from a completed series When automatic serial work is prepared Then it resumes that episode before the fresh-or-continuation roll."""
        class NewSeriesRoll:
            def random(self) -> float:
                return 0.99

            def choice(self, values):
                return values[0]

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("failed-series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
            store.record_plan(
                episode["episode_id"],
                segment_count=2,
                plan_payload={"goal": {"duration_seconds": 20}},
            )
            store.record_story(
                episode["episode_id"],
                [
                    {"visual": "The hero enters", "end_state": "A light appears"},
                    {"visual": "The hero follows", "end_state": "A door opens"},
                ],
            )
            store.fail_episode(episode["episode_id"], "Discord review timed out")
            store.set_series_lifecycle_status("failed-series", "completed", note="reviewed ending")

            selected = store.prepare_auto_next_context("Hero", random_source=NewSeriesRoll())

            self.assertEqual(selected["automation_mode"], "resume")
            self.assertEqual(selected["series_id"], "failed-series")
            self.assertEqual(selected["episode_number"], 1)
            self.assertEqual(selected["resume_episode_id"], episode["episode_id"])
            self.assertEqual(selected["episode_prompt"], "A hero enters a station")
            self.assertEqual(selected["episode_topic"], "月台盡頭的微光")

    def _start_rendering_episode(self, root: Path, series_id: str) -> dict[str, object]:
        identity = _write_reference(root / f"{series_id}-identity.png", b"identity")
        store = self._new_store()
        context = store.prepare_series_context(series_id, "Hero", [identity])
        episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
        store.record_plan(
            episode["episode_id"],
            segment_count=2,
            plan_payload={"goal": {"duration_seconds": 20}},
        )
        store.record_story(
            episode["episode_id"],
            [
                {"visual": "The hero enters", "end_state": "A light appears"},
                {"visual": "The hero follows", "end_state": "A door opens"},
            ],
        )
        store.begin_render(
            episode["episode_id"],
            1,
            workflow_name="minimax_h3_ref2va",
            h3_mode="ref2va",
            conditioning_paths=[identity],
        )
        return episode

    def _make_episode_silent(self, store: SerialVideoStore, episode_id: str, hours: int) -> None:
        silent_since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with store._connection(write=True) as connection:
            connection.execute(
                "UPDATE episodes SET updated_at = ? WHERE episode_id = ?",
                (silent_since, episode_id),
            )
            connection.execute(
                "UPDATE segments SET updated_at = ? WHERE episode_id = ?",
                (silent_since, episode_id),
            )

    def test_user_given_a_run_was_killed_mid_render_when_auto_next_selects_work_then_it_resumes_that_episode_instead_of_orphaning_the_series(self) -> None:
        """User Given a run died while rendering and nothing has touched the episode for hours When automatic serial work is prepared Then that episode is resumed before any new-or-continue roll."""
        class NewSeriesRoll:
            def random(self) -> float:
                return 0.99

            def choice(self, values):
                return values[0]

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            episode = self._start_rendering_episode(root, "killed-mid-render")
            store = self._new_store()
            self._make_episode_silent(store, episode["episode_id"], hours=12)

            selected = store.prepare_auto_next_context("Hero", random_source=NewSeriesRoll())

            self.assertEqual(selected["automation_mode"], "resume")
            self.assertEqual(selected["series_id"], "killed-mid-render")
            self.assertEqual(selected["resume_episode_id"], episode["episode_id"])

    def test_user_given_another_run_is_rendering_right_now_when_auto_next_selects_work_then_it_does_not_double_render_that_episode(self) -> None:
        """User Given an episode is mid-render with recent activity When another automatic run prepares work Then it leaves that episode alone."""
        class NewSeriesRoll:
            def random(self) -> float:
                return 0.99

            def choice(self, values):
                return values[0]

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            self._start_rendering_episode(root, "live-render")
            store = self._new_store()

            selected = store.prepare_auto_next_context("Hero", random_source=NewSeriesRoll())

            self.assertEqual(selected["automation_mode"], "new")
            self.assertNotEqual(selected["series_id"], "live-render")

    def test_user_given_a_resume_died_before_rendering_restarted_when_auto_next_selects_work_then_the_planned_episode_is_resumed(self) -> None:
        """User Given a blocked episode was reopened and the run died before any render When the episode has been silent for hours Then automatic serial work resumes it again."""
        class NewSeriesRoll:
            def random(self) -> float:
                return 0.99

            def choice(self, values):
                return values[0]

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            episode = self._start_rendering_episode(root, "died-after-reopen")
            store = self._new_store()
            store.fail_episode(episode["episode_id"], "Discord review timed out")
            store.resume_episode(episode["episode_id"])
            self._make_episode_silent(store, episode["episode_id"], hours=12)

            selected = store.prepare_auto_next_context("Hero", random_source=NewSeriesRoll())

            self.assertEqual(selected["automation_mode"], "resume")
            self.assertEqual(selected["resume_episode_id"], episode["episode_id"])

    def test_user_given_an_interrupted_rendering_episode_when_it_resumes_then_segment_one_is_runnable_again(self) -> None:
        """User Given an episode left in rendering by a killed run When the creator or automation resumes it Then no accepted work is invented and segment one is runnable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            episode = self._start_rendering_episode(root, "interrupted-resume")
            store = self._new_store()

            resumed = store.resume_episode(episode["episode_id"])

            self.assertEqual(resumed["completed_segment_count"], 0)
            self.assertEqual(resumed["next_segment_number"], 1)
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "planned")

    def _series_lifecycle(self, store: SerialVideoStore, series_id: str) -> str:
        with store._connection() as connection:
            row = connection.execute(
                "SELECT lifecycle_status FROM series WHERE series_id = ?", (series_id,)
            ).fetchone()
        return str(row["lifecycle_status"])

    def test_user_given_a_run_died_before_its_plan_was_saved_when_auto_next_selects_work_then_the_unresumable_series_is_discarded(self) -> None:
        """User Given an episode was created and its run died before any plan was saved When the episode has been silent for hours Then automatic serial work discards that series instead of leaving it active but unselectable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("died-before-plan", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
            self._make_episode_silent(store, episode["episode_id"], hours=12)

            selected = store.prepare_auto_next_context("Hero")

            self.assertEqual(self._series_lifecycle(store, "died-before-plan"), "discarded")
            self.assertNotEqual(selected["series_id"], "died-before-plan")

    def test_user_given_an_episode_failed_before_its_plan_was_saved_when_auto_next_selects_work_then_the_unresumable_series_is_discarded(self) -> None:
        """User Given an episode failed before any plan was saved When automatic serial work is prepared Then that series is discarded."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("failed-before-plan", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
            store.fail_episode(episode["episode_id"], "Plan could not be saved")

            store.prepare_auto_next_context("Hero")

            self.assertEqual(self._series_lifecycle(store, "failed-before-plan"), "discarded")

    def test_user_given_a_run_is_between_creating_an_episode_and_saving_its_plan_when_another_run_selects_work_then_the_series_stays_active(self) -> None:
        """User Given an episode was just created and its plan is about to be saved When another automatic run prepares work Then it does not discard the live series."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("live-before-plan", "Hero", [identity])
            store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")

            store.prepare_auto_next_context("Hero")

            self.assertEqual(self._series_lifecycle(store, "live-before-plan"), "active")

    def test_user_given_the_first_segment_failed_before_any_acceptance_when_episode_resumes_then_segment_one_is_runnable(self) -> None:
        """User Given a persisted episode whose first segment failed review When the creator resumes it Then the accepted prefix may be empty and segment one remains runnable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("first-segment-failure", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
            plan = {"goal": {"prompt": "A hero enters a station", "duration_seconds": 10}}
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload=plan)
            store.record_story(
                episode["episode_id"],
                [{"visual": "The hero enters", "end_state": "A light appears"}],
            )
            store.fail_episode(episode["episode_id"], "Discord review timed out")

            resumed = store.resume_episode(episode["episode_id"])

            self.assertEqual(resumed["completed_segment_count"], 0)
            self.assertEqual(resumed["next_segment_number"], 1)
            self.assertIn("script-plan", resumed["initial_node_outputs"])
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "planned")

    def test_user_given_story_planning_failed_before_any_story_was_saved_when_episode_resumes_then_saved_workflow_can_plan_it_again(self) -> None:
        """User Given a blocked episode has its plan and segments but no persisted story beats When it resumes before segment one Then it does not invent accepted work and leaves story planning runnable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("planning-failure", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="月台盡頭的微光")
            store.record_plan(
                episode["episode_id"],
                segment_count=1,
                plan_payload={"goal": {"prompt": "A hero enters a station", "duration_seconds": 10}},
            )
            store.fail_episode(episode["episode_id"], "Story planning provider failed")

            resumed = store.resume_episode(episode["episode_id"])

            self.assertEqual(resumed["next_segment_number"], 1)
            self.assertNotIn("script-plan", resumed["initial_node_outputs"])

    def test_user_given_ready_series_when_lifecycle_is_terminal_then_mysql_excludes_it_from_auto_continuation(self) -> None:
        """User Given active, completed, and discarded series with ready latest episodes When the MySQL continuation pool is read Then only the active series is eligible and each candidate points to its latest ready episode."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            expected: dict[str, str] = {}
            for series_id, outcome in (
                ("active-series", "active"),
                ("completed-series", "completed"),
                ("discarded-series", "discarded"),
            ):
                context = store.prepare_series_context(series_id, "Hero", [identity])
                episode = store.start_episode(context, f"{series_id} story", topic="新冒險")
                final_video = _write_video(root / f"{series_id}-final.mp4", "blue")
                final_video_sha256 = hashlib.sha256(final_video.read_bytes()).hexdigest()
                with store._connection(write=True) as connection:
                    connection.execute(
                        "UPDATE episodes SET status = 'ready', final_video_path = ?, final_video_sha256 = ? WHERE episode_id = ?",
                        (str(final_video.resolve()), final_video_sha256, episode["episode_id"]),
                    )
                    connection.execute(
                        "UPDATE series SET story_state_json = ? WHERE series_id = ?",
                        (
                            json.dumps(
                                {
                                    "version": 1,
                                    "episode_number": episode["episode_number"],
                                    "source_episode_id": episode["episode_id"],
                                    "final_video_sha256": final_video_sha256,
                                    "state_text": f"{series_id} story",
                                }
                            ),
                            series_id,
                        ),
                    )
                if outcome != "active":
                    store.set_series_lifecycle_status(series_id, outcome, note="creator decision")
                else:
                    expected[series_id] = episode["episode_id"]

            candidates = store.list_auto_continuation_candidates("Hero")

            self.assertEqual({item["series_id"] for item in candidates}, {"active-series"})
            self.assertEqual(candidates[0]["latest_episode_id"], expected["active-series"])
            self.assertEqual(candidates[0]["latest_episode_number"], 1)

    def test_user_given_no_eligible_existing_series_when_auto_next_runs_then_it_starts_a_new_series_regardless_of_weights(self) -> None:
        """User Given no existing series is eligible to continue When auto-next runs Then it chooses new-series onboarding even when configuration assigns no weight to new series."""
        store = self._new_store()

        selected = store.prepare_auto_next_context(
            "Hero",
            random_source=_FixedRandomSource(0.99),
            series_selection_weights={"new": 0, "continue": 1},
        )

        self.assertEqual(selected["automation_mode"], "new")
        self.assertTrue(selected["series_id"].startswith("auto-hero-"))
        self.assertEqual(
            self._mysql_one(store, "SELECT COUNT(*) AS total FROM series")["total"],
            0,
        )

    def test_user_given_an_eligible_series_and_default_weights_when_auto_next_rolls_then_new_and_continue_each_use_half_the_interval(self) -> None:
        """User Given an eligible existing series and default 0.5/0.5 weights When auto-next rolls below or at the midpoint Then it chooses new onboarding below the midpoint and continuation at the midpoint."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            _seed_accepted_existing_episode(
                store, root, "existing-series", approve_for_continuation=True
            )

            new_selection = store.prepare_auto_next_context(
                "Hero", random_source=_FixedRandomSource(0.49)
            )
            continue_selection = store.prepare_auto_next_context(
                "Hero", random_source=_FixedRandomSource(0.5)
            )

            self.assertEqual(new_selection["automation_mode"], "new")
            self.assertEqual(continue_selection["automation_mode"], "continue")
            self.assertEqual(continue_selection["series_id"], "existing-series")

    def test_user_given_an_eligible_series_and_custom_selection_weights_when_auto_next_rolls_then_it_uses_the_configured_ratio(self) -> None:
        """User Given an eligible existing series and configured 0.75/0.25 or 0.25/0.75 weights When the same roll is evaluated Then the selected mode follows each configured ratio."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            _seed_accepted_existing_episode(
                store, root, "existing-series", approve_for_continuation=True
            )

            new_favored = store.prepare_auto_next_context(
                "Hero",
                random_source=_FixedRandomSource(0.6),
                series_selection_weights={"new": 0.75, "continue": 0.25},
            )
            continue_favored = store.prepare_auto_next_context(
                "Hero",
                random_source=_FixedRandomSource(0.6),
                series_selection_weights={"new": 0.25, "continue": 0.75},
            )

            self.assertEqual(new_favored["automation_mode"], "new")
            self.assertEqual(continue_favored["automation_mode"], "continue")

    def test_user_given_an_approved_story_character_reference_and_episode_outline_when_a_new_serial_episode_starts_then_the_episode_records_all_approved_lineage(self) -> None:
        """User Given all onboarding stages are approved When serial generation starts a new series episode Then production is created from the locked character image and the episode records story, character and outline IDs and hashes."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"onboarded-production-{uuid.uuid4().hex[:12]}"
            story_draft = store.save_series_story_draft(series_id, "A keeper guides a lost cloud home.")
            story = store.record_series_story_decision(
                story_draft["story_plan_id"], "accepted", reviewed_treatment="A keeper guides a lost cloud home."
            )
            card_draft = store.save_series_character_card_draft(
                series_id, story["story_plan_id"], "Tide is a blue-gray seal carrier."
            )
            reference = _write_reference(root / "tide.png", b"approved Tide identity")
            character = store.record_series_character_card_decision(
                card_draft["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a blue-gray seal carrier.",
                reference_paths=[reference],
            )
            outline_draft = store.save_series_episode_outline_draft(
                series_id, 1, "The harbor signal", "Episode 1 story and approved shot list."
            )
            outline = store.record_series_episode_outline_decision(
                outline_draft["outline_id"],
                "accepted",
                reviewed_outline="Episode 1 story and approved shot list.",
            )

            context = store.prepare_series_context(
                series_id,
                "",
                require_story_first_for_new=True,
                require_episode_outline=True,
            )
            episode = store.start_episode(
                context,
                context["onboarding_lineage"]["episode_outline"],
                topic=context["onboarding_lineage"]["episode_topic"],
            )
            lineage = store.get_episode_onboarding_lineage(episode["episode_id"])

            self.assertEqual(context["character"], "Tide")
            self.assertEqual(context["identity_references"][0]["sha256"], character["reference_assets"][0]["sha256"])
            self.assertEqual(lineage["story_plan_id"], story["story_plan_id"])
            self.assertEqual(lineage["story_sha256"], story["approved_sha256"])
            self.assertEqual(lineage["character_card_id"], character["character_card_id"])
            self.assertEqual(lineage["character_sha256"], character["approved_sha256"])
            self.assertEqual(lineage["outline_id"], outline["outline_id"])
            self.assertEqual(lineage["outline_sha256"], outline["approved_sha256"])
            self.assertEqual(lineage["reference_assets"][0]["sha256"], character["reference_assets"][0]["sha256"])
            self.assertTrue(store.production_series_exists(series_id))

    def test_user_given_an_approved_character_reference_when_its_bytes_change_before_new_series_preparation_then_generation_is_blocked(self) -> None:
        """User Given a story, character image and outline have been approved When the image bytes change before preparation or episode start Then serial generation is blocked until the character is reviewed again."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"approved-reference-integrity-{uuid.uuid4().hex[:12]}"
            story_draft = store.save_series_story_draft(series_id, "A keeper guides a lost cloud home.")
            story = store.record_series_story_decision(
                story_draft["story_plan_id"], "accepted", reviewed_treatment="A keeper guides a lost cloud home."
            )
            card_draft = store.save_series_character_card_draft(
                series_id, story["story_plan_id"], "Tide is a blue-gray seal carrier."
            )
            approved_identity = b"approved Tide identity"
            reference = _write_reference(root / "tide.png", approved_identity)
            store.record_series_character_card_decision(
                card_draft["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a blue-gray seal carrier.",
                reference_paths=[reference],
            )
            outline_draft = store.save_series_episode_outline_draft(
                series_id, 1, "The harbor signal", "Episode 1 story and approved shot list."
            )
            store.record_series_episode_outline_decision(
                outline_draft["outline_id"],
                "accepted",
                reviewed_outline="Episode 1 story and approved shot list.",
            )
            reference.write_bytes(b"different Tide identity after approval")

            with self.assertRaises(ValueError):
                store.prepare_series_context(
                    series_id,
                    "",
                    require_story_first_for_new=True,
                    require_episode_outline=True,
                )
            self.assertFalse(store.production_series_exists(series_id))

            reference.write_bytes(approved_identity)
            context = store.prepare_series_context(
                series_id,
                "",
                require_story_first_for_new=True,
                require_episode_outline=True,
            )
            reference.write_bytes(b"different Tide identity after preparation")
            with self.assertRaises(ValueError):
                store.start_episode(
                    context,
                    context["onboarding_lineage"]["episode_outline"],
                    topic=context["onboarding_lineage"]["episode_topic"],
                )
            self.assertFalse(store.production_series_exists(series_id))

    def test_user_given_a_production_series_when_the_creator_approves_a_new_story_then_old_episode_assets_are_staled_and_generation_waits_for_new_character_and_outline_approvals(self) -> None:
        """User Given a production series with an approved story, character and outline When the creator approves a revised story Then prior episode outputs and identity references become stale until the revised character and episode outline are approved."""
        with _workspace_temporary_directory() as temporary:
            store = self._new_store()
            root = Path(temporary)
            series_id = f"story-revision-{uuid.uuid4().hex[:12]}"
            story_one = store.save_series_story_draft(series_id, "Tide restores the lighthouse signal.")
            story_one = store.record_series_story_decision(
                story_one["story_plan_id"], "accepted", reviewed_treatment="Tide restores the lighthouse signal."
            )
            card_one = store.save_series_character_card_draft(
                series_id, story_one["story_plan_id"], "Tide is a blue-gray seal."
            )
            ref_one = _write_reference(root / "tide-one.png", b"first Tide image")
            card_one = store.record_series_character_card_decision(
                card_one["character_card_id"],
                "accepted",
                character_name="Tide",
                reviewed_card="Tide is a blue-gray seal.",
                reference_paths=[ref_one],
            )
            outline_one = store.save_series_episode_outline_draft(
                series_id, 1, "The signal returns", "Episode 1: Tide cleans the lighthouse lens."
            )
            store.record_series_episode_outline_decision(
                outline_one["outline_id"],
                "accepted",
                reviewed_outline="Episode 1: Tide cleans the lighthouse lens.",
            )
            context_one = store.prepare_series_context(
                series_id, "Tide", require_story_first_for_new=True, require_episode_outline=True
            )
            episode_one = store.start_episode(
                context_one,
                context_one["onboarding_lineage"]["episode_outline"],
                topic=context_one["onboarding_lineage"]["episode_topic"],
            )

            story_two = store.save_series_story_draft(series_id, "Tide discovers a hidden second harbor.")
            story_two = store.record_series_story_decision(
                story_two["story_plan_id"],
                "accepted",
                reviewed_treatment="Tide discovers a hidden second harbor.",
            )

            self.assertEqual(store.get_episode(episode_one["episode_id"])["status"], "stale")
            self.assertEqual(store.get_series_character_card(card_one["character_card_id"])["status"], "superseded")
            self.assertEqual(store.get_series_episode_outline(outline_one["outline_id"])["status"], "superseded")
            stale_reference = self._mysql_one(
                store,
                "SELECT status FROM reference_assets WHERE series_id = ? AND role = 'character_identity'",
                (series_id,),
            )
            self.assertEqual(stale_reference["status"], "stale")

            with self.assertRaisesRegex(ValueError, "approved character card"):
                store.prepare_series_context(
                    series_id, "", require_story_first_for_new=True, require_episode_outline=True
                )

            card_two = store.save_series_character_card_draft(
                series_id, story_two["story_plan_id"], "Morrow is a white sea otter who knows hidden currents."
            )
            ref_two = _write_reference(root / "morrow.png", b"new Morrow image")
            card_two = store.record_series_character_card_decision(
                card_two["character_card_id"],
                "accepted",
                character_name="Morrow",
                reviewed_card="Morrow is a white sea otter who knows hidden currents.",
                reference_paths=[ref_two],
            )
            outline_two = store.save_series_episode_outline_draft(
                series_id, 1, "The hidden harbor", "Episode 1: Morrow leads Tide to the second harbor."
            )
            store.record_series_episode_outline_decision(
                outline_two["outline_id"],
                "accepted",
                reviewed_outline="Episode 1: Morrow leads Tide to the second harbor.",
            )

            context_two = store.prepare_series_context(
                series_id, "", require_story_first_for_new=True, require_episode_outline=True
            )
            episode_two = store.start_episode(
                context_two,
                context_two["onboarding_lineage"]["episode_outline"],
                topic=context_two["onboarding_lineage"]["episode_topic"],
            )

            self.assertEqual(context_two["character"], "Morrow")
            self.assertEqual(store.get_episode(episode_two["episode_id"])["episode_number"], 1)
            self.assertEqual(store.get_episode(episode_two["episode_id"])["version"], 2)
            self.assertEqual(store.get_episode_onboarding_lineage(episode_two["episode_id"])["character_card_id"], card_two["character_card_id"])
            self.assertEqual(
                self._mysql_one(store, "SELECT character_name FROM series WHERE series_id = ?", (series_id,))["character_name"],
                "Morrow",
            )

    def test_user_given_an_automatically_generated_episode_idea_when_topic_and_premise_are_validated_then_title_is_traditional_chinese(self) -> None:
        """User Given an automatic episode idea When MediaOverload validates its topic and premise Then it accepts a concise Traditional Chinese topic and rejects an unlabeled foreign-language title."""
        from agentic.runtime.prompt_tasks.serial_episodes import validate_serial_episode_idea

        valid = validate_serial_episode_idea(
            {"topic": "星光引路，找到回家的路", "premise": "Kirby follows a floating lantern seed through a garden."}
        )
        self.assertEqual(valid["topic"], "星光引路，找到回家的路")
        with self.assertRaises(ValueError):
            validate_serial_episode_idea(
                {"topic": "The Garden's Secret", "premise": "Kirby finds a hidden flower."}
            )

    def test_user_given_a_series_marked_terminal_when_it_is_marked_active_again_then_mysql_rejects_reopening(self) -> None:
        """User Given a series has been completed or discarded When automation tries to make it active again Then MySQL rejects that change so terminal series cannot silently re-enter the pool."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("closed-series", "Hero", [identity])
            store.start_episode(context, "A story with a final ending", topic="星花的終章")
            store.set_series_lifecycle_status("closed-series", "completed", note="final ending")

            with self.assertRaises(ValueError):
                store.set_series_lifecycle_status("closed-series", "active")

    def test_user_given_a_terminal_series_when_episode_one_is_requested_again_then_serial_context_rejects_reopening(self) -> None:
        """User Given a series is completed or discarded When its first episode is explicitly requested again Then serial planning rejects the request and keeps the series closed."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")

            for series_id, outcome in (("completed-series", "completed"), ("discarded-series", "discarded")):
                context = store.prepare_series_context(series_id, "Hero", [identity])
                episode = store.start_episode(context, "A story with a final ending", topic="星花的終章")
                with store._connection(write=True) as connection:
                    connection.execute(
                        "UPDATE episodes SET status = 'ready' WHERE episode_id = ?",
                        (episode["episode_id"],),
                    )
                store.set_series_lifecycle_status(series_id, outcome, note="creator decision")

                rerender_context = store.prepare_series_context(
                    series_id, "Hero", [], episode_number=1
                )
                with self.subTest(lifecycle_status=outcome), self.assertRaises(ValueError):
                    store.start_episode(
                        rerender_context, "A story with a final ending", topic="星花的終章"
                    )

    def test_user_given_a_titled_60_second_episode_when_its_approval_and_delivery_are_recorded_then_mysql_keeps_six_segments_and_each_platform_receipt(self) -> None:
        """User Given a titled 60-second serial episode When its plan, Discord approval, and three platform results are recorded Then MySQL stores its topic, six 10-second segment targets, and per-platform receipts."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("mysql-series", "Hero", [identity])
            episode = store.start_episode(
                context,
                "A hero guides fireflies home",
                topic="星花引路，陪迷路火螢回家",
                target_duration_seconds=60,
            )
            store.record_plan(
                episode["episode_id"],
                segment_count=6,
                plan_payload={"goal": {"duration_seconds": 60}},
            )
            video = _write_video(root / "episode.mp4", "purple")
            with store._connection(write=True) as connection:
                connection.execute(
                    """UPDATE episodes SET status = 'ready', final_video_path = ?,
                           final_video_sha256 = ? WHERE episode_id = ?""",
                    (str(video.resolve()), hashlib.sha256(video.read_bytes()).hexdigest(), episode["episode_id"]),
                )
            platforms = ["instagram_graph", "facebook", "youtube"]
            attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=platforms,
                episode_topic="星花引路，陪迷路火螢回家",
                caption="Kirby guides a lost firefly home with a star flower.",
                hashtags="#Kirby",
                media_path=video,
            )
            store.record_publication_review(
                episode["episode_id"],
                attempt["attempt_number"],
                decision="accepted",
                review_session_id="discord-session-2",
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                attempt["attempt_number"],
                results={platform: True for platform in platforms},
                receipts={platform: {"post_id": f"{platform}-post"} for platform in platforms},
                errors={},
                dispatch_status="success",
            )

            tables = {
                next(iter(row.values()))
                for row in self._mysql_all(store, "SHOW TABLES")
            }
            persisted_episode = store.get_episode(episode["episode_id"])
            persisted_segments = self._mysql_all(
                store,
                "SELECT target_duration_seconds FROM segments WHERE episode_id = ? ORDER BY ordinal",
                (episode["episode_id"],),
            )
            publications = store.get_episode_publications(episode["episode_id"])

            self.assertEqual(
                tables,
                {
                    "series", "episodes", "episode_plans", "segments", "render_attempts",
                    "reviews", "reference_assets", "episode_publications",
                    "series_story_plans", "series_character_cards", "series_episode_outlines",
                    "episode_onboarding_lineage",
                },
            )
            self.assertEqual(persisted_episode["episode_topic"], "星花引路，陪迷路火螢回家")
            self.assertEqual(persisted_episode["target_duration_seconds"], 60)
            self.assertEqual(len(persisted_segments), 6)
            self.assertTrue(all(float(row["target_duration_seconds"]) == 10 for row in persisted_segments))
            self.assertEqual({row["platform"] for row in publications}, set(platforms))
            self.assertTrue(all(row["status"] == "reported_success" for row in publications))
            self.assertTrue(all(json.loads(row["receipt_json"])["receipt"]["post_id"] for row in publications))

    def test_user_given_a_ready_serial_episode_when_discord_review_is_unavailable_then_registered_review_blocks_and_records_unavailable_attempt(self) -> None:
        """User Given a QA-approved serial episode and a local review service that reports unavailable When the registered final-publication review runs Then no asset is approved and MySQL records the publication attempt as unavailable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = f"review-unavailable-{uuid.uuid4().hex[:12]}"
            episode = _seed_accepted_existing_episode(store, root, series_id)
            video = root / f"{series_id}-final.mp4"

            class UnavailableDiscordReview:
                review_scope = ""
                allow_series_completion = False
                allow_series_discard = False

                def review_candidates(self, **request: object) -> HumanReviewDecision:
                    self.review_scope = str(request.get("review_scope") or "")
                    self.allow_series_completion = bool(request.get("allow_series_completion"))
                    self.allow_series_discard = bool(request.get("allow_series_discard"))
                    return HumanReviewDecision(
                        status="skipped",
                        selected_paths=[],
                        edited_text=str(request.get("text") or ""),
                        review_mode="auto",
                        fallback_reason="Discord review service is unavailable.",
                    )

            discord_review = UnavailableDiscordReview()

            skill_registry = SkillRegistry()
            tool_registry = ToolRegistry()
            register_media_service_tools(tool_registry, root)

            class FixedCaptionEngine:
                def prepare_publish_caption(self, _goal: object, **_kwargs: object) -> dict[str, str]:
                    return {"caption": "A new turn", "hashtags": "#story"}

            register_agent_social_skills(
                skill_registry,
                tool_registry,
                root,
                prompt_engine=FixedCaptionEngine(),
                discord_review=discord_review,
            )
            goal = GoalRequest(
                prompt="An existing story",
                media_type="publish_review",
                constraints={
                    "serial_mode": True,
                    "serial_database_name": store.database_name,
                    "serial_episode_id": episode["episode_id"],
                    "episode_topic": "A new turn",
                    "platforms": ["threads"],
                },
            )
            context = SkillContext(
                plan=ExecutionPlan(goal=goal, workflow_name="serial_publish", nodes=[]),
                node=ExecutionNode(
                    node_id="final-review",
                    skill_name="review.assets.select",
                    inputs={
                        "review_phase": "serial_final_publish",
                        "review_scope": "final_media",
                        "media_paths": [str(video)],
                        "platforms": ["threads"],
                    },
                ),
                state=RunState(goal={}, metadata={}),
            )

            result = skill_registry.get("review.assets.select").handler(context)

            publications = store.get_episode_publications(episode["episode_id"])
            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.outputs["selected_assets"], [])
            self.assertEqual(result.outputs["publication_attempt_number"], 1)
            self.assertEqual(len(publications), 1)
            self.assertEqual(publications[0]["status"], "review_unavailable")
            self.assertEqual(publications[0]["review_decision"], "unavailable")
            self.assertEqual(discord_review.review_scope, "final_media")
            self.assertTrue(discord_review.allow_series_completion)
            self.assertTrue(discord_review.allow_series_discard)
            self.assertEqual(Path(publications[0]["media_path"]).resolve(), video.resolve())

    def _review_first_segment(
        self,
        root: Path,
        store: SerialVideoStore,
        series_id: str,
        decision: HumanReviewDecision,
    ) -> tuple[SkillResult, dict[str, object], dict[str, object]]:
        """Render and QA segment one, then run the registered Discord segment review with a scripted creator decision."""
        identity = _write_reference(root / f"{series_id}-identity.png", b"identity")
        context = store.prepare_series_context(series_id, "Hero", [identity])
        episode = store.start_episode(context, "A hero follows a glowing flower", topic="守護星花")
        store.record_plan(
            episode["episode_id"],
            segment_count=2,
            plan_payload={"goal": {"duration_seconds": 20}},
        )
        render_id = store.begin_render(
            episode["episode_id"],
            1,
            workflow_name="minimax_h3_ref2va",
            h3_mode="ref2va",
            conditioning_paths=[identity],
        )
        segment_video = _write_video(root / f"{series_id}-segment-1.mp4", "red")
        store.complete_render(render_id, segment_video)
        store.record_technical_qa(episode["episode_id"], 1, segment_video, True, {"passed": True})

        class ScriptedDiscordReview:
            request: dict[str, object] = {}

            def review_candidates(self, **request: object) -> HumanReviewDecision:
                self.request = request
                return decision

        discord_review = ScriptedDiscordReview()
        skill_registry = SkillRegistry()
        tool_registry = ToolRegistry()
        register_media_service_tools(tool_registry, root)
        register_agent_social_skills(
            skill_registry, tool_registry, root, discord_review=discord_review
        )
        goal = GoalRequest(
            prompt="A hero follows a glowing flower",
            media_type="long_video",
            constraints={
                "serial_mode": True,
                "serial_database_name": store.database_name,
                "series_id": series_id,
                "serial_episode_id": episode["episode_id"],
                "serial_episode_number": 1,
            },
        )
        review_context = SkillContext(
            plan=ExecutionPlan(goal=goal, workflow_name="serial_long_video", nodes=[]),
            node=ExecutionNode(
                node_id="segment-1-review",
                skill_name="review.assets.select",
                inputs={
                    "limit": 1,
                    "review_all_candidates": True,
                    "review_scope": "final_video",
                    "review_phase": "segment_acceptance",
                    "segment_index": 0,
                    "require_human_review": True,
                    "media_paths": [str(segment_video)],
                },
            ),
            state=RunState(goal={}, metadata={}),
        )
        result = skill_registry.get("review.assets.select").handler(review_context)
        return result, dict(discord_review.request), episode

    def test_user_given_an_unsatisfying_first_segment_when_the_creator_discards_the_series_in_discord_then_it_is_never_resumed(self) -> None:
        """User Given segment one of a new episode awaits review When the creator discards the series instead of finishing it Then the series is discarded and the next automatic run no longer resumes this episode."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = f"discard-at-segment-{uuid.uuid4().hex[:12]}"

            result, request, _episode = self._review_first_segment(
                root,
                store,
                series_id,
                HumanReviewDecision(
                    status="rejected",
                    selected_paths=[],
                    edited_text="",
                    reviewer="creator",
                    review_mode="discord",
                    delivery={"status": "decision_received", "series_disposition": "discarded"},
                ),
            )

            lifecycle = self._mysql_one(
                store, "SELECT lifecycle_status FROM series WHERE series_id = ?", (series_id,)
            )["lifecycle_status"]
            self.assertTrue(request["allow_series_discard"])
            self.assertFalse(request["allow_series_completion"])
            self.assertEqual(result.status, "blocked")
            self.assertEqual(lifecycle, "discarded")
            self.assertEqual(store.list_auto_resume_candidates("Hero"), [])

    def test_user_given_an_unsatisfying_first_segment_when_the_creator_only_rejects_it_then_the_series_stays_resumable(self) -> None:
        """User Given segment one of a new episode awaits review When the creator rejects the segment without discarding the series Then the series stays active and the next automatic run resumes this episode."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = f"reject-at-segment-{uuid.uuid4().hex[:12]}"

            result, _request, episode = self._review_first_segment(
                root,
                store,
                series_id,
                HumanReviewDecision(
                    status="rejected",
                    selected_paths=[],
                    edited_text="",
                    reviewer="creator",
                    review_mode="discord",
                    delivery={"status": "decision_received", "series_disposition": ""},
                ),
            )

            lifecycle = self._mysql_one(
                store, "SELECT lifecycle_status FROM series WHERE series_id = ?", (series_id,)
            )["lifecycle_status"]
            resumable = [row["resumable_episode_id"] for row in store.list_auto_resume_candidates("Hero")]
            self.assertEqual(result.status, "blocked")
            self.assertEqual(lifecycle, "active")
            self.assertEqual(resumable, [episode["episode_id"]])

    def test_user_given_a_new_series_story_when_the_creator_discards_it_in_discord_then_onboarding_stops_without_creating_a_production_series(self) -> None:
        """User Given a pending story draft for a series that has not started production When the creator discards the series Then the draft is rejected, onboarding stops at the story stage, and no production series is created."""
        store = self._new_store()
        series_id = f"discard-at-story-{uuid.uuid4().hex[:12]}"
        draft = store.save_series_story_draft(series_id, "A keeper guides a cloud home.")

        blocked = _record_serial_story_review(
            store,
            series_id,
            str(draft["story_plan_id"]),
            "A keeper guides a cloud home.",
            HumanReviewDecision(
                status="rejected",
                selected_paths=[],
                edited_text="",
                reviewer="creator",
                review_mode="discord",
                delivery={"status": "decision_received", "series_disposition": "discarded"},
            ),
        )

        self.assertEqual(blocked["onboarding_stage"], "story")
        self.assertEqual(store.get_series_story_plan(str(draft["story_plan_id"]))["status"], "rejected")
        self.assertIsNone(store.get_production_series_character_name(series_id))

    def test_user_given_an_active_series_when_the_creator_discards_its_next_episode_outline_then_the_series_is_closed_as_discarded(self) -> None:
        """User Given an active production series with a pending next-episode outline When the creator discards the series while reviewing that outline Then the outline is rejected and the series lifecycle becomes discarded."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = f"discard-at-outline-{uuid.uuid4().hex[:12]}"
            _seed_accepted_existing_episode(store, root, series_id)
            story = store.save_series_story_draft(series_id, "A keeper guides a cloud home.")
            approved_story = store.record_series_story_decision(
                str(story["story_plan_id"]),
                "accepted",
                reviewed_treatment="A keeper guides a cloud home.",
            )
            card = store.save_series_character_card_draft(
                series_id, str(approved_story["story_plan_id"]), "Blue-gray seal post carrier."
            )
            store.record_series_character_card_decision(
                str(card["character_card_id"]),
                "accepted",
                character_name="Hero",
                reviewed_card="Hero is a blue-gray seal post carrier.",
                reference_paths=[root / f"{series_id}-identity.png"],
            )
            outline = store.save_series_episode_outline_draft(
                series_id, 2, "The lighthouse goes dark", "Hero clears salt from the lantern."
            )

            blocked = _record_serial_episode_outline_review(
                store,
                series_id,
                str(outline["outline_id"]),
                "Hero clears salt from the lantern.",
                HumanReviewDecision(
                    status="rejected",
                    selected_paths=[],
                    edited_text="",
                    reviewer="creator",
                    review_mode="discord",
                    delivery={"status": "decision_received", "series_disposition": "discarded"},
                ),
            )

            lifecycle = self._mysql_one(
                store, "SELECT lifecycle_status FROM series WHERE series_id = ?", (series_id,)
            )["lifecycle_status"]
            self.assertEqual(blocked["onboarding_stage"], "episode_outline")
            self.assertEqual(lifecycle, "discarded")

    def test_user_given_a_planned_episode_when_publication_is_requested_then_no_publication_attempt_is_created(self) -> None:
        """User Given a planned episode without final QA When publication is requested Then the runtime refuses to open Discord approval or create delivery rows."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("unready-publication", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            video = _write_video(root / "unverified-final.mp4", "blue")

            with self.assertRaises(ValueError):
                store.begin_publication_attempt(
                    episode["episode_id"],
                    platforms=["youtube"],
                    episode_topic="守護星花",
                    caption="A tiny flower shines.",
                    hashtags="#Hero",
                    media_path=video,
                )

            self.assertEqual(store.get_episode_publications(episode["episode_id"]), [])

    def test_user_given_a_partial_publication_when_an_episode_is_retried_then_only_unpublished_current_destinations_are_retried(self) -> None:
        """User Given an episode has a successful Instagram receipt and failed Facebook and YouTube receipts When it is retried with the current platform list including Threads Then the retry targets Facebook, Threads, and YouTube without duplicating Instagram."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("partial-publication", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            final_video = _write_video(root / "final.mp4", "blue")
            with store._connection(write=True) as connection:
                connection.execute(
                    """UPDATE episodes SET status = 'ready', final_video_path = ?,
                           final_video_sha256 = ? WHERE episode_id = ?""",
                    (str(final_video.resolve()), hashlib.sha256(final_video.read_bytes()).hexdigest(), episode["episode_id"]),
                )
            platforms = ["instagram_graph", "facebook", "youtube"]
            first_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=platforms,
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], first_attempt["attempt_number"], decision="accepted"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                first_attempt["attempt_number"],
                results={"instagram_graph": True, "facebook": False, "youtube": False},
                receipts={"instagram_graph": {"post_id": "instagram-post"}},
                errors={"facebook": "provider rejected request", "youtube": "timeout"},
                dispatch_status="partial",
            )

            character_platforms = ["instagram_graph", "facebook", "threads", "youtube"]
            retry_platforms = store.platforms_without_successful_receipt(
                episode["episode_id"], character_platforms
            )
            retry_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=retry_platforms,
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            self.assertEqual(
                retry_attempt["platforms"], ["facebook", "threads", "youtube"]
            )

    def test_user_given_a_completed_series_with_one_failed_platform_when_that_platform_is_retried_then_only_its_delivery_state_controls_dispatch(self) -> None:
        """User Given a completed series with Facebook reported successful and Threads failed When the creator approves a Threads-only retry and repeats the completed disposition Then Threads can finish while Facebook remains protected from duplicate delivery."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            series_id = f"completed-platform-retry-{uuid.uuid4().hex[:12]}"
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context(series_id, "Hero", [identity])
            episode = store.start_episode(
                context, "A hero protects a glowing flower", topic="守護星花"
            )
            final_video = _write_video(root / "final.mp4", "blue")
            with store._connection(write=True) as connection:
                connection.execute(
                    """UPDATE episodes SET status = 'ready', final_video_path = ?,
                           final_video_sha256 = ? WHERE episode_id = ?""",
                    (
                        str(final_video.resolve()),
                        hashlib.sha256(final_video.read_bytes()).hexdigest(),
                        episode["episode_id"],
                    ),
                )

            first_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["facebook", "threads"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], first_attempt["attempt_number"], decision="accepted"
            )
            store.set_series_lifecycle_status(
                series_id, "completed", note="creator completed the series"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                first_attempt["attempt_number"],
                results={"facebook": True, "threads": False},
                receipts={"facebook": {"post_id": "facebook-post"}},
                errors={"threads": "container status request failed"},
                dispatch_status="partial",
            )

            retry_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["threads"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], retry_attempt["attempt_number"], decision="accepted"
            )
            store.set_series_lifecycle_status(
                series_id, "completed", note="the approved retry keeps the series completed"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                retry_attempt["attempt_number"],
                results={"threads": True},
                receipts={"threads": {"post_id": "threads-post"}},
                errors={},
                dispatch_status="success",
            )

            publications = store.get_episode_publications(episode["episode_id"])
            statuses = {
                (row["attempt_number"], row["platform"]): row["status"]
                for row in publications
            }
            self.assertEqual(retry_attempt["platforms"], ["threads"])
            self.assertEqual(statuses[(1, "facebook")], "reported_success")
            self.assertEqual(statuses[(1, "threads")], "failed")
            self.assertEqual(statuses[(2, "threads")], "reported_success")

            with self.assertRaises(ValueError):
                store.begin_publication_attempt(
                    episode["episode_id"],
                    platforms=["facebook"],
                    episode_topic="守護星花",
                    caption="A tiny flower shines.",
                    hashtags="#Hero",
                    media_path=final_video,
                )

    def test_user_given_an_accepted_multi_platform_attempt_when_dispatch_finishes_then_each_platform_keeps_its_outcome_and_receipt(self) -> None:
        """User Given an accepted publication attempt spans several platforms When dispatch returns success, missing receipt, provider failure, and dry-run outcomes Then MySQL stores each platform's own status, receipt, and error."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("dispatch-outcomes", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            final_video = _write_video(root / "final.mp4", "blue")
            with store._connection(write=True) as connection:
                connection.execute(
                    """UPDATE episodes SET status = 'ready', final_video_path = ?,
                           final_video_sha256 = ? WHERE episode_id = ?""",
                    (str(final_video.resolve()), hashlib.sha256(final_video.read_bytes()).hexdigest(), episode["episode_id"]),
                )

            attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["instagram_graph", "facebook", "youtube"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], attempt["attempt_number"], decision="accepted"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                attempt["attempt_number"],
                results={"instagram_graph": True, "facebook": True, "youtube": False},
                receipts={"instagram_graph": {"post_id": "instagram-post"}},
                errors={"youtube": "provider timeout"},
                dispatch_status="partial",
            )

            dry_run_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["vimeo"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], dry_run_attempt["attempt_number"], decision="accepted"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                dry_run_attempt["attempt_number"],
                results={"vimeo": True},
                receipts={"vimeo": {"post_id": "dry-run-post"}},
                errors={},
                dispatch_status="dry_run",
            )

            rows = store.get_episode_publications(episode["episode_id"])
            by_platform = {row["platform"]: row for row in rows}

            self.assertEqual(by_platform["instagram_graph"]["status"], "reported_success")
            self.assertEqual(
                json.loads(by_platform["instagram_graph"]["receipt_json"])["receipt"]["post_id"],
                "instagram-post",
            )
            self.assertEqual(by_platform["facebook"]["status"], "receipt_missing")
            self.assertEqual(
                by_platform["facebook"]["error"],
                "provider reported success without a delivery receipt",
            )
            self.assertEqual(by_platform["youtube"]["status"], "failed")
            self.assertEqual(by_platform["youtube"]["error"], "provider timeout")
            self.assertEqual(by_platform["vimeo"]["status"], "dry_run")

    def test_user_given_an_automated_serial_episode_with_all_delivery_receipts_when_execution_finishes_then_completion_reports_success(self) -> None:
        """User Given an automated episode has reported-success receipts for every required platform When workflow completion is evaluated Then the result identifies the delivered episode as successful."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("automatic-complete", "Hero", [identity])
            context["automation_mode"] = "continue"
            episode = store.start_episode(
                context, "A hero protects a glowing flower", topic="守護星花"
            )
            final_video = _write_video(root / "final.mp4", "blue")
            with store._connection(write=True) as connection:
                connection.execute(
                    """UPDATE episodes SET status = 'ready', final_video_path = ?,
                           final_video_sha256 = ? WHERE episode_id = ?""",
                    (str(final_video.resolve()), hashlib.sha256(final_video.read_bytes()).hexdigest(), episode["episode_id"]),
                )
            attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["threads"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], attempt["attempt_number"], decision="accepted"
            )
            store.record_publication_dispatch(
                episode["episode_id"],
                attempt["attempt_number"],
                results={"threads": True},
                receipts={"threads": {"post_id": "thread-post"}},
                errors={},
                dispatch_status="success",
            )
            plan = ExecutionPlan(
                GoalRequest(prompt="A hero protects a glowing flower", constraints={"platforms": ["threads"]}),
                workflow_name="serial",
                nodes=[],
            )
            result = WorkflowRunResult("serial", "success", [], RunState(goal={}, metadata={}))

            completion = _serial_completion_result(
                argparse.Namespace(serial_auto_next=True, serial_require_delivery_receipts=False),
                store,
                context,
                store.get_episode(episode["episode_id"]),
                plan,
                result,
            )

            self.assertEqual(completion["status"], "success")
            self.assertEqual(completion["episode_id"], episode["episode_id"])
            self.assertEqual(completion["episode_number"], 1)

    def test_user_given_a_required_serial_delivery_without_receipts_when_execution_finishes_then_episode_is_blocked(self) -> None:
        """User Given a serial workflow completed without receipts for its required platforms When required delivery completion is evaluated Then the episode is blocked and the result reports failed delivery."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("missing-delivery", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower")
            plan = ExecutionPlan(
                GoalRequest(prompt="A hero protects a glowing flower", constraints={"platforms": ["threads"]}),
                workflow_name="serial",
                nodes=[],
            )
            result = WorkflowRunResult("serial", "success", [], RunState(goal={}, metadata={}))

            completion = _serial_completion_result(
                argparse.Namespace(serial_auto_next=False, serial_require_delivery_receipts=True),
                store,
                context,
                episode,
                plan,
                result,
            )

            self.assertEqual(completion["status"], "failed")
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "blocked")

    def test_user_given_a_manual_serial_episode_when_workflow_does_not_succeed_then_completion_keeps_episode_identity(self) -> None:
        """User Given a manually run serial episode When workflow completion is blocked Then the result keeps the episode identity and reports the blocked state."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("manual-blocked", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower")
            plan = ExecutionPlan(
                GoalRequest(prompt="A hero protects a glowing flower"),
                workflow_name="serial",
                nodes=[],
            )
            result = WorkflowRunResult("serial", "blocked", [], RunState(goal={}, metadata={}))

            completion = _serial_completion_result(
                argparse.Namespace(serial_auto_next=False, serial_require_delivery_receipts=False),
                store,
                context,
                episode,
                plan,
                result,
            )

            self.assertEqual(completion["status"], "blocked")
            self.assertEqual(completion["episode_id"], episode["episode_id"])
            self.assertEqual(completion["series_id"], "manual-blocked")

    def test_user_given_a_resumed_episode_with_a_saved_platform_list_when_character_targets_change_then_publish_nodes_use_the_current_character_targets(self) -> None:
        """User Given a resumed episode has a persisted platform list from an earlier run When the selected character currently allows another platform set Then the restored review and dispatch plan uses the current character targets and configs."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("resume-platforms", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            old_platforms = ["instagram_graph", "facebook", "youtube"]
            old_configs = {platform: {"enabled": True} for platform in old_platforms}
            saved_plan = {
                "goal": {
                    "prompt": "A hero protects a glowing flower",
                    "media_type": "long_video",
                    "duration_seconds": 30,
                    "style": "watercolor storybook animation",
                    "constraints": {
                        "series_id": "resume-platforms",
                        "character": "Hero",
                        "serial_episode_id": episode["episode_id"],
                        "serial_database_name": store.database_name,
                        "platforms": old_platforms,
                        "platform_configs": old_configs,
                    },
                },
                "workflow_name": "long_video",
                "nodes": [
                    {
                        "node_id": "serial-final-publish-review",
                        "skill_name": "review.assets.select",
                        "inputs": {"platforms": old_platforms},
                    },
                    {
                        "node_id": "serial-publish-process",
                        "skill_name": "publish.media.process",
                        "inputs": {"platforms": old_platforms},
                    },
                    {
                        "node_id": "serial-publish-caption",
                        "skill_name": "publish.caption.prepare",
                        "inputs": {"platforms": old_platforms},
                    },
                    {
                        "node_id": "serial-publish-dispatch",
                        "skill_name": "publish.social.dispatch",
                        "inputs": {
                            "platforms": old_platforms,
                            "platform_configs": old_configs,
                        },
                    },
                    {
                        "node_id": "persist-serial-publication-summary",
                        "skill_name": "agent.summary.persist",
                        "inputs": {"summary_metadata": {"platforms": old_platforms}},
                    },
                ],
                "metadata": {"serial_mode": True},
            }
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload=saved_plan)
            current_platforms = ["threads"]
            current_configs = {"threads": {"enabled": True, "config_folder_path": "threads"}}
            goal = GoalRequest(
                prompt="A hero protects a glowing flower",
                media_type="long_video",
                duration_seconds=30,
                style="watercolor storybook animation",
                constraints={
                    "series_id": "resume-platforms",
                    "character": "Hero",
                    "serial_episode_id": episode["episode_id"],
                    "serial_database_name": store.database_name,
                    "platforms": current_platforms,
                    "platform_configs": current_configs,
                },
            )

            restored_plan, _ = _prepare_serial_execution_plan(
                None,
                goal,
                store,
                store.get_episode(episode["episode_id"]),
                discord_review_timeout_seconds=720,
            )

            self.assertEqual(restored_plan.goal.constraints["platforms"], current_platforms)
            self.assertEqual(restored_plan.goal.constraints["platform_configs"], current_configs)
            for node in restored_plan.nodes:
                if "platforms" in node.inputs:
                    self.assertEqual(node.inputs["platforms"], current_platforms)
                if "platform_configs" in node.inputs:
                    self.assertEqual(node.inputs["platform_configs"], current_configs)
                summary_metadata = node.inputs.get("summary_metadata")
                if isinstance(summary_metadata, dict) and "platforms" in summary_metadata:
                    self.assertEqual(summary_metadata["platforms"], current_platforms)

    def test_user_given_an_approved_final_video_when_its_bytes_change_before_dispatch_then_the_publication_media_check_blocks_it(self) -> None:
        """User Given Discord approved a serial episode MP4 When its bytes change before social dispatch Then the publication media contract rejects those changed bytes."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("publication-hash", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            render_id = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            segment_video = _write_video(root / "segment.mp4", "red")
            store.complete_render(render_id, segment_video)
            store.record_technical_qa(
                episode["episode_id"], 1, segment_video, True, {"passed": True}
            )
            store.record_review(episode["episode_id"], 1, "accepted", segment_video)
            ending = _write_reference(root / "ending.png", b"accepted ending")
            store.record_tail(episode["episode_id"], 1, segment_video, ending, episode_end=True)
            final_video = _write_video(root / "final.mp4", "blue")
            store.complete_episode(episode["episode_id"], "success", final_video)
            attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["youtube"],
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"], attempt["attempt_number"], decision="accepted"
            )
            store.assert_publication_media_current(
                episode["episode_id"], attempt["attempt_number"]
            )

            with final_video.open("ab") as handle:
                handle.write(b"changed after Discord approval")

            with self.assertRaises(ValueError):
                store.assert_publication_media_current(
                    episode["episode_id"], attempt["attempt_number"]
                )

    def test_user_given_a_discarded_series_when_publication_is_requested_then_no_publication_attempt_is_created(self) -> None:
        """User Given a creator discarded a series When publication is requested for one of its episodes Then the runtime refuses to create a new publication attempt."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("discarded-publication", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower", topic="守護星花")
            final_video = _write_video(root / "final.mp4", "blue")
            with store._connection(write=True) as connection:
                connection.execute(
                    "UPDATE episodes SET status = 'ready', final_video_path = ?, final_video_sha256 = ? WHERE episode_id = ?",
                    (str(final_video.resolve()), hashlib.sha256(final_video.read_bytes()).hexdigest(), episode["episode_id"]),
                )
            store.set_series_lifecycle_status("discarded-publication", "discarded", note="creator decision")

            with self.assertRaises(ValueError):
                store.begin_publication_attempt(
                    episode["episode_id"],
                    platforms=["youtube"],
                    episode_topic="守護星花",
                    caption="A tiny flower shines.",
                    hashtags="#Hero",
                    media_path=final_video,
                )

    def test_user_given_incomplete_serial_identity_when_cli_is_parsed_then_request_is_rejected(self) -> None:
        """User Given serial mode without its explicit series ID When CLI arguments are parsed Then the request is rejected before planning."""
        with self.assertRaises(SystemExit):
            parse_args(["--goal", "episode premise", "--serial-mode", "--character", "Kirby"])

    def test_user_given_an_existing_series_when_its_cli_request_is_parsed_then_persisted_identity_images_need_not_be_repeated(self) -> None:
        """User Given an existing serial series When its CLI arguments omit identity image paths Then the stored series references can be used."""
        args = parse_args(
            [
                "--goal", "episode premise",
                "--episode-topic", "A new topic",
                "--serial-mode",
                "--series-id", "kirby",
                "--character", "Kirby",
            ]
        )

        self.assertTrue(args.serial_mode)
        self.assertEqual(args.serial_character_references, [])

    def test_user_given_a_blocked_serial_episode_when_resume_flags_are_parsed_then_resume_is_distinct_from_rerender(self) -> None:
        """User Given a blocked episode When its resume CLI request is parsed Then it resumes that version without selecting a rerender version."""
        args = parse_args(
            [
                "--goal", "episode premise",
                "--serial-mode",
                "--series-id", "kirby",
                "--character", "Kirby",
                "--serial-resume-episode-number", "1",
            ]
        )

        self.assertEqual(args.serial_resume_episode_number, 1)
        self.assertIsNone(args.serial_episode_number)

        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--goal", "episode premise",
                    "--serial-mode",
                    "--series-id", "kirby",
                    "--character", "Kirby",
                    "--serial-resume-episode-number", "1",
                    "--serial-episode-number", "1",
                ]
            )

    def test_user_given_a_timed_out_review_when_manual_review_flags_are_parsed_then_the_exact_segment_and_video_are_required(self) -> None:
        """User Given a timed-out Discord review When a creator records a manual decision Then CLI input identifies its episode, segment, decision, and reviewed MP4."""
        args = parse_args(
            [
                "--goal", "episode premise",
                "--serial-mode",
                "--series-id", "kirby",
                "--character", "Kirby",
                "--serial-review-episode-number", "1",
                "--serial-review-segment-number", "2",
                "--serial-review-decision", "accepted",
                "--serial-reviewed-video", "segment-02.mp4",
            ]
        )

        self.assertEqual(args.serial_review_episode_number, 1)
        self.assertEqual(args.serial_review_segment_number, 2)
        self.assertEqual(args.serial_review_decision, "accepted")
        self.assertEqual(args.serial_reviewed_video, "segment-02.mp4")

    def test_user_given_a_qa_passed_timed_out_clip_when_manual_cli_accepts_its_path_then_ledger_binds_that_sha(self) -> None:
        """User Given a QA-passed clip with a timed-out Discord review When the creator accepts its exact MP4 through the CLI Then the ledger stores the decision and reviewed SHA."""
        import json

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            output_root = root / "output"
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            attempt = store.begin_render(
                episode["episode_id"], 1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            video = _write_video(root / "approved.mp4", "red")
            store.complete_render(attempt, video)
            store.record_technical_qa(episode["episode_id"], 1, video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "timeout")
            store.fail_episode(episode["episode_id"], "Discord review timeout")

            repo_root = Path(__file__).resolve().parents[2]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root / "agentic" / "src")
            env["serial_video_mysql_database"] = store.database_name
            env["serial_video_mysql_user"] = str(self.mysql_options["user"])
            env["serial_video_mysql_password"] = str(self.mysql_options["password"])
            env["mysql_host"] = str(self.mysql_options["host"])
            env["mysql_port"] = str(self.mysql_options["port"])
            completed = subprocess.run(
                [
                    sys.executable,
                    "-X", "utf8",
                    "-m", "agentic.app.main",
                    "--goal", "A hero enters a station",
                    "--media-type", "long_video",
                    "--serial-mode",
                    "--series-id", "series",
                    "--character", "Hero",
                    "--serial-review-episode-number", "1",
                    "--serial-review-segment-number", "1",
                    "--serial-review-decision", "accepted",
                    "--serial-reviewed-video", str(video),
                    "--output-dir", str(output_root),
                ],
                cwd=repo_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )

            receipt = json.loads(completed.stdout)
            review = store.get_segment(episode["episode_id"], 1)
            with video.open("rb") as handle:
                expected_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            self.assertEqual(receipt["decision"], "accepted")
            self.assertEqual(receipt["video_path"], str(video.resolve()))
            self.assertEqual(receipt["video_sha256"], expected_sha)
            self.assertEqual(review["status"], "accepted")
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "blocked")

    def test_user_given_a_nonpositive_discord_review_timeout_when_cli_is_parsed_then_request_is_rejected(self) -> None:
        """User Given a zero-second review timeout When CLI arguments are parsed Then the invalid operational limit is rejected."""
        with self.assertRaises(SystemExit):
            parse_args(["--goal", "episode premise", "--discord-review-timeout-seconds", "0"])

    def test_user_given_a_user_accepted_review_after_timeout_when_episode_is_resumed_then_prior_video_is_seeded_and_next_segment_is_selected(self) -> None:
        """User Given a timed-out review later accepted by the creator When the episode resumes Then its same accepted MP4 is reused and the next segment becomes runnable."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            plan_payload = {"episode": 1, "goal": {"prompt": "A hero enters a station"}}
            store.record_plan(episode["episode_id"], segment_count=2, plan_payload=plan_payload)
            store.record_story(
                episode["episode_id"],
                [
                    {"visual": "The hero enters", "end_state": "The door glows"},
                    {"visual": "The hero follows", "end_state": "A new room opens"},
                ],
            )
            attempt = store.begin_render(
                episode["episode_id"], 1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            video = _write_video(root / "approved.mp4", "red")
            store.complete_render(attempt, video)
            store.record_technical_qa(
                episode["episode_id"], 1, video, True, {"passed": True, "width": 320, "height": 180}
            )
            store.record_review(
                episode["episode_id"], 1, "timeout", details={"decision_source": "discord"}
            )
            store.record_review(
                episode["episode_id"], 1, "accepted", video,
                details={"decision_source": "user_in_chat", "reviewer": "user"},
            )
            store.fail_episode(episode["episode_id"], "Discord review timed out before any decision was received")

            resume = store.resume_episode(episode["episode_id"])

            self.assertEqual(resume["next_segment_number"], 2)
            self.assertEqual(resume["completed_segment_count"], 1)
            self.assertEqual(
                resume["initial_node_outputs"]["segment-video-01"]["video_path"],
                str(video.resolve()),
            )
            self.assertEqual(
                resume["initial_node_outputs"]["segment-review-01"]["human_decision"],
                "accepted",
            )
            self.assertNotIn("segment-tail-01", resume["initial_node_outputs"])
            self.assertEqual(store.get_episode(episode["episode_id"])["status"], "planned")
            self.assertEqual(store.get_plan_payload(episode["episode_id"]), plan_payload)

    def test_user_given_serial_story_segments_when_story_planning_completes_then_episode_keeps_the_exact_beats_for_resume(self) -> None:
        """User Given a blocked episode with an accepted render and saved tail When it resumes Then the accepted video, delivered video, tail and story plan are restored for the next segment."""
        import json

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=2, plan_payload={"episode": 1})
            goal = GoalRequest(
                prompt="A hero enters a station",
                constraints={
                    "serial_mode": True,
                    "serial_episode_id": episode["episode_id"],
                    "serial_database_name": store.database_name,
                },
            )
            plan = ExecutionPlan(goal=goal, workflow_name="long_video", nodes=[])
            state = RunState(goal={}, metadata={}, node_outputs={"idea-brief": {"creative_brief": goal.prompt}})
            node = ExecutionNode(
                "script-plan",
                "agent.story.segment",
                inputs={"segment_count": 2, "tone": "coherent progression"},
                depends_on=["idea-brief"],
            )
            generated_segments = [
                {"segment_id": "segment-1", "visual": "The hero enters the station."},
                {"segment_id": "segment-2", "visual": "The hero follows a light inside."},
            ]

            class StoryPlannerFake:
                def segment_story(self, *_args: object, **_kwargs: object) -> list[dict[str, str]]:
                    return generated_segments

            result = AgentPlanningSkills(prompt_engine=StoryPlannerFake()).segment_story(
                SkillContext(plan, node, state)
            )

            persisted = json.loads(store.get_episode(episode["episode_id"])["story_json"])
            self.assertEqual(result.status, "success")
            self.assertEqual(persisted, result.outputs["segments"])

    def test_user_given_a_persisted_render_output_when_runner_resumes_then_the_saved_media_is_reused(self) -> None:
        """User Given a continuation plan with a persisted render output When the workflow resumes Then downstream skills receive that output without rendering it again."""
        registry = SkillRegistry()

        def render_again(_context: object) -> SkillResult:
            raise AssertionError("The persisted render must be reused during resume.")

        def package(context: object) -> SkillResult:
            state = getattr(context, "state")
            return SkillResult(
                status="success",
                outputs={"video_path": state.node_outputs["render-segment-01"]["video_path"]},
            )

        registry.register("render", render_again, "Render a segment")
        registry.register("package", package, "Package accepted media")
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="continue the accepted story"),
            workflow_name="serial_resume",
            nodes=[
                ExecutionNode("render-segment-01", "render"),
                ExecutionNode("package", "package", depends_on=["render-segment-01"]),
            ],
        )

        result = WorkflowRunner(registry).run(
            plan,
            initial_node_outputs={
                "render-segment-01": {"video_path": "accepted-segment.mp4"}
            },
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.state.node_outputs["package"]["video_path"], "accepted-segment.mp4")

    def test_user_given_a_series_when_its_next_episode_is_prepared_then_references_include_current_identity_and_approved_prior_tail(self) -> None:
        """User Given a human-approved completed episode and unchanged identity images When the next episode is prepared Then Ref2VA references include its approved ending frame and the fixed character images."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "kirby.png", b"identity version one")
            context = store.prepare_series_context("kirby", "Kirby", [identity])
            episode = store.start_episode(context, "Kirby finds a moon seed")
            store.record_plan(episode["episode_id"], segment_count=2, plan_payload={"episode": 1})
            store.record_story(
                episode["episode_id"],
                [
                    {"visual": "Kirby finds the seed", "end_state": "The seed glows"},
                    {"visual": "Kirby plants the seed", "end_state": "A moon flower opens"},
                ],
            )

            first_attempt = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            first_video = _write_video(root / "first.mp4", "red")
            store.complete_render(first_attempt, first_video)
            store.record_technical_qa(episode["episode_id"], 1, first_video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "accepted", first_video)
            first_tail = _write_reference(root / "first-tail.png", b"rendered final frame one")
            store.record_tail(episode["episode_id"], 1, first_video, first_tail)

            second_attempt = store.begin_render(
                episode["episode_id"],
                2,
                workflow_name="minimax_h3_i2va",
                h3_mode="i2va",
                conditioning_paths=[first_tail],
            )
            second_video = _write_video(root / "second.mp4", "blue")
            store.complete_render(second_attempt, second_video)
            store.record_technical_qa(episode["episode_id"], 2, second_video, True, {"passed": True})
            store.record_review(episode["episode_id"], 2, "accepted", second_video)
            episode_tail = _write_reference(root / "episode-tail.png", b"rendered final frame two")
            store.record_tail(episode["episode_id"], 2, second_video, episode_tail, episode_end=True)
            final_video = _write_video(root / "episode-final.mp4", "green")
            store.complete_episode(episode["episode_id"], "success", final_video)
            publication = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=["threads"],
                episode_topic="Kirby finds a moon seed",
                caption="Kirby finds a moon seed.",
                hashtags="",
                media_path=final_video,
            )
            store.record_publication_review(
                episode["episode_id"],
                publication["attempt_number"],
                decision="accepted",
                review_session_id="human-final-review",
                details={"reviewer": "operator"},
            )

            next_context = store.prepare_series_context("kirby", "Kirby", [])

            self.assertEqual(next_context["episode_number"], 2)
            self.assertEqual(
                [Path(item["path"]).resolve() for item in next_context["reference_manifest"]],
                [episode_tail.resolve(), identity.resolve()],
            )
            self.assertIn("Kirby finds a moon seed", next_context["story_context"])

    def test_user_given_an_accepted_review_when_its_video_bytes_change_then_continuation_is_blocked(self) -> None:
        """User Given a review accepted one video SHA When that video changes before continuation Then its old approval cannot supply a tail."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            attempt = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            video = _write_video(root / "approved.mp4", "red")
            store.complete_render(attempt, video)
            store.record_technical_qa(episode["episode_id"], 1, video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "accepted", video)
            video.write_bytes(b"changed after review")

            with self.assertRaises(ValueError):
                store.assert_can_continue(episode["episode_id"], 1, video)

    def test_user_given_a_registered_identity_image_changes_when_next_episode_context_is_prepared_then_serial_generation_is_blocked(self) -> None:
        """User Given a fixed character identity image changes after registration When the next episode context is prepared Then serial generation rejects the changed reference."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"registered identity")
            context = store.prepare_series_context("identity-change", "Hero", [identity])
            store.start_episode(context, "A hero protects the garden")
            identity.write_bytes(b"changed identity")

            with self.assertRaises(ValueError):
                store.prepare_series_context("identity-change", "Hero", [identity])

    def test_user_given_a_video_changed_after_technical_qa_when_the_user_accepts_it_then_review_is_rejected(self) -> None:
        """User Given a segment passed technical QA and its MP4 changed When the user accepts that path Then review rejects content that no longer matches the registered video SHA."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            attempt = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            video = _write_video(root / "reviewed.mp4", "red")
            store.complete_render(attempt, video)
            store.record_technical_qa(
                episode["episode_id"], 1, video, True, {"passed": True}
            )
            video.write_bytes(b"changed after technical QA")

            with self.assertRaises(ValueError):
                store.record_review(episode["episode_id"], 1, "accepted", video)

    def test_user_given_a_rejected_segment_when_its_tail_is_requested_then_continuation_is_blocked(self) -> None:
        """User Given a segment rejected by its human reviewer When the workflow requests its tail Then the ledger refuses to create a continuation asset."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            attempt = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            video = _write_video(root / "rejected.mp4", "red")
            store.complete_render(attempt, video)
            store.record_technical_qa(episode["episode_id"], 1, video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "rejected", video)

            with self.assertRaises(ValueError):
                store.assert_can_continue(episode["episode_id"], 1, video)

    def test_user_given_a_trimmed_segment_when_it_is_reviewed_then_qa_review_and_tail_bind_to_the_trimmed_video(self) -> None:
        """User Given a duration-normalized segment When QA, acceptance, and tail extraction run Then they bind to the trimmed MP4 produced from the H3 render."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            attempt = store.begin_render(
                episode["episode_id"], 1,
                workflow_name="minimax_h3_ref2va", h3_mode="ref2va", conditioning_paths=[identity],
            )
            raw_video = _write_video(root / "raw.mp4", "red")
            store.complete_render(attempt, raw_video)
            trimmed_video = _write_video(root / "trimmed.mp4", "blue")
            store.bind_delivered_video(episode["episode_id"], 1, trimmed_video)
            store.record_technical_qa(episode["episode_id"], 1, trimmed_video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "accepted", trimmed_video)
            tail = _write_reference(root / "tail.png", b"trimmed tail")

            approved = store.record_tail(episode["episode_id"], 1, trimmed_video, tail)

            self.assertEqual(Path(approved["path"]).resolve(), tail.resolve())
            with self.assertRaises(ValueError):
                store.assert_can_continue(episode["episode_id"], 1, raw_video)

    def test_user_given_an_accepted_trimmed_segment_when_a_later_review_times_out_then_resume_reuses_render_and_delivered_nodes(self) -> None:
        """User Given an H3 segment was trimmed and accepted before a later review timeout When the episode resumes Then its render node reuses the generated MP4 and its trim node reuses the approved delivered MP4."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("trimmed-resume", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station", topic="車站裡的星光")
            store.record_plan(episode["episode_id"], segment_count=2, plan_payload={"episode": 1})
            store.record_story(
                episode["episode_id"],
                [
                    {"visual": "The hero enters a station"},
                    {"visual": "The hero finds the lantern"},
                ],
            )
            render_id = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            generated_video = _write_video(root / "generated.mp4", "red")
            store.complete_render(render_id, generated_video)
            delivered_video = _write_video(root / "delivered.mp4", "blue")
            store.bind_delivered_video(episode["episode_id"], 1, delivered_video)
            store.record_technical_qa(
                episode["episode_id"], 1, delivered_video, True, {"passed": True}
            )
            store.record_review(episode["episode_id"], 1, "accepted", delivered_video)
            tail = _write_reference(root / "tail.png", b"accepted tail")
            store.record_tail(episode["episode_id"], 1, delivered_video, tail)
            store.fail_episode(episode["episode_id"], "segment 2 review timed out")

            resume = store.resume_episode(episode["episode_id"])
            outputs = resume["initial_node_outputs"]

            self.assertEqual(
                Path(outputs["segment-video-01"]["video_path"]).resolve(),
                generated_video.resolve(),
            )
            self.assertEqual(
                Path(outputs["segment-trim-01"]["video_path"]).resolve(),
                delivered_video.resolve(),
            )
            self.assertEqual(
                Path(outputs["segment-tail-01"]["frame_path"]).resolve(),
                tail.resolve(),
            )

    def test_user_given_an_accepted_downstream_segment_when_an_earlier_segment_is_rendered_again_then_old_downstream_is_stale(self) -> None:
        """User Given two accepted segments When the earlier segment receives a new render attempt Then the later segment's old approval is invalidated."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            store = self._new_store()
            identity = _write_reference(root / "identity.png", b"identity")
            context = store.prepare_series_context("series", "Hero", [identity])
            episode = store.start_episode(context, "A hero enters a station")
            store.record_plan(episode["episode_id"], segment_count=2, plan_payload={"episode": 1})
            first_attempt = store.begin_render(
                episode["episode_id"], 1,
                workflow_name="minimax_h3_ref2va", h3_mode="ref2va", conditioning_paths=[identity],
            )
            first_video = _write_video(root / "first.mp4", "red")
            store.complete_render(first_attempt, first_video)
            store.record_technical_qa(episode["episode_id"], 1, first_video, True, {"passed": True})
            store.record_review(episode["episode_id"], 1, "accepted", first_video)
            first_tail = _write_reference(root / "first-tail.png", b"tail")
            store.record_tail(episode["episode_id"], 1, first_video, first_tail)
            second_attempt = store.begin_render(
                episode["episode_id"], 2,
                workflow_name="minimax_h3_i2va", h3_mode="i2va", conditioning_paths=[first_tail],
            )
            second_video = _write_video(root / "second.mp4", "blue")
            store.complete_render(second_attempt, second_video)
            store.record_technical_qa(episode["episode_id"], 2, second_video, True, {"passed": True})
            store.record_review(episode["episode_id"], 2, "accepted", second_video)

            replacement = store.begin_render(
                episode["episode_id"], 1,
                workflow_name="minimax_h3_ref2va", h3_mode="ref2va", conditioning_paths=[identity],
            )

            self.assertEqual(store.get_segment(episode["episode_id"], 2)["status"], "stale")
            self.assertEqual(store.get_segment(episode["episode_id"], 1)["current_render_id"], replacement)

    def test_user_given_a_ready_later_episode_when_its_source_episode_is_regenerated_then_later_episode_lineage_is_stale(self) -> None:
        """User Given episode 2 follows episode 1's human-approved final video When episode 1 is rendered again Then episode 2's approval, tail, and final delivery are stale."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            identity = _write_reference(root / "identity.png", b"identity")
            store = self._new_store()
            tools = ToolRegistry()
            register_media_service_tools(tools, root / "media-output")

            first_context = store.prepare_series_context("garden", "Hero", [identity])
            first_episode = store.start_episode(first_context, "A hero protects a star flower")
            store.record_plan(first_episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            store.record_story(
                first_episode["episode_id"],
                [{"visual": "A hero shelters the star flower", "end_state": "The flower glows safely."}],
            )
            first_render = store.begin_render(
                first_episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            first_video = _write_video(root / "episode-1-segment.mp4", "red", duration_seconds=10)
            store.complete_render(first_render, first_video)
            store.record_technical_qa(first_episode["episode_id"], 1, first_video, True, {"passed": True})
            store.record_review(first_episode["episode_id"], 1, "accepted", first_video)
            first_tail = Path(
                str(
                    tools.call(
                        "media.extract_last_frame",
                        {
                            "video_path": str(first_video),
                            "output_path": str(root / "episode-1-tail.png"),
                        },
                    )["frame_path"]
                )
            )
            store.record_tail(first_episode["episode_id"], 1, first_video, first_tail, episode_end=True)
            first_final = _write_video(root / "episode-1-final.mp4", "green", duration_seconds=10)
            store.complete_episode(first_episode["episode_id"], "success", first_final)
            publication = store.begin_publication_attempt(
                first_episode["episode_id"],
                platforms=["threads"],
                episode_topic="A hero protects a star flower",
                caption="A hero protects the star flower.",
                hashtags="",
                media_path=first_final,
            )
            store.record_publication_review(
                first_episode["episode_id"],
                publication["attempt_number"],
                decision="accepted",
                review_session_id="human-final-review",
                details={"reviewer": "operator"},
            )

            second_context = store.prepare_series_context("garden", "Hero", [])
            second_episode = store.start_episode(second_context, "A hero follows the flower's light")
            store.record_plan(second_episode["episode_id"], segment_count=1, plan_payload={"episode": 2})
            second_render = store.begin_render(
                second_episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[item["path"] for item in second_context["reference_manifest"]],
            )
            second_video = _write_video(root / "episode-2-segment.mp4", "blue", duration_seconds=10)
            store.complete_render(second_render, second_video)
            store.record_technical_qa(second_episode["episode_id"], 1, second_video, True, {"passed": True})
            store.record_review(second_episode["episode_id"], 1, "accepted", second_video)
            second_tail = Path(
                str(
                    tools.call(
                        "media.extract_last_frame",
                        {
                            "video_path": str(second_video),
                            "output_path": str(root / "episode-2-tail.png"),
                        },
                    )["frame_path"]
                )
            )
            store.record_tail(second_episode["episode_id"], 1, second_video, second_tail, episode_end=True)
            second_final = _write_video(root / "episode-2-final.mp4", "yellow", duration_seconds=10)
            store.complete_episode(second_episode["episode_id"], "success", second_final)

            replacement = store.begin_render(
                first_episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )

            stale_episode = store.get_episode(second_episode["episode_id"])
            stale_segment = store.get_segment(second_episode["episode_id"], 1)
            review_stale = self._mysql_one(
                store, "SELECT stale FROM reviews WHERE render_id = ?", (second_render,)
            )["stale"]
            tail_status = self._mysql_one(
                store,
                "SELECT status FROM reference_assets WHERE source_render_id = ? AND role = 'episode_end'",
                (second_render,),
            )["status"]
            prior_render_status = self._mysql_one(
                store, "SELECT status FROM render_attempts WHERE render_id = ?", (second_render,)
            )["status"]

            self.assertEqual(store.get_segment(first_episode["episode_id"], 1)["current_render_id"], replacement)
            self.assertEqual(stale_episode["status"], "stale")
            self.assertEqual(stale_episode["is_current"], 0)
            self.assertEqual(stale_episode["final_video_path"], "")
            self.assertEqual(stale_segment["status"], "stale")
            self.assertIsNone(stale_segment["approved_render_id"])
            self.assertIsNone(stale_segment["tail_asset_id"])
            self.assertEqual(review_stale, 1)
            self.assertEqual(tail_status, "stale")
            self.assertEqual(prior_render_status, "superseded")

    def test_user_given_serial_mode_when_long_video_is_planned_then_first_segment_uses_ref2va_and_later_segments_use_i2va(self) -> None:
        """User Given explicit serial mode and validated references When the formal long-video plan is built Then it selects Ref2VA once, I2VA thereafter, and requires acceptance after each render."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with _workspace_temporary_directory() as temporary:
            identity = _write_reference(Path(temporary) / "identity.png", b"identity")
            goal = GoalRequest(
                prompt="A hero carries a moon seed to a quiet garden",
                media_type="long_video",
                duration_seconds=30,
                style="cinematic animation",
                constraints={
                    "serial_mode": True,
                    "series_id": "garden",
                    "character": "Hero",
                    "reference_manifest": [{"path": str(identity), "type": "image", "role": "subject"}],
                    "longvideo_review_policy": "every_segment",
                    "require_human_review": True,
                    "segment_count": 3,
                    "longvideo_width": 608,
                    "longvideo_height": 352,
                    "longvideo_length": 120,
                    "longvideo_steps": 16,
                },
            )

            plan = planner.build_plan(goal)

            self.assertEqual(plan.metadata["recipe_sequence"], ["reference_bundle", "anchor_first", "anchor_first"])
            reviews = [node for node in plan.nodes if "human-acceptance" in node.tags]
            self.assertEqual(len(reviews), 3)
            for node in reviews:
                self.assertTrue(node.inputs["require_human_review"])
            second_video = next(node for node in plan.nodes if node.node_id == "segment-video-02")
            self.assertEqual(second_video.inputs["h3_mode"] if "h3_mode" in second_video.inputs else second_video.inputs["recipe"], "anchor_first")
            self.assertEqual(second_video.inputs["anchor_nodes"]["first"], "segment-tail-01")

    def test_user_given_a_titled_30_second_serial_episode_when_final_qa_passes_then_discord_acceptance_gates_all_three_platforms(self) -> None:
        """User Given a 30-second serial episode with a topic and three targets When its plan is built Then QA, output collection, final review, and publication refer to the same post-processed video."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with _workspace_temporary_directory() as temporary:
            identity = _write_reference(Path(temporary) / "identity.png", b"identity")
            platforms = ["instagram_graph", "facebook", "youtube"]
            goal = GoalRequest(
                prompt="Kirby guides a lost firefly home with the star flower",
                media_type="long_video",
                duration_seconds=30,
                style="watercolor storybook animation",
                constraints={
                    "serial_mode": True,
                    "series_id": "garden",
                    "character": "Hero",
                    "episode_number": 2,
                    "episode_topic": "星花引路，陪迷路火螢回家",
                    "reference_manifest": [{"path": str(identity), "type": "image", "role": "identity"}],
                    "longvideo_review_policy": "every_segment",
                    "require_human_review": True,
                    "segment_count": 3,
                    "longvideo_width": 608,
                    "longvideo_height": 352,
                    "longvideo_length": 120,
                    "longvideo_steps": 16,
                    "platforms": platforms,
                    "platform_configs": {name: {"enabled": True} for name in platforms},
                    "publish_mode": "live",
                    "serial_database_name": "mediaoverload_serial",
                },
            )

            plan = planner.build_plan(goal)

            by_id = {node.node_id: node for node in plan.nodes}
            final_qa = by_id["longvideo-video-qa"]
            final_review = by_id["serial-final-publish-review"]
            dispatch = by_id["serial-publish-dispatch"]
            final_video_node = plan.metadata["final_video_node"]
            self.assertEqual(plan.metadata["segment_count"], 3)
            self.assertIn("longvideo-video-qa", final_review.depends_on)
            self.assertIn(final_video_node, final_qa.depends_on)
            self.assertEqual(
                by_id["collect-longvideo-outputs"].inputs["preferred_video_node"],
                final_video_node,
            )
            self.assertIn(final_video_node, final_review.depends_on)
            self.assertIn("segment-review-01", by_id["concat-final-video"].depends_on)
            self.assertIn("segment-review-02", by_id["concat-final-video"].depends_on)
            self.assertIn("segment-review-03", by_id["concat-final-video"].depends_on)
            self.assertEqual(final_review.inputs["review_scope"], "final_video")
            self.assertEqual(final_review.inputs["review_phase"], "serial_final_publish")
            self.assertEqual(final_review.inputs["platforms"], platforms)
            self.assertNotIn("caption_prefix", final_review.inputs)
            self.assertNotIn("prefix", by_id["serial-publish-caption"].inputs)
            self.assertIn("serial-final-publish-review", dispatch.depends_on)
            self.assertEqual(dispatch.inputs["platforms"], platforms)
            self.assertEqual(dispatch.inputs["publish_mode"], "live")
            self.assertIsNotNone(final_qa)

    def test_user_given_an_upgraded_serial_video_when_final_qa_runs_then_its_dimensions_do_not_block_review(self) -> None:
        """User Given a valid serial video at improved resolution When final QA runs Then its dimensions do not block review."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            identity = _write_reference(root / "identity.png", b"identity")
            goal = GoalRequest(
                prompt="Kirby follows a firefly home",
                media_type="long_video",
                duration_seconds=15,
                style="watercolor storybook animation",
                constraints={
                    "serial_mode": True,
                    "series_id": "garden",
                    "character": "Kirby",
                    "reference_manifest": [{"path": str(identity), "type": "image", "role": "identity"}],
                },
            )

            plan = planner.build_plan(goal)
            by_id = {node.node_id: node for node in plan.nodes}
            render = by_id["segment-video-01"]
            final_qa = by_id["longvideo-video-qa"]
            upgraded_width = render.inputs["width"] * 9 // 4
            upgraded_height = render.inputs["height"] * 9 // 4
            video_path = root / "upgraded.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c=blue:s={upgraded_width}x{upgraded_height}:r={int(final_qa.inputs['expected_fps'])}:d=15",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
            )
            tools = ToolRegistry()
            register_media_service_tools(tools, root / "media-output")
            result = tools.call(
                "media.video_qa",
                {
                    **final_qa.inputs,
                    "video_path": str(video_path),
                    "contact_sheet_path": str(root / "contact.jpg"),
                },
            )

        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["checks"]["dimensions"])

    def test_user_given_a_segment_trim_when_serial_plan_is_built_then_the_delivered_clip_is_reviewed_and_assembled(self) -> None:
        """User Given an H3 clip longer than its planned segment When the serial plan inserts duration trimming Then QA, review, tail extraction, and assembly use the delivered clip."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with _workspace_temporary_directory() as temporary:
            identity = _write_reference(Path(temporary) / "identity.png", b"identity")
            goal = GoalRequest(
                prompt="A hero carries a moon seed to a quiet garden",
                media_type="long_video",
                duration_seconds=30,
                style="cinematic animation",
                constraints={
                    "serial_mode": True,
                    "series_id": "garden",
                    "character": "Hero",
                    "reference_manifest": [{"path": str(identity), "type": "image", "role": "identity"}],
                    "segment_count": 3,
                    "longvideo_width": 608,
                    "longvideo_height": 352,
                    "longvideo_length": 360,
                    "longvideo_steps": 16,
                },
            )

            plan = planner.build_plan(goal)

            by_id = {node.node_id: node for node in plan.nodes}
            self.assertIn("segment-trim-01", by_id)
            self.assertIn("segment-trim-01", by_id["segment-qa-01"].depends_on)
            self.assertIn("segment-trim-01", by_id["segment-review-01"].depends_on)
            self.assertIn("segment-review-01", by_id["segment-tail-01"].depends_on)
            self.assertIn("segment-trim-01", by_id["concat-final-video"].depends_on)

    def test_user_given_serial_episode_when_final_qa_inspects_assembled_video_then_episode_is_ready_and_segment_review_stays_bound_to_its_render(self) -> None:
        """User Given a serial episode with every segment accepted When final QA inspects the assembled MP4 Then MySQL marks the assembled episode ready and keeps segment review bound to its render."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            identity = _write_reference(root / "identity.png", b"identity image")
            store = self._new_store()
            context = store.prepare_series_context("garden", "Hero", [identity])
            episode = store.start_episode(context, "A hero protects a glowing flower")
            store.record_plan(episode["episode_id"], segment_count=1, plan_payload={"episode": 1})
            render_id = store.begin_render(
                episode["episode_id"],
                1,
                workflow_name="minimax_h3_ref2va",
                h3_mode="ref2va",
                conditioning_paths=[identity],
            )
            segment_video = _write_video(root / "segment.mp4", "red", duration_seconds=10)
            store.complete_render(render_id, segment_video)
            store.record_technical_qa(
                episode["episode_id"], 1, segment_video, True, {"passed": True}
            )
            store.record_review(episode["episode_id"], 1, "accepted", segment_video)
            ending = _write_reference(root / "ending.png", b"accepted ending")
            store.record_tail(
                episode["episode_id"], 1, segment_video, ending, episode_end=True
            )
            assembled_video = _write_video(root / "assembled.mp4", "blue", duration_seconds=10)

            tools = ToolRegistry()
            register_media_service_tools(tools, root / "media-output")
            media = AgentMediaSkills(tools, root / "media-output")
            goal = GoalRequest(
                prompt="A hero protects a glowing flower",
                media_type="long_video",
                duration_seconds=10,
                constraints={
                    "serial_mode": True,
                    "serial_database_name": store.database_name,
                    "serial_episode_id": episode["episode_id"],
                },
            )
            node = ExecutionNode(
                node_id="longvideo-video-qa",
                skill_name="media.video.inspect",
                inputs={"video_path": str(assembled_video), "target_duration": 10},
            )
            plan = ExecutionPlan(goal=goal, workflow_name="longvideo_real_v2", nodes=[node])

            result = media.qa_video(
                SkillContext(plan=plan, node=node, state=RunState(goal={}, metadata={}))
            )

            self.assertEqual(result.status, "success")
            self.assertTrue(result.outputs["passed"])
            ready_episode = store.get_episode(episode["episode_id"])
            self.assertEqual(ready_episode["status"], "ready")
            self.assertEqual(ready_episode["final_video_path"], str(assembled_video.resolve()))
            self.assertEqual(store.get_segment(episode["episode_id"], 1)["current_render_id"], render_id)
            self.assertEqual(store.get_segment(episode["episode_id"], 1)["status"], "accepted")

            node.inputs["target_duration"] = 11
            failed_result = media.qa_video(
                SkillContext(plan=plan, node=node, state=RunState(goal={}, metadata={}))
            )

            self.assertEqual(failed_result.status, "failed")
            self.assertFalse(failed_result.outputs["passed"])
            self.assertEqual(store.get_segment(episode["episode_id"], 1)["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
