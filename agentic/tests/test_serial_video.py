from __future__ import annotations

import subprocess
import unittest
import os
import hashlib
import json
import random
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import pymysql
from dotenv import dotenv_values
from pymysql.cursors import DictCursor

from agentic.app.main import _allow_runtime_output_for_visual_evidence, parse_args
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext, SkillResult
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.serial_video import SerialVideoStore
from agentic.skills.agent_primitives import AgentMediaSkills, AgentPlanningSkills
from agentic.tools.media_services import register_media_service_tools


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

    def test_user_given_a_registered_identity_image_when_visual_evidence_roots_are_configured_then_its_parent_is_allowed_without_dropping_existing_roots(self) -> None:
        """User Given an explicitly registered serial identity image When visual evidence roots are configured Then its parent and existing caller roots remain allowed."""
        with TemporaryDirectory() as temporary:
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
        self.assertEqual(args.media_type, "long_video")

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

    def test_user_given_a_blocked_episode_in_a_completed_series_when_auto_next_selects_work_then_it_retries_before_random_selection(self) -> None:
        """User Given a failed episode from a completed series When automatic serial work is prepared Then it resumes that episode before the fresh-or-continuation roll."""
        class NewSeriesRoll:
            def random(self) -> float:
                return 0.99

            def choice(self, values):
                return values[0]

        with TemporaryDirectory() as temporary:
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

    def test_user_given_the_first_segment_failed_before_any_acceptance_when_episode_resumes_then_segment_one_is_runnable(self) -> None:
        """User Given a persisted episode whose first segment failed review When the creator resumes it Then the accepted prefix may be empty and segment one remains runnable."""
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
                with store._connection(write=True) as connection:
                    connection.execute(
                        "UPDATE episodes SET status = 'ready' WHERE episode_id = ?",
                        (episode["episode_id"],),
                    )
                if outcome != "active":
                    store.set_series_lifecycle_status(series_id, outcome, note="creator decision")
                else:
                    expected[series_id] = episode["episode_id"]

            candidates = store.list_auto_continuation_candidates("Hero")

            self.assertEqual({item["series_id"] for item in candidates}, {"active-series"})
            self.assertEqual(candidates[0]["latest_episode_id"], expected["active-series"])
            self.assertEqual(candidates[0]["latest_episode_number"], 1)

    def test_user_given_a_random_50_50_auto_choice_when_no_active_series_exists_then_the_runtime_starts_new(self) -> None:
        """User Given the automatic 50/50 decision When its random roll is below the continuation boundary or no eligible series exists Then it continues only with eligible history and otherwise starts new."""
        choose = SerialVideoStore.choose_auto_next_mode

        self.assertEqual(choose(0.49, has_continuation_candidates=True), "continue")
        self.assertEqual(choose(0.50, has_continuation_candidates=True), "new")
        self.assertEqual(choose(0.10, has_continuation_candidates=False), "new")

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
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
                caption="第 2 集｜主題：星花引路，陪迷路火螢回家",
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
                },
            )
            self.assertEqual(persisted_episode["episode_topic"], "星花引路，陪迷路火螢回家")
            self.assertEqual(persisted_episode["target_duration_seconds"], 60)
            self.assertEqual(len(persisted_segments), 6)
            self.assertTrue(all(float(row["target_duration_seconds"]) == 10 for row in persisted_segments))
            self.assertEqual({row["platform"] for row in publications}, set(platforms))
            self.assertTrue(all(row["status"] == "reported_success" for row in publications))
            self.assertTrue(all(json.loads(row["receipt_json"])["receipt"]["post_id"] for row in publications))

    def test_user_given_a_planned_episode_when_publication_is_requested_then_no_publication_attempt_is_created(self) -> None:
        """User Given a planned episode without final QA When publication is requested Then the runtime refuses to open Discord approval or create delivery rows."""
        with TemporaryDirectory() as temporary:
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

    def test_user_given_a_partial_publication_when_an_episode_is_retried_then_successful_platforms_are_not_dispatched_again(self) -> None:
        """User Given one platform has a reported-success receipt and two platforms failed When the episode is retried Then review and dispatch target only the two platforms without successful receipts."""
        with TemporaryDirectory() as temporary:
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

            pending = store.platforms_without_successful_receipt(
                episode["episode_id"], platforms
            )

            self.assertEqual(pending, ["facebook", "youtube"])
            with self.assertRaises(ValueError):
                store.begin_publication_attempt(
                    episode["episode_id"],
                    platforms=["instagram_graph"],
                    episode_topic="守護星花",
                    caption="A tiny flower shines.",
                    hashtags="#Hero",
                    media_path=final_video,
                )
            retry_attempt = store.begin_publication_attempt(
                episode["episode_id"],
                platforms=pending,
                episode_topic="守護星花",
                caption="A tiny flower shines.",
                hashtags="#Hero",
                media_path=final_video,
            )
            self.assertEqual(retry_attempt["platforms"], pending)

    def test_user_given_an_approved_final_video_when_its_bytes_change_before_dispatch_then_the_publication_media_check_blocks_it(self) -> None:
        """User Given Discord approved a serial episode MP4 When its bytes change before social dispatch Then the publication media contract rejects those changed bytes."""
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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

        with TemporaryDirectory() as temporary:
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
                    os.environ.get("PYTHON", "python"),
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
        with TemporaryDirectory() as temporary:
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
        """User Given a serial episode's story planner When its beats are generated Then the persisted episode story matches the beats sent to rendering."""
        import json

        with TemporaryDirectory() as temporary:
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
        """User Given a completed episode and unchanged identity images When the next episode is prepared Then Ref2VA references include its approved ending frame and the fixed character images."""
        with TemporaryDirectory() as temporary:
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

            next_context = store.prepare_series_context("kirby", "Kirby", [])

            self.assertEqual(next_context["episode_number"], 2)
            self.assertEqual(
                [Path(item["path"]).resolve() for item in next_context["reference_manifest"]],
                [episode_tail.resolve(), identity.resolve()],
            )
            self.assertIn("A moon flower opens", next_context["story_context"])

    def test_user_given_an_accepted_review_when_its_video_bytes_change_then_continuation_is_blocked(self) -> None:
        """User Given a review accepted one video SHA When that video changes before continuation Then its old approval cannot supply a tail."""
        with TemporaryDirectory() as temporary:
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

    def test_user_given_a_rejected_segment_when_its_tail_is_requested_then_continuation_is_blocked(self) -> None:
        """User Given a segment rejected by its human reviewer When the workflow requests its tail Then the ledger refuses to create a continuation asset."""
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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

    def test_user_given_an_accepted_downstream_segment_when_an_earlier_segment_is_rendered_again_then_old_downstream_is_stale(self) -> None:
        """User Given two accepted segments When the earlier segment receives a new render attempt Then the later segment's old approval is invalidated."""
        with TemporaryDirectory() as temporary:
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
        """User Given episode 2 is ready from episode 1's accepted ending When episode 1 is rendered again Then episode 2's approval, tail, and final delivery are stale."""
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
        """User Given a 30-second serial episode with a topic and three targets When its plan is built Then final QA precedes a Discord publish review and acceptance gates all three platform deliveries."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with TemporaryDirectory() as temporary:
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
            self.assertEqual(plan.metadata["segment_count"], 3)
            self.assertIn("longvideo-video-qa", final_review.depends_on)
            self.assertEqual(final_review.inputs["review_scope"], "final_video")
            self.assertEqual(final_review.inputs["review_phase"], "serial_final_publish")
            self.assertEqual(final_review.inputs["platforms"], platforms)
            self.assertNotIn("caption_prefix", final_review.inputs)
            self.assertNotIn("prefix", by_id["serial-publish-caption"].inputs)
            self.assertIn("serial-final-publish-review", dispatch.depends_on)
            self.assertEqual(dispatch.inputs["platforms"], platforms)
            self.assertEqual(dispatch.inputs["publish_mode"], "live")
            self.assertIsNotNone(final_qa)

    def test_user_given_a_segment_trim_when_serial_plan_is_built_then_the_delivered_clip_is_reviewed_and_assembled(self) -> None:
        """User Given an H3 clip longer than its planned segment When the serial plan inserts duration trimming Then QA, review, tail extraction, and assembly use the delivered clip."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(AssetRegistry(project_root, asset_root=project_root))
        with TemporaryDirectory() as temporary:
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
        with TemporaryDirectory() as temporary:
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
