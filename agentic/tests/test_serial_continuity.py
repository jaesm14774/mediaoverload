from __future__ import annotations

import subprocess
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import pymysql
from dotenv import dotenv_values
from pymysql.cursors import DictCursor

from agentic.runtime.serial_ledger import LedgerError
from agentic.runtime.serial_video import SerialVideoStore


def _workspace_temporary_directory() -> TemporaryDirectory:
    temp_root = Path(__file__).resolve().parents[2] / ".tmp-tests"
    temp_root.mkdir(parents=True, exist_ok=True)
    return TemporaryDirectory(dir=temp_root)


def _write_video(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=24:d=0.5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _write_image(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _outline(summary: str, *ledger_lines: str, goal: str = "追著印記去郵局") -> str:
    entries = "\n".join(ledger_lines)
    return (
        f"一、本集一句話\n{summary}\n\n二、節拍\n1. 開場\n\n"
        f"三、本集結尾狀態卡\n人物位置：郵筒旁。\n\n"
        f"四、線索帳本\n{entries}\n- 下集目標｜{goal}\n\n五、畫面提示\n1. 畫面\n"
    )


OPENING = _outline(
    "Kirby 撿到一封沒有地址的信。",
    "- 新增 主線｜這封信到底是寫給誰的",
    "- 新增｜信封角落有一個星形印記",
)


class SerialContinuityTests(unittest.TestCase):
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
        self.database = f"mediaoverload_serial_test_{uuid.uuid4().hex}"
        connection = pymysql.connect(**self.mysql_options)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE `{self.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            connection.close()
        self.addCleanup(self._drop_database)
        self.store = SerialVideoStore(self.database, mysql_options=self.mysql_options)

    def _drop_database(self) -> None:
        assert self.database.startswith("mediaoverload_serial_test_")
        connection = pymysql.connect(**self.mysql_options, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE `{self.database}`")
        finally:
            connection.close()

    def _approve_series(self, root: Path, series_id: str, *, identity_count: int = 1) -> list[Path]:
        store = self.store
        story = store.save_series_story_draft(series_id, "A story about a letter.")
        story = store.record_series_story_decision(
            story["story_plan_id"], "accepted", reviewed_treatment="A story about a letter."
        )
        card = store.save_series_character_card_draft(
            series_id, story["story_plan_id"], "Kirby is a round pink creature."
        )
        references = [
            _write_image(root / f"{series_id}-kirby-{index}.png", f"kirby identity {index}".encode())
            for index in range(identity_count)
        ]
        store.record_series_character_card_decision(
            card["character_card_id"],
            "accepted",
            character_name="Kirby",
            reviewed_card="Kirby is a round pink creature.",
            reference_paths=references,
        )
        return references

    def _approve_outline(self, series_id: str, number: int, text: str) -> None:
        draft = self.store.save_series_episode_outline_draft(
            series_id, number, f"第{number}集", text
        )
        self.store.record_series_episode_outline_decision(
            draft["outline_id"], "accepted", reviewed_outline=text
        )

    def _start_episode(self, series_id: str, number: int) -> dict:
        context = self.store.prepare_series_context(
            series_id, "", episode_number=number,
            require_story_first_for_new=True, require_episode_outline=True,
        )
        lineage = context["onboarding_lineage"]
        episode = self.store.start_episode(
            context, lineage["episode_outline"], topic=lineage["episode_topic"]
        )
        return {"context": context, "episode": episode}

    def _finish_episode(self, root: Path, started: dict, *, accept: bool) -> None:
        store = self.store
        episode_id = started["episode"]["episode_id"]
        number = started["episode"]["episode_number"]
        store.record_plan(episode_id, segment_count=1, plan_payload={"episode": number})
        render = store.begin_render(
            episode_id, 1, workflow_name="minimax_h3_ref2va", h3_mode="ref2va",
            conditioning_paths=[item["path"] for item in started["context"]["reference_manifest"]],
        )
        video = _write_video(root / f"segment-{episode_id}.mp4", "red")
        store.complete_render(render, video)
        store.record_technical_qa(episode_id, 1, video, True, {"passed": True})
        store.record_review(episode_id, 1, "accepted", video)
        tail = _write_image(root / f"tail-{episode_id}.png", f"tail {episode_id}".encode())
        store.record_tail(episode_id, 1, video, tail, episode_end=True)
        final = _write_video(root / f"final-{episode_id}.mp4", "blue")
        store.complete_episode(episode_id, "success", final)
        publication = store.begin_publication_attempt(
            episode_id, platforms=["threads"], episode_topic=f"第{number}集",
            caption="caption", hashtags="", media_path=final,
        )
        store.record_publication_review(
            episode_id, publication["attempt_number"],
            decision="accepted" if accept else "rejected",
        )

    def test_user_given_an_outline_that_breaks_a_continuity_rule_when_the_creator_approves_it_then_approval_is_refused(self) -> None:
        """User Given a series whose episode one is on the record When episode two's outline advances a thread that does not exist Then approval is refused, and an outline that advances a real thread is approved."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-gate-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            self._finish_episode(root, self._start_episode(series_id, 1), accept=True)

            draft = self.store.save_series_episode_outline_draft(
                series_id, 2, "第2集", _outline("第二集", "- 推進 T9｜不存在")
            )
            with self.assertRaises(LedgerError):
                self.store.record_series_episode_outline_decision(
                    draft["outline_id"], "accepted",
                    reviewed_outline=_outline("第二集", "- 推進 T9｜不存在"),
                )

            fixed = _outline("第二集", "- 推進 T2｜印記和郵局戳記一樣")
            draft = self.store.save_series_episode_outline_draft(series_id, 2, "第2集", fixed)
            approved = self.store.record_series_episode_outline_decision(
                draft["outline_id"], "accepted", reviewed_outline=fixed
            )
            self.assertEqual(approved["status"], "approved")

    def _checked_outline(self, series_id: str, *replies: str, episode_number: int = 1) -> str:
        from agentic.app.main import _generate_checked_episode_outline

        class ScriptedClient:
            """Answers with the given texts in order, then keeps giving the last one."""

            def __init__(self) -> None:
                self.replies = list(replies)

            def generate_text(self, system_prompt, user_prompt, **kwargs) -> str:
                return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        return _generate_checked_episode_outline(
            ScriptedClient(), self.store, series_id, episode_number, "system", "user"
        )

    def test_user_given_the_previous_episode_is_not_accepted_when_the_next_outline_is_drafted_then_it_stops_without_asking_for_a_new_ledger(self) -> None:
        """User Given episode one is finished but not yet accepted When episode two's outline is drafted Then drafting stops at once with that reason, because no rewritten ledger can fix it and the model must not be asked again."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-order-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            self._start_episode(series_id, 1)
            asked: list[str] = []

            from agentic.app.main import _generate_checked_episode_outline

            class OnceClient:
                def generate_text(self, system_prompt, user_prompt, **kwargs) -> str:
                    asked.append(user_prompt)
                    assert len(asked) == 1, "the model was asked again for a problem it cannot fix"
                    return _outline("第二集", "- 推進 T2｜印記和郵局戳記一樣")

            with self.assertRaisesRegex(LedgerError, "第 1 集"):
                _generate_checked_episode_outline(
                    OnceClient(), self.store, series_id, 2, "system", "user"
                )

    def test_user_given_a_good_story_with_a_malformed_ledger_when_the_outline_is_drafted_then_the_story_is_kept_and_only_the_ledger_is_fixed(self) -> None:
        """User Given the model wrote a readable first episode but misspelled a ledger entry or left the ledger out When the outline is drafted Then the creator receives that same story with a valid ledger, not a different rewritten story."""
        story = "一、本集一句話\nKirby 撿到一把鑰匙。\n\n二、節拍\n1. 開場：Kirby 撿起鑰匙。\n"
        misspelled = story + "\n四、線索帳本\n- 出演｜Dede\n- 下集目標｜找鎖\n"
        missing = story
        repaired_ledger = "四、線索帳本\n- 新增 主線｜鑰匙能開哪扇門\n- 下集目標｜找鎖"
        with _workspace_temporary_directory() as temporary:
            for first_draft in (misspelled, missing):
                series_id = f"ledger-repair-{uuid.uuid4().hex[:8]}"
                self._approve_series(Path(temporary), series_id)

                outline = self._checked_outline(series_id, first_draft, repaired_ledger)

                self.store.validate_episode_outline(series_id, 1, outline)
                self.assertIn("1. 開場：Kirby 撿起鑰匙。", outline)

    def test_user_given_a_ledger_the_model_cannot_fix_when_the_outline_is_drafted_then_it_stops_with_the_reason(self) -> None:
        """User Given a model that keeps writing an invalid ledger When the outline is drafted Then drafting stops with the ledger reason instead of looping or handing over an unchecked outline."""
        story = "一、本集一句話\nKirby 撿到一把鑰匙。\n\n四、線索帳本\n- 出演｜Dede\n"
        with _workspace_temporary_directory() as temporary:
            series_id = f"ledger-stuck-{uuid.uuid4().hex[:8]}"
            self._approve_series(Path(temporary), series_id)

            with self.assertRaises(LedgerError):
                self._checked_outline(series_id, story, "四、線索帳本\n- 出演｜Dede")

    def test_user_given_an_episode_that_is_not_accepted_yet_when_the_next_outline_is_checked_then_it_waits_for_that_acceptance(self) -> None:
        """User Given episode one is finished but its final review is still pending When episode two's outline is checked Then it is refused until episode one is accepted, so episode two is never planned on a ledger that is missing episode one."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-wait-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            started = self._start_episode(series_id, 1)
            second = _outline("第二集", "- 推進 T2｜印記和郵局戳記一樣")

            with self.assertRaisesRegex(LedgerError, "第 1 集"):
                self.store.validate_episode_outline(series_id, 2, second)

            self._finish_episode(root, started, accept=True)
            self.store.validate_episode_outline(series_id, 2, second)

    def test_user_given_an_accepted_episode_when_the_next_one_is_planned_then_open_threads_and_the_recap_come_from_the_record(self) -> None:
        """User Given episode one was accepted in the final review When the next episode is planned Then the planning context lists its open threads and recaps episode one with the goal it handed on."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-plan-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            self._finish_episode(root, self._start_episode(series_id, 1), accept=True)

            planning = self.store.get_series_planning_context(series_id, 2)

            self.assertIn("T1", planning["ledger_text"])
            self.assertIn("這封信到底是寫給誰的", planning["ledger_text"])
            self.assertIn("Kirby 撿到一封沒有地址的信", planning["recap_text"])
            self.assertIn("追著印記去郵局", planning["recap_text"])

    def test_user_given_an_episode_the_creator_did_not_accept_when_the_next_one_is_planned_then_its_threads_are_not_on_the_record(self) -> None:
        """User Given episode one finished but its final review was rejected When the next planning context is built Then the series has no threads from it."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-reject-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            self._finish_episode(root, self._start_episode(series_id, 1), accept=False)

            planning = self.store.get_series_planning_context(series_id, 2)

            self.assertNotIn("T1", planning["ledger_text"])
            self.assertEqual(planning["recap_text"], "")

    def test_user_given_an_episode_that_is_redone_when_the_next_one_is_planned_then_the_old_version_leaves_no_threads(self) -> None:
        """User Given episode one was accepted and then redone from a new outline When the next planning context is built Then only the redone version's threads remain until it is accepted."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"ledger-redo-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self._approve_outline(series_id, 1, OPENING)
            self._finish_episode(root, self._start_episode(series_id, 1), accept=True)
            redone = _outline("重拍的第一集", "- 新增 主線｜另一個問題")
            self._approve_outline(series_id, 1, redone)
            self._start_episode(series_id, 1)

            planning = self.store.get_series_planning_context(series_id, 2)

            self.assertNotIn("這封信到底是寫給誰的", planning["ledger_text"])

    def test_user_given_a_recurring_character_when_it_is_registered_then_an_outline_may_put_it_on_screen_and_unknown_names_are_refused(self) -> None:
        """User Given an approved series and a registered recurring character When an outline lists that character as on screen Then it is approved, while an unregistered name is refused."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-roster-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            self.store.register_series_companion(
                series_id, "小明", "戴草帽的瘦高青年，穿藍色吊帶褲。",
                _write_image(root / "xiaoming.png", b"xiaoming identity"),
            )

            known = _outline(
                "第一集", "- 新增 主線｜問題", "- 出場｜小明"
            )
            unknown = _outline("第一集", "- 新增 主線｜問題", "- 出場｜陌生人")
            draft = self.store.save_series_episode_outline_draft(series_id, 1, "第1集", unknown)
            with self.assertRaises(LedgerError):
                self.store.record_series_episode_outline_decision(
                    draft["outline_id"], "accepted", reviewed_outline=unknown
                )
            draft = self.store.save_series_episode_outline_draft(series_id, 1, "第1集", known)
            approved = self.store.record_series_episode_outline_decision(
                draft["outline_id"], "accepted", reviewed_outline=known
            )
            self.assertEqual(approved["status"], "approved")

    def test_user_given_a_series_with_two_recurring_characters_when_a_third_is_registered_then_it_is_refused(self) -> None:
        """User Given a series that already has two recurring characters When a third is registered Then it is refused because the first clip can only carry a few reference images."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-limit-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            for name in ("小明", "阿福"):
                self.store.register_series_companion(
                    series_id, name, f"{name}的外觀", _write_image(root / f"{name}.png", name.encode())
                )

            with self.assertRaises(ValueError):
                self.store.register_series_companion(
                    series_id, "第三人", "外觀", _write_image(root / "third.png", b"third")
                )

    def test_user_given_a_recurring_character_in_the_first_beat_when_the_episode_starts_then_its_reference_joins_the_bundle_in_a_stable_order(self) -> None:
        """User Given episode one puts a recurring character on screen in its first beat When the episode starts and renders its first clip Then the reference bundle holds the protagonist images then the character's image in the same order the store verifies."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-bundle-{uuid.uuid4().hex[:8]}"
            identity = self._approve_series(root, series_id, identity_count=2)
            companion_image = _write_image(root / "aaa-xiaoming.png", b"xiaoming identity")
            self.store.register_series_companion(series_id, "小明", "戴草帽的青年", companion_image)
            self._approve_outline(
                series_id, 1,
                _outline("第一集", "- 新增 主線｜問題", "- 出場｜小明"),
            )

            started = self._start_episode(series_id, 1)
            manifest_paths = [Path(item["path"]).resolve() for item in started["context"]["reference_manifest"]]

            self.assertEqual(
                manifest_paths,
                [*sorted(path.resolve() for path in identity), companion_image.resolve()],
            )
            self._finish_episode(root, started, accept=True)

    def test_user_given_a_recurring_character_who_is_not_in_the_first_beat_when_the_episode_starts_then_its_reference_is_left_out(self) -> None:
        """User Given a registered recurring character that the outline does not put on screen in the first beat When the episode starts Then the reference bundle holds only the protagonist images."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-absent-{uuid.uuid4().hex[:8]}"
            identity = self._approve_series(root, series_id)
            self.store.register_series_companion(
                series_id, "小明", "戴草帽的青年", _write_image(root / "xiaoming.png", b"xiaoming identity")
            )
            self._approve_outline(series_id, 1, OPENING)

            started = self._start_episode(series_id, 1)

            self.assertEqual(
                [Path(item["path"]).resolve() for item in started["context"]["reference_manifest"]],
                [path.resolve() for path in identity],
            )

    def test_user_given_two_recurring_characters_and_an_outline_that_puts_one_on_screen_when_the_video_cast_is_read_then_only_that_one_is_the_cast(self) -> None:
        """User Given two registered recurring characters and an approved outline that puts only one of them on screen When the episode's video cast is read Then it holds only that character and their look, and an outline with nobody on screen gives an empty cast, so the clip writer is never told to draw a character whose reference image is not in the bundle."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-video-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            for name in ("小明", "阿福"):
                self.store.register_series_companion(
                    series_id, name, f"{name}的外觀", _write_image(root / f"{name}.png", name.encode())
                )

            with_one = _outline("第一集", "- 新增 主線｜問題", "- 出場｜小明")
            with_nobody = _outline("第一集", "- 新增 主線｜問題")

            cast = self.store.get_episode_cast_text(series_id, with_one)

            self.assertIn("小明", cast)
            self.assertIn("小明的外觀", cast)
            self.assertNotIn("阿福", cast)
            self.assertEqual(self.store.get_episode_cast_text(series_id, with_nobody), "")

    def test_user_given_a_protagonist_with_three_images_when_a_recurring_character_is_registered_then_it_is_refused(self) -> None:
        """User Given a protagonist that already uses three identity images When a recurring character is registered Then it is refused because the first clip would exceed four reference images with the prior ending frame."""
        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-budget-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id, identity_count=3)

            with self.assertRaises(ValueError):
                self.store.register_series_companion(
                    series_id, "小明", "外觀", _write_image(root / "xiaoming.png", b"x")
                )

    def test_user_given_the_creator_picks_an_identity_image_for_a_recurring_character_when_the_review_is_recorded_then_the_character_joins_the_cast(self) -> None:
        """User Given the creator accepted a recurring character's candidate in Discord When the review is recorded Then the character is on the series cast with the picked image, and a rejected review leaves the cast unchanged and stops onboarding."""
        from agentic.app.main import _record_serial_companion_review
        from agentic.tools.context_services import HumanReviewDecision

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-review-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)
            picked = _write_image(root / "xiaoming.png", b"xiaoming")

            rejected = _record_serial_companion_review(
                self.store, series_id, "小明", "戴草帽的青年",
                HumanReviewDecision(
                    status="rejected", selected_paths=[], edited_text="",
                    reviewer="creator", review_mode="discord",
                ),
            )
            self.assertIsNotNone(rejected)
            self.assertEqual(self.store.get_series_companions(series_id), [])

            accepted = _record_serial_companion_review(
                self.store, series_id, "小明", "戴草帽的青年",
                HumanReviewDecision(
                    status="approved", selected_paths=[str(picked)], edited_text="",
                    reviewer="creator", review_mode="discord",
                ),
            )
            self.assertIsNone(accepted)
            cast = self.store.get_series_companions(series_id)
            self.assertEqual([row["name"] for row in cast], ["小明"])
            self.assertEqual(Path(cast[0]["reference_path"]), picked.resolve())

    def test_user_given_an_approval_without_a_picked_image_when_the_review_is_recorded_then_the_character_is_not_added(self) -> None:
        """User Given Discord approved a recurring character but picked no image When the review is recorded Then onboarding stops and no character joins the cast."""
        from agentic.app.main import _record_serial_companion_review
        from agentic.tools.context_services import HumanReviewDecision

        with _workspace_temporary_directory() as temporary:
            root = Path(temporary)
            series_id = f"cast-nopick-{uuid.uuid4().hex[:8]}"
            self._approve_series(root, series_id)

            blocked = _record_serial_companion_review(
                self.store, series_id, "小明", "戴草帽的青年",
                HumanReviewDecision(
                    status="approved", selected_paths=[], edited_text="",
                    reviewer="creator", review_mode="discord",
                ),
            )

            self.assertIsNotNone(blocked)
            self.assertEqual(self.store.get_series_companions(series_id), [])


if __name__ == "__main__":
    unittest.main()
