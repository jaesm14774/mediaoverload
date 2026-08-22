from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scheduler.scheduler import (
    REPO_ROOT,
    _run_scheduled_job_safe,
    load_scheduler_config,
    run_scheduled_job,
)


class SchedulerTests(unittest.TestCase):
    def test_load_scheduler_config_resolves_character_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SCHEDULER_CHARACTER": "kirby",
                "SCHEDULER_MODE": "interval",
                "SCHEDULER_INTERVAL_HOURS": "6",
            },
            clear=False,
        ):
            config = load_scheduler_config()

        self.assertEqual(config.character, "kirby")
        self.assertEqual(config.interval_hours, 6)
        self.assertEqual(config.config_path, REPO_ROOT / "configs" / "characters" / "kirby.yaml")

    def test_run_scheduled_job_delegates_to_character_workflow(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SCHEDULER_CHARACTER": "kirby",
                "SCHEDULER_PROMPT": "Kirby neon short",
                "SCHEDULER_NEWS_DRIVEN": "true",
                "SCHEDULER_NEWS_HISTORY_PATH": "/tmp/kirby-news.json",
                "SCHEDULER_DRY_RUN_PUBLISH": "true",
                "SCHEDULER_ENABLE_REVIEW_LOOP": "false",
            },
            clear=False,
        ):
            config = load_scheduler_config()

        with patch(
            "scheduler.scheduler.run_character_workflow",
            return_value={"status": "success", "source_generation_type": "text2image2video"},
        ) as workflow_mock:
            result = run_scheduled_job(config)

        self.assertEqual(result["status"], "success")
        workflow_mock.assert_called_once()
        request = workflow_mock.call_args.args[0]
        self.assertEqual(request.repo_root, REPO_ROOT)
        self.assertEqual(request.config_path, Path(config.config_path))
        self.assertEqual(request.generation.prompt, "Kirby neon short")
        self.assertTrue(request.generation.news_driven)
        self.assertEqual(request.generation.news_history_path, "/tmp/kirby-news.json")
        self.assertTrue(request.review.dry_run_publish)
        self.assertFalse(request.review.enable_review_loop)
        self.assertIsNotNone(request.generation.rng)

    def test_scheduled_job_is_skipped_during_quiet_hours(self) -> None:
        config = self._load_kirby_config()

        with (
            patch("scheduler.scheduler.time.localtime", return_value=SimpleNamespace(tm_hour=2)),
            patch("scheduler.scheduler.run_scheduled_job") as job_mock,
        ):
            result = _run_scheduled_job_safe(config)

        self.assertEqual(result, {"status": "skipped", "reason": "quiet_hours"})
        job_mock.assert_not_called()

    def test_scheduled_job_runs_at_quiet_hours_end(self) -> None:
        config = self._load_kirby_config()

        with (
            patch("scheduler.scheduler.time.localtime", return_value=SimpleNamespace(tm_hour=6)),
            patch(
                "scheduler.scheduler.run_scheduled_job",
                return_value={"status": "success"},
            ) as job_mock,
        ):
            result = _run_scheduled_job_safe(config)

        self.assertEqual(result, {"status": "success"})
        job_mock.assert_called_once_with(config, rng=None)

    @staticmethod
    def _load_kirby_config():
        with patch.dict(
            os.environ,
            {
                "SCHEDULER_CHARACTER": "kirby",
                "SCHEDULER_MODE": "interval",
            },
            clear=False,
        ):
            return load_scheduler_config()


if __name__ == "__main__":
    unittest.main()
