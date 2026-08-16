from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scheduler.scheduler import REPO_ROOT, load_scheduler_config, run_scheduled_job


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
        args, kwargs = workflow_mock.call_args
        self.assertEqual(args[0], REPO_ROOT)
        self.assertEqual(args[1], Path(config.config_path))
        self.assertEqual(kwargs["prompt"], "Kirby neon short")
        self.assertTrue(kwargs["news_driven"])
        self.assertEqual(kwargs["news_history_path"], "/tmp/kirby-news.json")
        self.assertTrue(kwargs["dry_run_publish"])
        self.assertFalse(kwargs["enable_review_loop"])
        self.assertIsNotNone(kwargs["rng"])


if __name__ == "__main__":
    unittest.main()
