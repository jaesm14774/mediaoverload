from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_vox_director_news_arc_ab import (
    _final_video_probe_pass,
    _load_cases,
    _validate_resume_manifest_hash,
)


class VoxDirectorNewsArcRunnerTests(unittest.TestCase):
    def test_user_given_news_manifest_when_case_id_is_a_path_then_reject_before_writing(self) -> None:
        """User Given a news benchmark manifest When a case id is a path Then the runner rejects it before writing files."""
        repo_root = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (repo_root / "docs" / "research" / "vox_director_news_arc_cases.json").read_text(
                encoding="utf-8"
            )
        )
        manifest["cases"][0]["case_id"] = "../outside"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unsafe-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe relative path component"):
                _load_cases(data_path=path)

    def test_user_given_final_video_when_probe_has_no_decodable_video_then_reject_technical_pass(self) -> None:
        """User Given a final-video artifact When its probe lacks a valid video stream Then it cannot pass technical QA."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "final.mp4"
            path.write_bytes(b"test artifact")

            self.assertFalse(
                _final_video_probe_pass(
                    path,
                    {"duration_seconds": 12.0, "streams": [{"codec_type": "audio"}]},
                )
            )
            self.assertTrue(
                _final_video_probe_pass(
                    path,
                    {
                        "duration_seconds": 12.0,
                        "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
                    },
                )
            )

    def test_user_given_resumed_run_when_its_manifest_was_changed_then_reject_resume(self) -> None:
        """User Given a saved benchmark run When its article manifest changed Then resume rejects mismatched evidence controls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "news_cases.json"
            path.write_text('{"cases": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash does not match"):
                _validate_resume_manifest_hash(path, "unexpected-hash")


if __name__ == "__main__":
    unittest.main()
