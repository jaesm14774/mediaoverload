from __future__ import annotations

import importlib.util
import json
import os
import shutil
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cleanup_media.py"
spec = importlib.util.spec_from_file_location("cleanup_media_script", SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)

class PublishedOutputCleanupTests(unittest.TestCase):
    # Files are created now, so a publish that already happened must be in the past
    # for their ctime to precede it, and in the future for the opposite.
    published_at = datetime.now(timezone.utc) + timedelta(hours=1)

    def make_media_root(self) -> Path:
        base_dir = Path(__file__).resolve().parents[2] / ".tmp-tests"
        base_dir.mkdir(parents=True, exist_ok=True)
        root = base_dir / f"cleanup-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root.resolve()

    def write_media(self, path: Path, *, hours_from_publish: float) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"media")
        stamp = self.published_at.timestamp() + hours_from_publish * 3600
        os.utime(path, (stamp, stamp))
        return path

    def write_manifest(self, folder: Path, state: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "publish_manifest.json").write_text(
            json.dumps({"publication_state": state, "created_at": self.published_at.isoformat()}),
            encoding="utf-8",
        )

    def test_user_sees_media_published_before_the_publish_removed_and_newer_runs_kept(self) -> None:
        # Given a character folder whose latest publish is fully public
        root = self.make_media_root()
        folder = root / "kirby"
        generated = self.write_media(folder / "run_a" / "video.mp4", hours_from_publish=-3)
        delivered = self.write_media(folder / "publish_ready" / "post.jpg", hours_from_publish=-1)
        next_run = self.write_media(folder / "run_b" / "video.mp4", hours_from_publish=2)
        notes = folder / "run_a" / "story.txt"
        notes.write_text("keep", encoding="utf-8")
        self.write_manifest(folder, "published")

        # When the user previews the cleanup
        groups, preserved = cleanup._published_output_media(root)

        # Then only media written up to the publish is listed, never text or newer runs
        listed = set().union(*groups.values())
        self.assertEqual({generated.resolve(), delivered.resolve()}, listed)
        self.assertNotIn(next_run.resolve(), listed)
        self.assertNotIn(notes.resolve(), listed)
        self.assertEqual({}, preserved)

    def test_user_keeps_a_fresh_copy_of_an_old_asset(self) -> None:
        # Given a publish that happened an hour ago and a new run that copied an old asset
        # (copying preserves the old modified time but the file itself is new)
        self.published_at = datetime.now(timezone.utc) - timedelta(hours=1)
        root = self.make_media_root()
        folder = root / "kirby"
        fresh_copy = self.write_media(folder / "run_b" / "video.mp4", hours_from_publish=-48)
        self.write_manifest(folder, "published")

        # When the user previews the cleanup
        groups, _ = cleanup._published_output_media(root)

        # Then the new run's file is not listed
        self.assertNotIn(fresh_copy.resolve(), set().union(*groups.values()))

    def test_user_cannot_point_the_cleanup_inside_a_serial_folder(self) -> None:
        # Given a media root that sits inside the serial-owned folder
        # When the user passes it as the output directory
        # Then the cleanup refuses it
        with self.assertRaises(ValueError):
            cleanup._media_root(cleanup.PROJECT_ROOT / "output" / "serial_series")

    def test_user_keeps_media_when_publish_is_not_fully_public(self) -> None:
        # Given a folder whose latest publish still waits on the user
        root = self.make_media_root()
        folder = root / "metaknight"
        self.write_media(folder / "run_a" / "image.png", hours_from_publish=-1)
        self.write_manifest(folder, "awaiting_user_action")

        # When the user previews the cleanup
        groups, preserved = cleanup._published_output_media(root)

        # Then nothing is listed and the folder is reported as preserved
        self.assertEqual({}, groups)
        self.assertEqual(1, len(preserved))

    def test_user_serial_folders_are_left_to_the_series_cleanup(self) -> None:
        # Given a published manifest inside a serial-owned folder
        root = self.make_media_root()
        folder = root / "serial_publication_retries"
        self.write_media(folder / "attempt_1" / "video.mp4", hours_from_publish=-1)
        self.write_manifest(folder, "published")

        # When the user previews the cleanup
        groups, preserved = cleanup._published_output_media(root)

        # Then the serial folder is neither listed nor reported
        self.assertEqual({}, groups)
        self.assertEqual({}, preserved)


if __name__ == "__main__":
    unittest.main()
