from __future__ import annotations

import subprocess
from pathlib import Path

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


def test_user_given_video_with_unicode_metadata_when_probed_then_video_stream_is_detected(
    tmp_path: Path,
) -> None:
    """User Given an MP4 with UTF-8 metadata When probed Then its video stream is detected."""

    video_path = tmp_path / "unicode_metadata.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=2:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-metadata",
            "comment=Sprite prompt: 8\u2011second clip",
            "-y",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    probe = FFmpegAdapter().probe_media(str(video_path))

    assert probe["has_video"] is True
    assert probe["width"] == 64
