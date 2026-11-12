"""Review uploads must fit the channel's upload limit without changing what gets published."""

import shutil
import subprocess
from pathlib import Path

import pytest

from agentic.tools.context_services import fit_review_uploads

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)


def _video(path: Path, *, duration: int) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=320x192:rate=24:duration={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_user_given_a_final_video_over_the_upload_limit_when_review_is_posted_then_a_smaller_copy_is_uploaded_and_the_original_is_kept(tmp_path):
    """User Given the final video is larger than the channel upload limit When the review is posted Then the reviewer gets a copy under the limit and the original file is unchanged."""
    video = _video(tmp_path / "final.mp4", duration=6)
    document = tmp_path / "state.md"
    document.write_text("完整狀態", encoding="utf-8")
    original_size = video.stat().st_size
    limit = original_size // 2

    uploads = fit_review_uploads(
        [str(video)], [str(document)], limit_bytes=limit, work_dir=tmp_path / "preview"
    )

    assert sum(Path(path).stat().st_size for path in uploads.media_paths + uploads.document_paths) <= limit
    assert len(uploads.media_paths) == 1
    assert video.stat().st_size == original_size


def test_user_given_a_video_within_the_upload_limit_when_review_is_posted_then_it_is_uploaded_as_is(tmp_path):
    """User Given the final video fits the upload limit When the review is posted Then the original file is what the reviewer watches."""
    video = _video(tmp_path / "final.mp4", duration=2)

    uploads = fit_review_uploads(
        [str(video)], [], limit_bytes=video.stat().st_size * 2, work_dir=tmp_path / "preview"
    )

    assert uploads.media_paths == [str(video)]


def test_user_given_review_files_that_cannot_fit_the_limit_when_review_is_posted_then_it_fails_explicitly(tmp_path):
    """User Given files that cannot be shrunk under the limit When the review is posted Then the failure is explicit instead of a silent oversize upload."""
    video = _video(tmp_path / "final.mp4", duration=6)

    with pytest.raises(ValueError, match="upload limit"):
        fit_review_uploads([str(video)], [], limit_bytes=500, work_dir=tmp_path / "preview")


def test_user_given_a_large_image_and_a_large_video_when_review_is_posted_then_only_the_video_is_shrunk(tmp_path):
    """User Given a review with a big image and an oversized video When the review is posted Then the image is untouched and the video shrinks into the remaining room."""
    video = _video(tmp_path / "final.mp4", duration=6)
    limit = video.stat().st_size // 2
    image = tmp_path / "cover.png"
    image.write_bytes(b"\0" * int(limit * 0.6))

    uploads = fit_review_uploads(
        [str(image), str(video)], [], limit_bytes=limit, work_dir=tmp_path / "preview"
    )

    assert uploads.media_paths[0] == str(image)
    assert sum(Path(path).stat().st_size for path in uploads.media_paths) <= limit
