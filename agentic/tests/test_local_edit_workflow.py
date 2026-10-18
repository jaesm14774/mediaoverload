"""Local end-to-end acceptance: real planner, skills, FFmpeg and media QA."""

from pathlib import Path
import shutil

import pytest
from PIL import Image

from agentic.app.main import build_runtime
from agentic.runtime.contracts import GoalRequest
from agentic.tools.ffmpeg_adapter import FFmpegAdapter


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)
def test_user_given_an_image_when_editing_through_the_runtime_then_a_playable_qa_checked_video_is_delivered(
    tmp_path,
):
    """User Given a real image When its edit goal runs end to end Then a playable video preserves the requested canvas and duration and passes media QA."""
    image_path = tmp_path / "source.png"
    Image.new("RGB", (320, 320), "green").save(image_path)
    planner, runner, _ = build_runtime(
        Path(__file__).resolve().parents[2],
        output_root=tmp_path / "output",
        input_roots=[tmp_path],
    )
    # Avoid adding test examples to the application's persistent portfolio.
    runner.portfolio_memory = None
    goal = GoalRequest(
        prompt="Pan across the source image",
        media_type="image_sequence_edit",
        duration_seconds=2,
        constraints={
            "edit_input_paths": [str(image_path)],
            "edit_width": 320,
            "edit_height": 320,
            "edit_fps": 24,
            "edit_profile": "motion_cut_v1",
        },
    )
    result = runner.run(planner.build_plan(goal))
    assert result.status == "success", result.to_dict()
    video_path = Path(result.state.node_outputs["compose-edit"]["video_path"])
    assert video_path.is_file()
    media = FFmpegAdapter().probe_media(str(video_path))
    assert media["width"] == 320 and media["height"] == 320
    assert abs(media["duration"] - 10) < 0.35
    assert all(record.status == "success" for record in result.records)
