from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agentic.tools.comfy_nvidia_vfx import ComfyNvidiaVfxToolset


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)


def test_user_given_material_audio_video_drift_when_nvidia_vfx_starts_then_it_is_rejected_before_comfyui(
    tmp_path: Path,
):
    """User Given a real video whose audio extends beyond one source frame When NVIDIA VFX starts Then it rejects the source before creating output or contacting ComfyUI."""
    source = tmp_path / "drifted.mp4"
    output = tmp_path / "result.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:size=160x96:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=2.5",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    tool = ComfyNvidiaVfxToolset(comfy_host="127.0.0.1", comfy_port=1)
    with pytest.raises(
        ValueError,
        match="Source audio and video durations differ by more than one source frame",
    ):
        tool.execute({"video_path": str(source), "output_path": str(output)})

    assert not output.exists()
