from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic.tools.comfy_backend import ComfyUploadResult
from agentic.tools.comfy_nvidia_vfx import ComfyNvidiaVfxToolset


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)


class _FfmpegComfyGenerator:
    """Local Comfy generator substitute that returns real, decodable MP4 chunks."""

    def __init__(self) -> None:
        self.communicator = SimpleNamespace(ws=None)

    @staticmethod
    def get_object_info(node_type: str) -> dict[str, dict[str, object]]:
        return {node_type: {"input": {}, "output": []}}

    @staticmethod
    def upload_image(image_path: str) -> ComfyUploadResult:
        return ComfyUploadResult(Path(image_path).name, None)

    @staticmethod
    def generate(
        *,
        workflow_path: str,
        updates: list[dict[str, object]],
        output_dir: str,
        file_prefix: str,
        workflow: dict[str, object] | None = None,
        timing_sink: list[dict[str, object]] | None = None,
    ) -> list[str]:
        del workflow_path, updates, timing_sink
        assert workflow is not None
        inputs = workflow["1"]["inputs"]
        frame_count = int(inputs["frame_load_cap"])
        expected_output_frames = (frame_count - 1) * 2 + 1 if frame_count > 1 else 1
        output_fps = float(workflow["4"]["inputs"]["fps"])
        # Like the real NVIDIA node, enlarge the 64x64 source by the requested scale.
        side = int(64 * float(workflow["2"]["inputs"]["resize_type.scale"]))
        output = Path(output_dir) / f"{file_prefix}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=blue:size={side}x{side}:rate=48:duration=16",
                "-vf",
                f"setpts=N/({output_fps:.12g}*TB)",
                "-vsync",
                "0",
                "-frames:v",
                str(expected_output_frames),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-video_track_timescale",
                "12288",
                str(output),
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        return [str(output)]


class _FfmpegComfyAdapter:
    def build_generator(self, host: str | None = None, port: int | None = None) -> _FfmpegComfyGenerator:
        del host, port
        return _FfmpegComfyGenerator()


def test_user_given_a_24fps_nominal_source_with_fractional_average_rate_when_nvidia_vfx_processes_it_then_exact_double_frames_and_source_duration_are_accepted(
    tmp_path: Path,
) -> None:
    """User Given a valid video whose 24 FPS nominal rate differs from its fractional average When NVIDIA VFX processes it Then the exact doubled frame count and preserved duration are accepted."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "result.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:size=64x64:rate=24:duration=16",
            "-vf",
            "setpts='N/(24*TB)+if(eq(N,365),1/(24*TB),0)'",
            "-vsync",
            "0",
            "-frames:v",
            "366",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "12288",
            str(source),
        ],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    result = ComfyNvidiaVfxToolset(adapter=_FfmpegComfyAdapter()).execute(
        {"video_path": str(source), "output_path": str(output)}
    )

    assert output.is_file()
    assert result["result"]["frame_count"] == 2 * result["source"]["frame_count"]
    assert result["result"]["width"] == 3 * result["source"]["width"]
    assert result["result"]["height"] == 3 * result["source"]["height"]
    duration_tolerance = 1 / (result["source"]["fps"] * 2) + result["result"]["time_base_seconds"]
    assert abs(result["result"]["duration_seconds"] - result["source"]["duration_seconds"]) <= duration_tolerance
