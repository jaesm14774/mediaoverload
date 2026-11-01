from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agentic.tools.comfy_nvidia_vfx import (
    ComfyNvidiaVfxToolset,
    _assemble_chunks,
    _probe_video,
)


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)


def _decoded_rgb_pixel(video: Path, timestamp: float) -> tuple[int, int, int]:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1:flags=area",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return tuple(completed.stdout)


def test_user_given_overlapping_vfx_chunks_and_aac_source_when_assembling_then_frame_contract_and_audio_are_preserved(
    tmp_path: Path,
):
    """User Given ordered red and blue overlapping 48 FPS chunks and AAC source When VFX assembly joins them Then frame order, media contract, full decode, and source audio are preserved."""
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=160x96:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            "48",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    chunks = [tmp_path / "chunk-01.mp4", tmp_path / "chunk-02.mp4"]
    for path, color, frame_count in zip(chunks, ("red", "blue"), (49, 48), strict=True):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:size=160x96:rate=48:duration=2",
                "-frames:v",
                str(frame_count),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )

    output = tmp_path / "assembled.mp4"
    source_info = _probe_video(source)
    _assemble_chunks(
        chunks,
        source,
        output,
        source_info=source_info,
        output_fps=48,
        expected_frames=96,
    )

    result = _probe_video(output)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "NUL",
        ],
        check=True,
        capture_output=True,
    )

    def decoded_audio_hash(path: Path) -> str:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    assert result["width"] == 160
    assert result["height"] == 96
    assert result["fps"] == pytest.approx(48)
    assert result["frame_count"] == 96
    assert result["duration_seconds"] == pytest.approx(2)
    assert result["video_codec"] == "h264"
    assert result["pixel_format"] == "yuv420p"
    assert result["audio_codec"] == "aac"
    assert result["audio_channels"] == 2
    assert decoded_audio_hash(output) == decoded_audio_hash(source)
    first_color = _decoded_rgb_pixel(output, 0.5)
    second_color = _decoded_rgb_pixel(output, 1.5)
    assert first_color[0] > first_color[2] * 2
    assert second_color[2] > second_color[0] * 2


def test_user_given_decodable_av_video_when_nvidia_vfx_probes_it_then_full_decode_reports_exact_frames(
    tmp_path: Path,
):
    """User Given a decodable MP4 with audio When NVIDIA VFX probes it Then one full decode reports the exact video frame count and stream metadata."""
    source = tmp_path / "source.mp4"
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
            "sine=frequency=440:sample_rate=32000:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            "48",
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

    info = _probe_video(source)

    assert info["frame_count"] == 48
    assert info["audio_codec"] == "aac"
    assert info["audio_channels"] == 1


def test_user_given_video_without_audio_when_nvidia_vfx_probes_it_then_video_frames_are_still_counted(
    tmp_path: Path,
):
    """User Given a valid video-only MP4 When NVIDIA VFX counts frames Then full video decoding succeeds without requiring an audio stream."""
    source = tmp_path / "silent.mp4"
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
            "-frames:v",
            "48",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    info = _probe_video(source)

    assert info["frame_count"] == 48
    assert info["audio_stream_count"] == 0


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


def test_user_given_an_existing_vfx_output_when_postprocessing_starts_then_the_existing_file_is_preserved(
    tmp_path: Path,
):
    """User Given a valid source and an existing output file When NVIDIA VFX post-processing starts Then the existing output remains byte-for-byte unchanged."""
    source = tmp_path / "source.mp4"
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
            "color=c=green:size=160x96:rate=24:duration=1",
            "-frames:v",
            "24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    existing_content = b"keep the previously delivered output"
    output.write_bytes(existing_content)

    tool = ComfyNvidiaVfxToolset(comfy_host="127.0.0.1", comfy_port=1)
    with pytest.raises(FileExistsError, match="output already exists"):
        tool.execute({"video_path": str(source), "output_path": str(output)})

    assert output.read_bytes() == existing_content
