"""Real FFmpeg delivery boundary; GPU generation is outside these scenarios."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from agentic.tools.comfy_h3 import deliver_video


def native_video(tmp_path: Path, native_frames: int, audio_channels: int = 2) -> Path:
    width, height = 64, 40
    frame_size = width * height
    frames = b"".join(
        bytes((255 - round(index * 255 / (native_frames - 1)), 0, round(index * 255 / (native_frames - 1)))) * frame_size
        for index in range(native_frames)
    )
    source = tmp_path / "native.mp4"
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", "24", "-i", "pipe:0",
    ]
    if audio_channels:
        layout = "stereo" if audio_channels == 2 else "mono"
        command += ["-f", "lavfi", "-i", f"anullsrc=sample_rate=48000:channel_layout={layout}"]
    command += [
        "-frames:v", str(native_frames), "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "0", "-c:a", "aac", "-shortest", str(source),
    ]
    subprocess.run(command, input=frames, check=True, capture_output=True)
    return source


@pytest.mark.parametrize("native_frames,delivered_frames,width,height", [
    (124, 120, 80, 48), (362, 360, 80, 48), (107, 1, 80, 48),
    (120, 120, 80, 48), (124, 124, 64, 40), (107, 24, 80, 48),
])
def test_user_given_native_h3_video_when_delivered_then_timeline_anchors_and_audio_survive(
    tmp_path: Path, native_frames: int, delivered_frames: int, width: int, height: int,
):
    """User Given native H3 video with distinct endpoint anchors When delivered Then its full timeline, requested frames, canvas, duration and stereo audio are preserved."""
    source, target = native_video(tmp_path, native_frames), tmp_path / "delivered.mp4"
    requested = {"width": width, "height": height, "length": delivered_frames, "fps": 24}
    actual = deliver_video(source, target, requested)
    assert {key: actual[key] for key in requested} == requested
    assert actual["duration_seconds"] == pytest.approx(delivered_frames / 24, abs=1 / 24)
    assert actual["audio"]["channels"] == 2
    decoded = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(target), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    for fraction in [0, 0.25, 0.5, 0.75, 1]:
        index = round(fraction * (delivered_frames - 1))
        offset = (index * width * height + (height // 2) * width + width // 2) * 3
        red, _green, blue = decoded[offset:offset + 3]
        expected_blue = 255 * index / (delivered_frames - 1) if delivered_frames > 1 else 255
        # Allow codec rounding while checking that every part of the timeline survives.
        assert blue == pytest.approx(expected_blue, abs=16)
        assert red == pytest.approx(255 - expected_blue, abs=16)


@pytest.mark.parametrize("audio_channels,requested_frames", [(0, 124), (2, 125)])
def test_user_given_unusable_native_media_when_delivered_then_incomplete_output_is_rejected(tmp_path, audio_channels, requested_frames):
    """User Given native media without audio or enough frames When delivered Then the unusable clip is rejected before publication."""
    source = native_video(tmp_path, 124, audio_channels)
    with pytest.raises(RuntimeError, match="insufficient frames or missing native audio"):
        deliver_video(source, tmp_path / "rejected.mp4", {"width": 64, "height": 40, "length": requested_frames, "fps": 24})


def test_user_given_mono_native_audio_when_delivered_then_the_stereo_contract_is_enforced(tmp_path):
    """User Given a native clip with mono audio When delivered Then it cannot pass the required stereo delivery contract."""
    source = native_video(tmp_path, 124, 1)
    with pytest.raises(RuntimeError, match="delivery contract failed"):
        deliver_video(source, tmp_path / "rejected.mp4", {"width": 64, "height": 40, "length": 124, "fps": 24})
