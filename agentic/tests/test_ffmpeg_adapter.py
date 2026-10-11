"""Lower-level media boundary: real FFmpeg output, no subprocess replacements.

These specific stream and image contracts complement the complete runtime edit
acceptance; each must inspect the adapter's actual output.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and ffprobe required",
)


@pytest.fixture
def source_factory(tmp_path):
    def create(*, audio=True, changing=False, variable_timing=False):
        target = tmp_path / ("with_audio.mp4" if audio else "silent.mp4")
        source_filter = (
            "testsrc2=size=160x96:rate=24:duration=2" if changing else
            "color=c=green:size=160x96:rate=24:duration=2,drawbox=x=10:y=10:w=20:h=20:color=red:t=fill:enable='lt(t,1)'"
        )
        if variable_timing:
            source_filter += ",setpts='if(lt(N,12),N/(24*TB),(0.5+(N-12)/12)/TB)'"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source_filter,
        ]
        if audio:
            command.extend(
                ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=2"]
            )
        command.extend(
            ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        )
        if audio:
            command.extend(["-c:a", "aac", "-ac", "2"])
        if variable_timing:
            command.extend(["-vsync", "vfr"])
        command.append(str(target))
        subprocess.run(command, check=True, capture_output=True)
        return target

    return create


@pytest.mark.parametrize("variable_timing", [False, True])
def test_user_given_fixed_sample_times_when_extracting_a_batch_then_pixels_match_individual_samples(
    source_factory, tmp_path, variable_timing,
):
    """User Given fixed video and timestamps When extracting a batch Then each frame matches the individual sample pixels."""
    source = source_factory(audio=False, changing=True, variable_timing=variable_timing)
    adapter = FFmpegAdapter()
    timestamps = [0.0, 0.137, 0.137, 0.75, 1.001, 1.875]
    outputs = [str(tmp_path / f"batch-{index}.png") for index in range(len(timestamps))]
    result = adapter.extract_frames_at(str(source), outputs, timestamps)
    assert result == outputs
    for index, timestamp in enumerate(timestamps):
        individual = tmp_path / f"individual-{index}.png"
        adapter.extract_frame_at(str(source), str(individual), timestamp)
        with Image.open(individual) as expected, Image.open(outputs[index]) as actual:
            assert actual.size == expected.size
            assert actual.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()


@pytest.mark.parametrize("timestamps", [[-1], [float("nan")], [float("inf")]])
def test_user_given_invalid_sample_times_when_extracting_then_no_output_is_created(tmp_path, timestamps):
    """User Given an invalid timestamp When extracting Then the request fails without creating output."""
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="finite.*non-negative"):
        FFmpegAdapter().extract_frames_at("unused.mp4", [str(output)], timestamps)
    assert not output.exists()


def test_user_given_a_sample_beyond_the_clip_when_extracting_then_missing_frames_are_reported(
    source_factory, tmp_path,
):
    """User Given a timestamp beyond the clip When extracting Then missing media fails explicitly instead of returning a stale frame."""
    output = tmp_path / "existing.png"
    Image.new("RGB", (4, 4), "red").save(output)
    previous = output.read_bytes()
    with pytest.raises(RuntimeError, match="frame"):
        FFmpegAdapter().extract_frames_at(str(source_factory(audio=False)), [str(output)], [3.0])
    assert output.read_bytes() == previous


@pytest.mark.parametrize("outputs,timestamps", [([], []), (["a.png"], [0, 1]), (["a.png", "a.png"], [0, 1]), (["unused.mp4"], [0])])
def test_user_given_an_ambiguous_batch_when_extracting_then_the_request_is_rejected(outputs, timestamps):
    """User Given missing, duplicate or source-overwriting destinations When extracting Then the ambiguous batch is rejected."""
    with pytest.raises(ValueError):
        FFmpegAdapter().extract_frames_at("unused.mp4", outputs, timestamps)


def test_user_given_a_fixed_motion_clip_when_packaging_then_the_atlas_and_animation_keep_the_sample_pixels(
    source_factory, tmp_path,
):
    """User Given fixed motion media When the sprite tool packages it Then all sixteen frames and exported atlas retain the individual-sample pixels."""
    import json
    from agentic.runtime.registry import ToolRegistry
    from agentic.tools.media_services import register_media_service_tools
    from agentic.tools.sprite_adapter import SpriteAdapter

    source = source_factory(audio=False, changing=True)
    tools = ToolRegistry()
    register_media_service_tools(tools, tmp_path)
    result = tools.call("media.video_to_sprite", {
        "video_path": str(source), "output_dir": str(tmp_path / "batch_sprite"),
        "cell_width": 64, "cell_height": 64, "fps": 12,
    })
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    samples = []
    for index, timestamp in enumerate(manifest["sampling"]["timestamps_seconds"]):
        path = tmp_path / f"reference-{index}.png"
        FFmpegAdapter().extract_frame_at(str(source), str(path), timestamp)
        samples.append(str(path))
    reference = SpriteAdapter().build_from_frames(samples, output_dir=str(tmp_path / "reference_sprite"))
    assert len(result["frame_paths"]) == 16
    assert all(result["qa"]["checks"].values())
    for key in ("atlas_path", "gif_path", "webp_path"):
        assert Path(result[key]).read_bytes() == Path(reference[key]).read_bytes()


@pytest.mark.parametrize("copies", [1, 2])
def test_user_given_video_clips_when_concatenating_then_a_playable_copy_or_sequence_keeps_all_streams(
    source_factory,
    tmp_path,
    copies,
):
    """User Given one or multiple video clips When concatenating Then a playable result retains the complete duration and audio."""
    source = source_factory()
    output = tmp_path / "joined.mp4"
    adapter = FFmpegAdapter()
    adapter.concat_videos([str(source)] * copies, str(output))
    media = adapter.probe_media(str(output))
    assert media["has_video"] and media["has_audio"]
    assert abs(media["duration"] - 2 * copies) < 0.15
    if copies == 1:
        assert source.read_bytes() == output.read_bytes()


def test_user_given_a_changing_video_when_creating_a_preview_then_the_gif_is_decodable_and_animated(
    source_factory,
    tmp_path,
):
    """User Given a video with visible changes When creating a GIF preview Then its frames decode at the requested width and preserve the changes."""
    source = source_factory(audio=False)
    output = tmp_path / "preview.gif"
    FFmpegAdapter().video_to_gif(
        str(source), str(output), fps=10, max_colors=128, scale_width=160
    )
    with Image.open(output) as preview:
        assert preview.width == 160
        assert preview.n_frames > 1
        first = preview.convert("RGB").tobytes()
        preview.seek(preview.n_frames - 1)
        assert first != preview.convert("RGB").tobytes()


@pytest.mark.parametrize("speed,audio", [(0.5, True), (2.0, True), (2.0, False)])
def test_user_given_a_video_when_changing_playback_speed_then_duration_and_optional_audio_remain_synchronized(
    source_factory,
    tmp_path,
    speed,
    audio,
):
    """User Given a video with or without audio When changing playback speed Then its duration changes accordingly and any audio stays synchronized."""
    source = source_factory(audio=audio)
    output = tmp_path / "speed.mp4"
    adapter = FFmpegAdapter()
    adapter.change_video_speed(str(source), str(output), speed)
    media = adapter.probe_media(str(output))
    assert abs(media["duration"] - 2 / speed) < 0.2
    assert media["frame_rate"] == pytest.approx(24)
    assert media["has_audio"] is audio
    if audio:
        assert abs(media["audio_duration"] - media["video_duration"]) < 0.2


def test_user_given_a_landscape_video_when_fitting_a_canvas_then_the_picture_is_uncropped_and_audio_is_preserved(
    source_factory,
    tmp_path,
):
    """User Given a landscape video When fitting it on a square canvas Then the complete picture keeps its aspect ratio inside padding and retains audio."""
    source = source_factory()
    output = tmp_path / "square.mp4"
    adapter = FFmpegAdapter()
    adapter.normalize_video_canvas(
        str(source),
        str(output),
        target_width=320,
        target_height=320,
        background="#000000",
    )
    media = adapter.probe_media(str(output))
    assert (media["width"], media["height"]) == (320, 320)
    assert media["has_audio"]
    assert media["frame_rate"] == pytest.approx(24)
    frame = tmp_path / "square.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(output),
            "-frames:v",
            "1",
            str(frame),
        ],
        check=True,
        capture_output=True,
    )
    with Image.open(frame) as image:
        bounds = image.convert("RGB").getbbox()
        assert bounds is not None
        assert abs((bounds[2] - bounds[0]) / (bounds[3] - bounds[1]) - 160 / 96) < 0.05


def test_user_given_a_video_when_preparing_a_platform_canvas_then_the_delivered_streams_match_the_platform_contract(
    source_factory,
    tmp_path,
):
    """User Given a video with audio When preparing a vertical platform canvas Then output has the requested size, 30 FPS, and stereo 48 kHz audio."""
    source = source_factory()
    output = tmp_path / "vertical.mp4"
    adapter = FFmpegAdapter()
    adapter.pad_video_to_aspect(
        str(source), str(output), target_width=180, target_height=320
    )
    media = adapter.probe_media(str(output))
    assert (media["width"], media["height"]) == (180, 320)
    assert media["frame_rate"] == pytest.approx(30)
    assert media["channels"] == 2 and media["sample_rate"] == 48000


@pytest.mark.parametrize(
    "audio,normalize", [(False, False), (True, False), (True, True)]
)
def test_user_given_a_video_when_trimming_then_duration_and_optional_audio_match_the_requested_delivery(
    source_factory,
    tmp_path,
    audio,
    normalize,
):
    """User Given a video with optional audio When trimming it for delivery Then its requested duration and audio presence are retained and requested normalization uses 48 kHz."""
    source = source_factory(audio=audio)
    output = tmp_path / "trimmed.mp4"
    adapter = FFmpegAdapter()
    adapter.trim_video(str(source), str(output), 1, normalize_audio=normalize)
    media = adapter.probe_media(str(output))
    assert abs(media["duration"] - 1) < 0.1
    assert media["has_audio"] is audio
    if normalize:
        assert media["sample_rate"] == 48000


def test_user_given_an_invalid_canvas_background_when_normalizing_then_the_request_is_rejected_before_output(
    source_factory,
    tmp_path,
):
    """User Given an invalid filter expression as a background When normalizing the canvas Then the request fails without creating an output."""
    source = source_factory()
    output = tmp_path / "invalid.mp4"
    with pytest.raises(ValueError, match="background"):
        FFmpegAdapter().normalize_video_canvas(
            str(source),
            str(output),
            target_width=320,
            target_height=320,
            background="red,format=yuv444p",
        )
    assert not output.exists()
