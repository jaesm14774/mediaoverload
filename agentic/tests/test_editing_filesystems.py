"""Local public renderer acceptance for real filesystem boundaries."""

import tempfile
from pathlib import Path

from PIL import Image, ImageStat
import pytest

from agentic.runtime.editing import EditClip, EditPlan, EditTransition
from agentic.tools.editing_adapter import EditRenderError, OpenCutEditAdapter
from agentic.tools.ffmpeg_adapter import FFmpegAdapter


@pytest.mark.parametrize("target_duration", [0.75, 1.5])
def test_user_given_outputs_on_another_filesystem_when_a_clip_is_trimmed_or_padded_then_the_edit_is_saved(target_duration, tmp_path):
    """User Given an output directory on a different filesystem from system temporary files When a clip needs trimming or padding Then the public renderer saves a playable edit and preserves the source."""
    other = Path("/dev/shm")
    if not other.is_dir() or other.stat().st_dev == Path(tempfile.gettempdir()).stat().st_dev:
        pytest.skip("Two distinct local filesystems are required")
    source = tmp_path / "source.png"
    Image.new("RGB", (240, 426), "blue").save(source)
    original = source.read_bytes()
    with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-acceptance-", dir=other) as directory:
        root = Path(directory)
        plan = EditPlan(clips=(EditClip(str(source), duration_seconds=1),),
            output_width=240, output_height=426, fps=12, target_duration_seconds=target_duration)
        adapter = OpenCutEditAdapter(root, input_roots=[tmp_path])
        result = adapter.render(plan, str(root / "edited.mp4"))
        probe = FFmpegAdapter().probe_media(result["video_path"])
        assert probe["has_video"] and probe["has_audio"]
        assert abs(probe["video_duration"] - target_duration) <= 1 / plan.fps
        assert source.read_bytes() == original


def color_sources(root):
    sources = []
    for index, color in enumerate(("red", "green", "blue")):
        path = root / f"source-{index}.png"
        Image.new("RGB", (240, 426), color).save(path)
        sources.append(path)
    return sources


@pytest.mark.parametrize("mode", ["hard_cut", "fade_without_audio_crossfade"])
def test_user_given_multiple_clips_when_using_cut_or_separate_audio_settings_then_the_saved_video_preserves_each_beat(mode, tmp_path):
    """User Given three colored clips When editing with implicit hard cuts or fades without audio crossfading Then every beat reaches the saved video and both media tracks span the intended timeline."""
    sources = color_sources(tmp_path)
    originals = [path.read_bytes() for path in sources]
    cuts = mode == "hard_cut"
    plan = EditPlan(
        clips=tuple(EditClip(str(path), duration_seconds=1) for path in sources),
        transitions=() if cuts else (EditTransition("fade", 0.25), EditTransition("fade", 0.25)),
        profile="motion_cut_v1" if cuts else "xfade_clean_v1", audio_crossfade=False,
        output_width=240, output_height=426, fps=12,
    )
    root = tmp_path / "output"
    result = OpenCutEditAdapter(root, input_roots=[tmp_path]).render(plan, str(root / "edited.mp4"))
    media = FFmpegAdapter().probe_media(result["video_path"])
    assert media["has_video"] and media["has_audio"]
    assert abs(media["video_duration"] - (3 if cuts else 2.5)) <= 1 / plan.fps
    assert abs(media["audio_duration"] - media["video_duration"]) <= 1 / plan.fps
    timestamps = (0.5, 1.5, 2.5) if cuts else (0.5, 1.25, 2.0)
    for channel, timestamp in enumerate(timestamps):
        frame = root / f"beat-{channel}.png"
        FFmpegAdapter().extract_frame_at(result["video_path"], str(frame), timestamp)
        with Image.open(frame) as image:
            means = ImageStat.Stat(image.convert("RGB")).mean
        assert means[channel] > max(means[(channel + 1) % 3], means[(channel + 2) % 3]) * 1.4
    assert [path.read_bytes() for path in sources] == originals


def test_user_given_an_unsupported_mixed_transition_when_rendering_then_existing_results_and_sources_are_preserved(tmp_path):
    """User Given an unsupported combination of hard cuts and fades When rendering is requested Then the renderer reports the transition error and preserves the existing output and every source."""
    sources = color_sources(tmp_path)
    originals = [path.read_bytes() for path in sources]
    root = tmp_path / "output"
    root.mkdir()
    output = root / "edited.mp4"
    output.write_bytes(b"existing result")
    plan = EditPlan(
        clips=tuple(EditClip(str(path), duration_seconds=1) for path in sources),
        transitions=(EditTransition("hard_cut", 0), EditTransition("fade", 0.25)),
        output_width=240, output_height=426, fps=12,
    )
    with pytest.raises(EditRenderError, match="hard_cut cannot be mixed"):
        OpenCutEditAdapter(root, input_roots=[tmp_path]).render(plan, str(output))
    assert output.read_bytes() == b"existing result"
    assert [path.read_bytes() for path in sources] == originals
