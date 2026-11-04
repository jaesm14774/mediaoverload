"""Local public renderer acceptance for real filesystem boundaries."""

import hashlib
import json
import subprocess
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


def test_user_given_an_input_changes_after_its_snapshot_when_staging_then_the_changed_copy_is_rejected_and_removed(tmp_path):
    """User Given an approved input whose contents change after its initial hash When the file is staged Then the mismatch is rejected and no staged copy remains."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"approved source")
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"changed source")
    destination = tmp_path / "staged" / "source.mp4"
    adapter = OpenCutEditAdapter(tmp_path / "output", input_roots=[tmp_path])

    with pytest.raises(EditRenderError):
        adapter._stage_input(
            source,
            expected_sha256=expected_sha256,
            destination=destination,
        )

    assert not destination.exists()


def color_sources(root):
    sources = []
    for index, color in enumerate(("red", "green", "blue")):
        path = root / f"source-{index}.png"
        Image.new("RGB", (240, 426), color).save(path)
        sources.append(path)
    return sources


def test_user_given_a_video_with_a_pan_request_when_the_edit_is_rendered_then_the_motion_reaches_the_output(tmp_path):
    """User Given a video made from a still image with distinct left and right colors When a pan edit is rendered Then the saved output moves from the left color toward the right and keeps its media contract."""
    still = tmp_path / "split-colors.png"
    source = tmp_path / "static-source.mp4"
    image = Image.new("RGB", (320, 240), "red")
    image.paste(Image.new("RGB", (160, 240), "blue"), (160, 0))
    image.save(still)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-loop", "1",
            "-framerate", "12", "-i", str(still), "-t", "1.5", "-c:v",
            "libx264", "-pix_fmt", "yuv420p", "-y", str(source),
        ],
        check=True,
        capture_output=True,
    )
    original = source.read_bytes()
    plan = EditPlan(
        clips=(EditClip(str(source), duration_seconds=1.5, motion="pan_left"),),
        output_width=240,
        output_height=240,
        fps=12,
        target_duration_seconds=1.5,
    )
    root = tmp_path / "output"
    result = OpenCutEditAdapter(root, input_roots=[tmp_path]).render(
        plan, str(root / "panned.mp4")
    )
    media = FFmpegAdapter().probe_media(result["video_path"])
    first_frame = root / "first.png"
    last_frame = root / "last.png"
    FFmpegAdapter().extract_frames_at(result["video_path"], [str(first_frame)], [0.1])
    FFmpegAdapter().extract_frames_at(result["video_path"], [str(last_frame)], [1.4])
    with Image.open(first_frame) as first_image:
        first_means = ImageStat.Stat(first_image.convert("RGB")).mean
    with Image.open(last_frame) as last_image:
        last_means = ImageStat.Stat(last_image.convert("RGB")).mean

    assert media["has_video"] and media["has_audio"]
    assert (media["width"], media["height"]) == (240, 240)
    assert abs(media["video_duration"] - 1.5) <= 1 / plan.fps
    assert first_means[0] > first_means[2]
    assert last_means[2] > last_means[0]
    assert source.read_bytes() == original


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
        FFmpegAdapter().extract_frames_at(result["video_path"], [str(frame)], [timestamp])
        with Image.open(frame) as image:
            means = ImageStat.Stat(image.convert("RGB")).mean
        assert means[channel] > max(means[(channel + 1) % 3], means[(channel + 2) % 3]) * 1.4
    assert [path.read_bytes() for path in sources] == originals


def test_user_given_a_rendered_candidate_when_materialized_then_the_selected_media_and_manifest_are_preserved(tmp_path):
    """User Given a rendered edit candidate When it is materialized at selected output paths Then playable media and its manifest describe the copied artifacts."""
    source = tmp_path / "source.png"
    Image.new("RGB", (240, 426), "blue").save(source)
    original_source = source.read_bytes()
    root = tmp_path / "output"
    candidate_dir = root / "candidate"
    adapter = OpenCutEditAdapter(root, input_roots=[tmp_path])
    candidate = adapter.render(
        EditPlan(
            clips=(EditClip(str(source), duration_seconds=1),),
            output_width=240,
            output_height=426,
            fps=12,
        ),
        str(candidate_dir / "candidate.mp4"),
        contact_sheet_path=str(candidate_dir / "contact_sheet.jpg"),
    )

    final_video = root / "selected" / "edited.mp4"
    final_contact_sheet = root / "selected" / "contact_sheet.jpg"
    materialized = adapter.materialize_result(
        candidate,
        output_path=str(final_video),
        contact_sheet_path=str(final_contact_sheet),
    )
    candidate_manifest = json.loads(Path(str(candidate["manifest_path"])).read_text(encoding="utf-8"))
    final_manifest = json.loads(Path(str(materialized["manifest_path"])).read_text(encoding="utf-8"))
    media = FFmpegAdapter().probe_media(str(final_video))

    assert media["has_video"] and media["has_audio"]
    assert final_video.read_bytes() == Path(str(candidate["video_path"])).read_bytes()
    assert final_contact_sheet.read_bytes() == Path(str(candidate["contact_sheet_path"])).read_bytes()
    assert final_manifest["output"]["path"] == str(final_video)
    assert final_manifest["output"]["sha256"] == candidate_manifest["output"]["sha256"]
    assert materialized["contact_sheet_path"] == str(final_contact_sheet)
    assert source.read_bytes() == original_source


def test_user_given_candidate_media_changes_after_render_when_materialized_then_the_copy_is_rejected_without_overwriting_selected_output(tmp_path):
    """User Given candidate media changed after its manifest was written When materialization is requested Then the copy is rejected and an existing selected output remains intact."""
    source = tmp_path / "source.png"
    Image.new("RGB", (240, 426), "blue").save(source)
    root = tmp_path / "output"
    adapter = OpenCutEditAdapter(root, input_roots=[tmp_path])
    candidate = adapter.render(
        EditPlan(clips=(EditClip(str(source), duration_seconds=1),), output_width=240, output_height=426, fps=12),
        str(root / "candidate" / "candidate.mp4"),
    )
    candidate_video = Path(str(candidate["video_path"]))
    candidate_video.write_bytes(b"changed candidate media")
    selected_video = root / "selected" / "edited.mp4"
    selected_video.parent.mkdir(parents=True)
    selected_video.write_bytes(b"existing selected output")

    with pytest.raises(EditRenderError):
        adapter.materialize_result(candidate, output_path=str(selected_video))

    assert selected_video.read_bytes() == b"existing selected output"


@pytest.mark.parametrize("mode", ["hard_cut", "fade"])
def test_user_given_three_clips_when_review_evidence_is_requested_then_opening_joins_and_ending_are_returned_as_images(mode, tmp_path):
    """User Given three rendered clips with hard cuts or fades When review evidence is requested Then the public renderer returns readable opening, join-adjacent, and ending frames."""
    sources = color_sources(tmp_path)
    cuts = mode == "hard_cut"
    plan = EditPlan(
        clips=tuple(EditClip(str(path), duration_seconds=1) for path in sources),
        transitions=() if cuts else (EditTransition("fade", 0.25), EditTransition("fade", 0.25)),
        profile="motion_cut_v1" if cuts else "xfade_clean_v1",
        output_width=240,
        output_height=426,
        fps=12,
    )
    root = tmp_path / "output"
    evidence_dir = root / "review_frames"
    result = OpenCutEditAdapter(root, input_roots=[tmp_path]).render(
        plan,
        str(root / "edited.mp4"),
        review_evidence_dir=str(evidence_dir),
    )

    evidence_paths = [Path(path) for path in result["review_evidence_paths"]]
    names = {path.name for path in evidence_paths}
    assert len(evidence_paths) == 8
    assert any(name.startswith("opening_") for name in names)
    assert any(name.startswith("ending_") for name in names)
    for boundary in (1, 2):
        for position in ("before", "join", "after"):
            assert any(
                name.startswith(f"boundary_{boundary:02d}_{position}_")
                for name in names
            )
    assert all(path.is_file() and path.parent == evidence_dir for path in evidence_paths)
    for path in evidence_paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert image.size == (240, 426)


def test_user_given_many_clips_when_review_evidence_hits_its_limit_then_opening_ending_and_all_joins_are_sampled(tmp_path):
    """User Given ten clips with nine joins When review evidence reaches its frame limit Then opening, ending, and every join remain represented."""
    sources = color_sources(tmp_path)
    clips = tuple(EditClip(str(sources[index % len(sources)]), duration_seconds=1) for index in range(10))
    plan = EditPlan(
        clips=clips,
        transitions=tuple(EditTransition("fade", 0.25) for _ in range(len(clips) - 1)),
        output_width=240,
        output_height=426,
        fps=12,
    )
    root = tmp_path / "output"
    evidence_dir = root / "review_frames"
    result = OpenCutEditAdapter(root, input_roots=[tmp_path]).render(
        plan,
        str(root / "edited.mp4"),
        review_evidence_dir=str(evidence_dir),
    )

    evidence_paths = [Path(path) for path in result["review_evidence_paths"]]
    names = {path.name for path in evidence_paths}
    assert len(evidence_paths) == 24
    assert any(name.startswith("opening_") for name in names)
    assert any(name.startswith("ending_") for name in names)
    for boundary in range(1, len(clips)):
        assert any(name.startswith(f"boundary_{boundary:02d}_") for name in names)
    for path in evidence_paths:
        with Image.open(path) as image:
            assert image.format == "JPEG"


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
