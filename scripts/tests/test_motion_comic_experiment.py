"""Acceptance at the experiment CLI boundary with real Pillow and FFmpeg."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "run_motion_comic_experiment.py"


def story_file(root: Path) -> Path:
    panels = []
    for index, color in enumerate(("#DD3333", "#33BB55", "#3355DD")):
        path = root / f"panel-{index}.png"
        Image.new("RGB", (240, 180), color).save(path)
        panels.append(
            {
                "image_path": path.name,
                "caption": ("只吃一口", "哎呀", "也不錯")[index],
                "duration_seconds": 0.75,
            }
        )
    story = root / "story.json"
    story.write_text(
        json.dumps(
            {
                "title": "一口就好",
                "panels": panels,
                "width": 240,
                "height": 426,
                "fps": 12,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return story


def run_cli(story: Path, output: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--story",
            str(story),
            "--output-dir",
            str(output),
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="real FFmpeg required",
)
def test_user_given_three_panels_when_comparing_presentations_then_both_playable_videos_preserve_the_story_and_sources(
    tmp_path,
):
    """User Given three images and captions When the comparison CLI runs Then both videos preserve the requested story, canvas, FPS and duration and expose QA and timing evidence."""
    story = story_file(tmp_path)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.glob("panel-*.png")
    }
    output = tmp_path / "result"
    completed = run_cli(story, output)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["technical_passed"] is True
    assert report["perceptual_review"]["status"] == "awaiting_review"
    assert (
        report["story"]["panels"]
        == json.loads(story.read_text(encoding="utf-8"))["panels"]
    )
    for variant in ("slideshow", "motion_comic"):
        media = FFmpegAdapter().probe_media(report["outputs"][variant]["video_path"])
        assert media["has_video"] and (media["width"], media["height"]) == (240, 426)
        assert abs(media["frame_rate"] - 12) < 0.01
        assert abs(media["video_duration"] - 2.25) <= 1 / 12
        assert abs(media["audio_duration"] - media["video_duration"]) <= 1 / 12
        assert report["median_seconds"][variant] > 0
        assert Path(report["outputs"][variant]["contact_sheet_path"]).is_file()
    assert hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.glob("panel-*.png")
    }
    # Each settled beat must contain its own panel and every earlier panel.
    for beat, timestamp in enumerate((0.5, 1.25, 2.0)):
        frame = output / f"comic-beat-{beat}.png"
        FFmpegAdapter().extract_frame_at(
            report["outputs"]["motion_comic"]["video_path"], str(frame), timestamp
        )
        with Image.open(frame) as image:
            colors = list(image.convert("RGB").getdata())
        for channel in range(3):
            visible_fraction = sum(
                    pixel[channel] > 110
                    and pixel[channel]
                    > max(pixel[(channel + 1) % 3], pixel[(channel + 2) % 3]) * 1.4
                    for pixel in colors
                ) / len(colors)
            if channel <= beat:
                assert visible_fraction > 0.01, (beat, channel, visible_fraction)
            else:
                assert visible_fraction < 0.001, (beat, channel, visible_fraction)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_image",
        "bad_duration",
        "caption_too_long",
        "existing_output",
        "missing_beat",
        "odd_canvas",
        "invalid_fps",
        "fractional_frame",
        "empty_title",
        "unreadable_canvas",
    ],
)
def test_user_given_invalid_or_existing_inputs_when_starting_a_trial_then_no_source_or_existing_result_is_overwritten(
    tmp_path, failure
):
    """User Given a missing asset, invalid story or existing output When starting a trial Then it fails before rendering and preserves existing files."""
    story = story_file(tmp_path)
    raw = json.loads(story.read_text(encoding="utf-8"))
    output = tmp_path / "result"
    if failure == "missing_image":
        raw["panels"][1]["image_path"] = "missing.png"
    elif failure == "bad_duration":
        raw["panels"][1]["duration_seconds"] = -1
    elif failure == "caption_too_long":
        raw["panels"][1]["caption"] = "字" * 100
    elif failure == "existing_output":
        output.mkdir()
        (output / "keep.txt").write_text("existing result", encoding="utf-8")
    elif failure == "missing_beat":
        raw["panels"].pop()
    elif failure == "odd_canvas":
        raw["width"] = 241
    elif failure == "invalid_fps":
        raw["fps"] = "NaN"
    elif failure == "fractional_frame":
        raw["panels"][1]["duration_seconds"] = 0.73
    elif failure == "empty_title":
        raw["title"] = ""
    elif failure == "unreadable_canvas":
        raw["width"] = 120
    story.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    completed = run_cli(story, output)
    assert completed.returncode != 0
    assert not list(output.glob("*.mp4"))
    if failure == "existing_output":
        assert (output / "keep.txt").read_text(encoding="utf-8") == "existing result"
    else:
        assert not output.exists()


@pytest.mark.parametrize("canvas", [(240, 426), (576, 1024)])
def test_user_given_valid_long_captions_when_running_the_trial_then_dynamic_layout_preserves_the_story(tmp_path, canvas):
    """User Given captions at the valid character limit When a trial runs on a minimum or default portrait canvas Then dynamic layout accepts them and both playable videos preserve every caption."""
    story = story_file(tmp_path)
    raw = json.loads(story.read_text(encoding="utf-8"))
    raw.update(width=canvas[0], height=canvas[1])
    raw["panels"][1]["caption"] = "字" * 24
    story.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "result"
    completed = run_cli(story, output)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["story"]["panels"] == raw["panels"]
    for variant in ("slideshow", "motion_comic"):
        media = FFmpegAdapter().probe_media(report["outputs"][variant]["video_path"])
        assert media["has_video"] and media["has_audio"]
        assert (media["width"], media["height"]) == canvas
