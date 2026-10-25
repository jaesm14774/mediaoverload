"""Compare progressive comic panels with a slideshow using the same assets.

User Given three illustrated beats and reading durations When both presentations
render Then two social MP4s expose source lineage, technical QA and paired timing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic/src"))
from agentic.runtime.registry import ToolRegistry
from agentic.tools.media_services import register_media_service_tools
from agentic.tools.motion_comic_adapter import compose_pages, load_recipe, render_pages


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    story: dict, prepared: list[Image.Image], directory: Path, tools: ToolRegistry
) -> dict:
    result = compose_pages(story, prepared, directory, tools)
    result["qa"] = tools.call(
        "media.video_qa",
        {
            "video_path": result["video_path"],
            "target_duration": sum(
                panel["duration_seconds"] for panel in story["panels"]
            ),
            "duration_tolerance": 1 / story["fps"] + 0.01,
            "expected_fps": story["fps"],
            "analyze_audio": False,
        },
    )
    if not result["qa"]["passed"]:
        raise RuntimeError(f"Media QA failed: {result['qa']}")
    return result


def experiment(story_path: Path, output: Path, repeats: int, warmup: int) -> dict:
    story_path, output = story_path.resolve(strict=True), output.resolve()
    if output.exists():
        raise FileExistsError(
            "Output directory must be new; existing results are preserved"
        )
    if not 1 <= repeats <= 10 or not 0 <= warmup <= 3:
        raise ValueError("Use 1 to 10 measured pairs and 0 to 3 warmup pairs")
    story, sources = load_recipe(story_path)
    original_hashes = [digest(path) for path in sources]
    # Preflight both layouts (including readable text) before creating outputs.
    for variant in ("slideshow", "motion_comic"):
        render_pages(story, sources, variant)
    output.mkdir(parents=True)
    tools = ToolRegistry()
    register_media_service_tools(tools, output, input_roots=[story_path.parent])
    report = {
        "story": story,
        "story_sha256": digest(story_path),
        "sources": [{"path": str(path), "sha256": digest(path)} for path in sources],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ffmpeg": subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.splitlines()[0],
        },
        "measurement_scope": "Pillow page composition, FFmpeg encoding and technical QA; excludes shared image generation and social publishing",
        "changed_factor": "progressive comic layout versus one-panel presentation; fonts, text, reading time, transitions, canvas, FPS and source assets held fixed",
        "audio": "same silent stereo audio supplied by the existing editor; this is a visual trial",
        "warmup_pairs_excluded": warmup,
        "trials": [],
        "technical_passed": False,
        "perceptual_review": {
            "status": "awaiting_review",
            "criteria": [
                "story comprehension",
                "caption readability",
                "payoff visibility",
                "pacing",
            ],
            "scores": None,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    last = {}
    for pair in range(warmup + repeats):
        measured = pair >= warmup
        order = (
            ("slideshow", "motion_comic")
            if pair % 2 == 0
            else ("motion_comic", "slideshow")
        )
        trial = {
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "measured": measured,
            "order": list(order),
            "seconds": {},
            "outputs": {},
        }
        for variant in order:
            directory = output / f"pair-{pair + 1}" / variant
            started = time.perf_counter()
            prepared = render_pages(story, sources, variant)
            result = render(story, prepared, directory, tools)
            trial["seconds"][variant] = time.perf_counter() - started
            trial["outputs"][variant] = result
            last[variant] = result
        report["trials"].append(trial)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report["median_seconds"] = {
        variant: statistics.median(
            trial["seconds"][variant] for trial in report["trials"] if trial["measured"]
        )
        for variant in ("slideshow", "motion_comic")
    }
    report["outputs"] = {
        variant: tools.call(
            "media.materialize_edit",
            {
                "result": result,
                "output_path": str(output / f"{variant}.mp4"),
                "contact_sheet_path": str(output / f"{variant}.contact-sheet.jpg"),
            },
        )
        for variant, result in last.items()
    }
    report["sources_unchanged"] = [
        digest(path) for path in sources
    ] == original_hashes and digest(story_path) == report["story_sha256"]
    report["technical_passed"] = report["sources_unchanged"]
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report["technical_passed"]:
        raise RuntimeError("Source assets changed during the experiment")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--story",
        required=True,
        type=Path,
        help="JSON with title and three image_path/caption/duration_seconds panels",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New trial output directory"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    report = experiment(args.story, args.output_dir, args.repeats, args.warmup)
    print(
        json.dumps(
            {
                "technical_passed": report["technical_passed"],
                "outputs": {
                    variant: {
                        "video_path": result["video_path"],
                        "contact_sheet_path": result["contact_sheet_path"],
                    }
                    for variant, result in report["outputs"].items()
                },
                "median_seconds": report["median_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
