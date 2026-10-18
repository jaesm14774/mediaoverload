"""Compare sprite sampling with fixed real media, no GPU or provider calls.

User Given unchanged source media and export settings When both sampling paths
run in alternating order Then pixels, exported assets and wall time are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentic" / "src"))
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.sprite_adapter import SPRITE_FRAME_COUNT, SpriteAdapter


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def same_pixels(left: str, right: str) -> bool:
    with Image.open(left) as first, Image.open(right) as second:
        return first.size == second.size and first.convert("RGBA").tobytes() == second.convert("RGBA").tobytes()


def benchmark(source: Path, root: Path, *, repeats: int = 3, warmup: int = 1,
              cell_size: int = 64, fps: float = 12) -> dict:
    source = source.resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    ffmpeg = FFmpegAdapter()
    probe = ffmpeg.probe_media(str(source))
    duration = float(probe.get("video_duration") or probe.get("duration") or 0)
    if not probe.get("has_video") or duration <= 0:
        raise ValueError("Benchmark input must contain a video with positive duration")
    epsilon = min(1 / max(float(probe.get("frame_rate") or 24), 1), duration / 100)
    timestamps = [min(duration - epsilon, duration * ((index + 0.5) / SPRITE_FRAME_COUNT))
                  for index in range(SPRITE_FRAME_COUNT)]
    parameters = {"cell_width": cell_size, "cell_height": cell_size, "fps": fps}
    source_digest = digest(source)
    report = {"source": str(source), "source_sha256": source_digest, "source_probe": probe,
              "parameters": parameters, "timestamps_seconds": timestamps, "repeats": repeats,
              "warmup_pairs_excluded": warmup, "trials": [], "passed": False,
              "environment": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "python": platform.python_version(), "ffmpeg": subprocess.run(
                                  ["ffmpeg", "-version"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", check=True).stdout.splitlines()[0]},
              "measurement_scope": "CPU sampling and packaging only; source inference, GPU quality and aesthetic preference are unmeasured"}
    report_path = root / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    try:
        for ordinal in range(warmup + repeats):
            trial = {"index": ordinal, "warmup": ordinal < warmup, "variants": {}}
            order = ["individual", "batch"] if ordinal % 2 == 0 else ["batch", "individual"]
            trial["execution_order"] = order
            for variant in order:
                output = root / f"trial-{ordinal:02d}" / variant
                raw = output / "source_frames"
                raw.mkdir(parents=True)
                paths = [str(raw / f"source-{index + 1:02d}.png") for index in range(SPRITE_FRAME_COUNT)]
                started = time.perf_counter()
                if variant == "batch":
                    ffmpeg.extract_frames_at(str(source), paths, timestamps)
                else:
                    for path, timestamp in zip(paths, timestamps):
                        ffmpeg.extract_frame_at(str(source), path, timestamp)
                sample_seconds = time.perf_counter() - started
                package = SpriteAdapter(ffmpeg).build_from_frames(paths, output_dir=str(output), **parameters)
                trial["variants"][variant] = {"sampling_seconds": sample_seconds,
                                              "total_seconds": time.perf_counter() - started,
                                              "raw_paths": paths, "package": package}
            first, second = (trial["variants"][name] for name in ("individual", "batch"))
            trial["identical_sample_count"] = sum(same_pixels(left, right) for left, right in
                                                   zip(first["raw_paths"], second["raw_paths"]))
            trial["identical_exports"] = {key: digest(Path(first["package"][key])) == digest(Path(second["package"][key]))
                                          for key in ("atlas_path", "gif_path", "webp_path")}
            trial["source_unchanged"] = digest(source) == source_digest
            trial["passed"] = (trial["identical_sample_count"] == SPRITE_FRAME_COUNT
                               and all(trial["identical_exports"].values()) and trial["source_unchanged"]
                               and all(variant["package"]["qa"]["passed"] for variant in trial["variants"].values()))
            report["trials"].append(trial)
            save()
            print(json.dumps({key: trial[key] for key in ("index", "warmup", "identical_sample_count", "passed")}), flush=True)
        measured = [trial for trial in report["trials"] if not trial["warmup"]]
        report["medians"] = {
            name: {metric: statistics.median(trial["variants"][name][metric] for trial in measured)
                   for metric in ("sampling_seconds", "total_seconds")}
            for name in ("individual", "batch")
        }
        report["speedup"] = {metric: report["medians"]["individual"][metric] / report["medians"]["batch"][metric]
                             for metric in ("sampling_seconds", "total_seconds")}
        report["passed"] = all(trial["passed"] for trial in report["trials"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New evidence directory; existing data is never overwritten")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cell-size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=12)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup must be non-negative")
    report = benchmark(args.video, args.output_dir, repeats=args.repeats, warmup=args.warmup,
                       cell_size=args.cell_size, fps=args.fps)
    print(json.dumps({key: report[key] for key in ("medians", "speedup", "passed")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
