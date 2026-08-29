"""Run independent production-style edit strategy experiments.

This script is intentionally outside the production planner.  It invokes the
same ``image_sequence_edit`` execution graph used by production, but marks the
compose node as an external-review benchmark so one trial evaluates exactly
one fixed strategy.  It never changes routing configuration or publishes
media.

The command exits non-zero when any selected strategy fails the required
repeat gate.  A single passing trial is therefore useful evidence, but never
promotes a strategy to stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageChops, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.app.main import build_runtime
from agentic.runtime.contracts import ExecutionPlan
from agentic.runtime.editing import EDIT_PROFILES, IMAGE_SUFFIXES
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.step_logger import create_run_logger
from agentic.tools.ffmpeg_adapter import FFmpegAdapter


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
DEFAULT_STRATEGIES = tuple(sorted(EDIT_PROFILES))
MINIMUM_REPEATS = 3
DEFAULT_MINIMUM_SCORE = 85
DEFAULT_MAX_BLACK_RATIO = 0.18
DEFAULT_MAX_FROZEN_RATIO = 0.90


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allow_image_root(path: Path) -> None:
    root = path.expanduser().resolve()
    configured = [
        Path(raw.strip()).expanduser().resolve()
        for raw in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if raw.strip()
    ]
    if not any(root == item or item in root.parents for item in configured):
        configured.append(root)
        os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(str(item) for item in configured)


def _configure_openrouter(review_model: str | None) -> None:
    """Make benchmark reviewer selection explicit without changing prod files."""

    os.environ["AGENTIC_LLM_MODE"] = "llm"
    os.environ["AGENTIC_TEXT_MODEL_PROVIDER"] = "openrouter"
    os.environ["AGENTIC_VISION_MODEL_PROVIDER"] = "openrouter"
    os.environ["AGENTIC_OPENROUTER_ROTATE_TEXT_MODELS"] = "false"
    os.environ["AGENTIC_OPENROUTER_ROTATE_VISION_MODELS"] = "false"
    os.environ["AGENTIC_RANDOM_MODELS"] = "false"
    os.environ["AGENTIC_OPENROUTER_MAX_TEXT_MODELS_PER_CALL"] = "1"
    os.environ["AGENTIC_OPENROUTER_MAX_VISION_MODELS_PER_CALL"] = "1"
    if review_model:
        os.environ["AGENTIC_VISION_MODEL"] = review_model.strip()
        os.environ["AGENTIC_OPENROUTER_VISION_MODEL_STRATEGY"] = "explicit"
    else:
        os.environ.pop("AGENTIC_VISION_MODEL", None)
        os.environ["AGENTIC_OPENROUTER_VISION_MODEL_STRATEGY"] = "free_pool"


def _resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    if len(raw_inputs) < 1:
        raise ValueError("At least one --input is required")
    resolved: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark input does not exist: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported benchmark input type: {path}")
        resolved.append(path)
    return resolved


def _trial_inputs(inputs: list[Path], trial_index: int, preserve_order: bool) -> list[Path]:
    if preserve_order or len(inputs) <= 1:
        return list(inputs)
    offset = trial_index % len(inputs)
    return inputs[offset:] + inputs[:offset]


def _source_record(path: Path, ffmpeg: FFmpegAdapter) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "suffix": path.suffix.lower(),
    }
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as image:
            record.update(
                {
                    "media_type": "image",
                    "width": int(image.width),
                    "height": int(image.height),
                    "duration_seconds": 3.0,
                    "has_audio": False,
                }
            )
        return record
    probe = ffmpeg.probe_media(str(path))
    record.update(
        {
            "media_type": "video",
            "width": int(probe.get("width") or 0),
            "height": int(probe.get("height") or 0),
            "duration_seconds": float(probe.get("duration") or 0.0),
            "has_audio": bool(probe.get("has_audio")),
            "probe": probe,
        }
    )
    return record


def _boundary_times(
    plan: ExecutionPlan,
    source_records: list[dict[str, Any]],
    output_duration: float,
) -> list[tuple[int, float]]:
    durations = [float(item.get("duration_seconds") or 0.0) for item in source_records]
    if not durations:
        return []
    compose = next((node for node in plan.nodes if node.node_id == "compose-edit"), None)
    if compose is None:
        return []
    raw_plan = compose.inputs.get("edit_plan")
    transitions = []
    if isinstance(raw_plan, dict):
        transitions = list(raw_plan.get("transitions") or [])
    else:
        profile = str(compose.inputs.get("profile") or "")
        duration = float(compose.inputs.get("transition_duration_seconds") or 0.10)
        if profile != "baseline_concat":
            transitions = [{"duration_seconds": duration} for _ in durations[1:]]
    boundaries: list[tuple[int, float]] = []
    if transitions:
        current = durations[0] if durations else 0.0
        for index, transition in enumerate(transitions, start=1):
            transition_duration = float(dict(transition).get("duration_seconds") or 0.0)
            boundary = max(0.0, current - transition_duration / 2.0)
            if boundary < output_duration:
                boundaries.append((index, boundary))
            if index < len(durations):
                current += durations[index] - transition_duration
    else:
        current = 0.0
        for index, duration in enumerate(durations[:-1], start=1):
            current += duration
            if current < output_duration:
                boundaries.append((index, current))
    return boundaries


def _extract_evidence(
    ffmpeg: FFmpegAdapter,
    video_path: Path,
    evidence_dir: Path,
    plan: ExecutionPlan,
    source_records: list[dict[str, Any]],
    probe: dict[str, Any],
) -> list[str]:
    duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
    fps = float(probe.get("frame_rate") or 24.0)
    if duration <= 0:
        return []
    times: list[tuple[str, float]] = [("opening", 0.0)]
    for index, boundary in _boundary_times(plan, source_records, duration):
        for label, offset in (("before", -0.12), ("join", 0.0), ("after", 0.12)):
            times.append((f"boundary_{index:02d}_{label}", boundary + offset))
    times.append(("ending", max(0.0, duration - 1.0 / max(fps, 1.0))))
    max_timestamp = max(0.0, duration - 1.0 / max(fps, 1.0))
    seen: set[float] = set()
    evidence: list[str] = []
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for label, raw_timestamp in times:
        timestamp = min(max(0.0, float(raw_timestamp)), max_timestamp)
        rounded = round(timestamp, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        output = evidence_dir / f"{label}_{timestamp:.3f}s.jpg"
        ffmpeg.extract_frame_at(str(video_path), str(output), timestamp)
        evidence.append(str(output))
    return evidence


def _image_black_ratio(path: Path) -> float:
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 64))
        pixels = list(gray.getdata())
    return sum(1 for value in pixels if value <= 8) / max(1, len(pixels))


def _image_delta(first: Path, second: Path) -> float:
    with Image.open(first) as first_image, Image.open(second) as second_image:
        left = first_image.convert("L").resize((64, 64))
        right = second_image.convert("L").resize((64, 64))
        difference = ImageChops.difference(left, right)
        return float(ImageStat.Stat(difference).mean[0]) / 255.0


def _deterministic_visual_metrics(
    ffmpeg: FFmpegAdapter,
    video_path: Path,
    evidence_paths: list[str],
    metrics_dir: Path,
    probe: dict[str, Any],
    *,
    max_black_ratio: float,
    max_frozen_ratio: float,
) -> dict[str, Any]:
    evidence = {Path(path).stem: Path(path) for path in evidence_paths if Path(path).is_file()}
    join_frames = [path for key, path in evidence.items() if "_join_" in key]
    join_black_ratios = [_image_black_ratio(path) for path in join_frames]

    sample_dir = metrics_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
    sample_count = 12
    sample_paths: list[Path] = []
    for index in range(sample_count):
        timestamp = 0.0 if sample_count == 1 else min(
            duration * index / (sample_count - 1),
            max(0.0, duration - 1.0 / max(float(probe.get("frame_rate") or 24.0), 1.0)),
        )
        path = sample_dir / f"sample_{index + 1:02d}_{timestamp:.3f}s.jpg"
        ffmpeg.extract_frame_at(str(video_path), str(path), timestamp)
        sample_paths.append(path)
    sample_deltas = [
        _image_delta(first, second)
        for first, second in zip(sample_paths, sample_paths[1:])
    ]
    frozen_pairs = sum(1 for delta in sample_deltas if delta <= 0.002)
    frozen_ratio = frozen_pairs / max(1, len(sample_deltas))
    join_delta_pairs: list[float] = []
    for key, path in evidence.items():
        if "_before_" not in key:
            continue
        after_key = key.replace("_before_", "_after_")
        after = evidence.get(after_key)
        if after:
            join_delta_pairs.append(_image_delta(path, after))
    max_join_black = max(join_black_ratios, default=0.0)
    return {
        "sample_frame_paths": [str(path) for path in sample_paths],
        "join_frame_paths": [str(path) for path in join_frames],
        "join_black_ratios": [round(value, 6) for value in join_black_ratios],
        "max_join_black_ratio": round(max_join_black, 6),
        "join_before_after_deltas": [round(value, 6) for value in join_delta_pairs],
        "mean_sample_delta": round(mean(sample_deltas), 6) if sample_deltas else 0.0,
        "frozen_pair_ratio": round(frozen_ratio, 6),
        "checks": {
            "evidence_complete": bool(evidence_paths) and all(Path(path).is_file() for path in evidence_paths),
            "join_not_black_flash": max_join_black <= max_black_ratio,
            "not_predominantly_frozen": frozen_ratio <= max_frozen_ratio,
        },
        "thresholds": {
            "max_join_black_ratio": max_black_ratio,
            "max_frozen_pair_ratio": max_frozen_ratio,
        },
    }


def _technical_summary(run_result: Any, state: dict[str, Any]) -> dict[str, Any]:
    qa = state.get("node_outputs", {}).get("edit-video-qa", {})
    if not isinstance(qa, dict):
        qa = {}
    return {
        "workflow_status": str(getattr(run_result, "status", "failed")),
        "passed": bool(qa.get("passed")) and str(getattr(run_result, "status", "failed")) == "success",
        "checks": qa.get("checks", {}),
        "errors": qa.get("errors", []),
        "warnings": qa.get("warnings", []),
        "probe": qa.get("probe", {}),
        "audio_analysis": qa.get("audio_analysis", {}),
    }


def _run_trial(
    *,
    strategy: str,
    trial_index: int,
    inputs: list[Path],
    output_root: Path,
    log_root: Path,
    goal: str,
    style: str,
    width: int,
    height: int,
    fps: float,
    target_duration: float | None,
    transition_duration: float,
    analyze_audio: bool,
    minimum_score: int,
    max_black_ratio: float,
    max_frozen_ratio: float,
) -> dict[str, Any]:
    trial_id = f"{strategy}_trial_{trial_index + 1:02d}"
    trial_dir = output_root / strategy / f"trial_{trial_index + 1:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = trial_dir / "artifacts"
    trial_log_root = log_root / strategy / f"trial_{trial_index + 1:02d}"
    recorder = RunRecorder(trial_log_root, trial_id)
    logger, _ = create_run_logger(trial_log_root, trial_id, recorder=recorder)
    ffmpeg = FFmpegAdapter()
    _allow_image_root(trial_dir)

    try:
        source_records = [_source_record(path, ffmpeg) for path in inputs]
        planner, runner, _ = build_runtime(
            REPO_ROOT,
            output_root=artifact_root,
            run_id=trial_id,
            logger=logger,
            recorder=recorder,
            input_roots=sorted({path.parent for path in inputs}),
        )
        # Benchmark runs still use the production graph but do not append to
        # the normal portfolio history or production scheduler state.
        runner.portfolio_memory = None
        output_path = artifact_root / "final.mp4"
        goal_request = planner.create_goal(
            prompt=goal,
            media_type="image_sequence_edit",
            duration_seconds=int(round(target_duration or 0)),
            style=style,
            auto_download_assets=False,
            constraints={
                "edit_input_paths": [str(path) for path in inputs],
                "edit_profile": strategy,
                "edit_variant_seed": trial_index,
                "edit_transition_duration": transition_duration,
                "edit_output_path": str(output_path),
                "edit_require_audio": False,
                "edit_analyze_audio": analyze_audio,
                "edit_creative_review": False,
            },
        )
        plan = planner.build_plan(goal_request)
        compose_node = next(node for node in plan.nodes if node.node_id == "compose-edit")
        compose_node.inputs["benchmark_mode"] = True
        plan.metadata["benchmark"] = {
            "strategy": strategy,
            "trial_index": trial_index + 1,
            "independent_run_id": trial_id,
            "external_review": True,
        }

        run_result = runner.run(plan)
        state = run_result.state.to_dict()
        node_outputs = state.get("node_outputs", {})
        compose_output = node_outputs.get("compose-edit", {})
        if not isinstance(compose_output, dict):
            compose_output = {}
        video_path = Path(str(compose_output.get("video_path") or output_path)).expanduser().resolve()
        probe = ffmpeg.probe_media(str(video_path)) if video_path.is_file() else {}
        contact_sheet = trial_dir / "contact_sheet.jpg"
        if video_path.is_file():
            ffmpeg.make_contact_sheet(
                str(video_path),
                str(contact_sheet),
                frame_count=12,
                columns=4,
                scale_width=360,
                duration_seconds=float(probe.get("duration") or 0.0),
            )
        plan_for_review = compose_output.get("plan") if isinstance(compose_output.get("plan"), dict) else plan.to_dict()
        evidence_paths = (
            _extract_evidence(
                ffmpeg,
                video_path,
                trial_dir / "join_evidence",
                plan,
                source_records,
                probe,
            )
            if video_path.is_file()
            else []
        )
        visual_metrics = (
            _deterministic_visual_metrics(
                ffmpeg,
                video_path,
                evidence_paths,
                trial_dir / "visual_metrics",
                probe,
                max_black_ratio=max_black_ratio,
                max_frozen_ratio=max_frozen_ratio,
            )
            if video_path.is_file()
            else {
                "checks": {
                    "evidence_complete": False,
                    "join_not_black_flash": False,
                    "not_predominantly_frozen": False,
                },
                "error": "production edit did not produce a video",
            }
        )
        reviewer = PromptEngine(
            llm_engine=LLMPromptEngine(
                mode=os.environ.get("AGENTIC_LLM_MODE", "llm"),
                recorder=recorder,
            )
        )
        review = reviewer.evaluate_edit_contact_sheet(
            contact_sheet_path=str(contact_sheet),
            evidence_paths=evidence_paths,
            goal=goal,
            style=style,
            plan=plan_for_review,
            candidate_attempt=1,
            previous_review=None,
        )
        technical = _technical_summary(run_result, state)
        visual_passed = all(bool(value) for value in (visual_metrics.get("checks") or {}).values())
        creative_passed = bool(review.get("passed") is True) and int(review.get("score") or 0) >= minimum_score
        trial_passed = bool(technical.get("passed")) and visual_passed and creative_passed
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "benchmark_type": "independent_production_style_edit_trial",
            "status": "pass" if trial_passed else "fail",
            "strategy": strategy,
            "trial_index": trial_index + 1,
            "independent_run_id": trial_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "production_workflow": plan.workflow_name,
            "source_records": source_records,
            "plan": plan.to_dict(),
            "output": {
                "video_path": str(video_path),
                "contact_sheet_path": str(contact_sheet),
                "probe": probe,
                "sha256": _sha256(video_path) if video_path.is_file() else "",
            },
            "join_evidence_paths": evidence_paths,
            "technical_qa": technical,
            "visual_metrics": visual_metrics,
            "creative_review": review,
            "pass_policy": {
                "minimum_score": minimum_score,
                "requires_technical_qa": True,
                "requires_deterministic_visual_checks": True,
                "requires_strict_creative_review": True,
            },
            "passed_components": {
                "technical": bool(technical.get("passed")),
                "deterministic_visual": visual_passed,
                "creative": creative_passed,
            },
        }
        receipt_path = _write_json(trial_dir / "trial_receipt.json", receipt)
        manifest_path = recorder.finalize(
            {
                "status": receipt["status"],
                "workflow_name": plan.workflow_name,
                "benchmark_trial_receipt": str(receipt_path),
                "benchmark": receipt,
            }
        )
        receipt["run_manifest"] = str(manifest_path)
        _write_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "benchmark_type": "independent_production_style_edit_trial",
            "status": "fail",
            "strategy": strategy,
            "trial_index": trial_index + 1,
            "independent_run_id": trial_id,
            "error": f"{type(exc).__name__}: {exc}",
            "trial_dir": str(trial_dir),
        }
        receipt_path = _write_json(trial_dir / "trial_receipt.json", receipt)
        recorder.finalize({"status": "fail", "benchmark_trial_receipt": str(receipt_path), "benchmark": receipt})
        return receipt


def _aggregate_strategy(strategy: str, trials: list[dict[str, Any]], minimum_score: int) -> dict[str, Any]:
    scores = [
        float(item.get("creative_review", {}).get("score"))
        for item in trials
        if isinstance(item.get("creative_review"), dict) and item.get("creative_review", {}).get("score") is not None
    ]
    passed = [bool(item.get("status") == "pass") for item in trials]
    failures: list[str] = []
    for item in trials:
        if item.get("status") == "pass":
            continue
        review = item.get("creative_review") if isinstance(item.get("creative_review"), dict) else {}
        technical = item.get("technical_qa") if isinstance(item.get("technical_qa"), dict) else {}
        visual = item.get("visual_metrics") if isinstance(item.get("visual_metrics"), dict) else {}
        failures.extend(str(value) for value in review.get("issues", []) if str(value).strip())
        failures.extend(str(value) for value in technical.get("errors", []) if str(value).strip())
        for check, value in (visual.get("checks") or {}).items():
            if value is False:
                failures.append(f"deterministic_visual:{check}")
        if item.get("error"):
            failures.append(str(item["error"]))
    all_passed = len(trials) >= MINIMUM_REPEATS and all(passed)
    return {
        "strategy": strategy,
        "trial_count": len(trials),
        "minimum_required_trials": MINIMUM_REPEATS,
        "status": "stable_for_fixture" if all_passed else "unstable_or_rejected",
        "all_trials_passed": all_passed,
        "scores": scores,
        "score_min": min(scores) if scores else None,
        "score_mean": mean(scores) if scores else None,
        "score_population_stddev": pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
        "minimum_score": minimum_score,
        "failure_reasons": sorted(set(failures))[:50],
        "trials": trials,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Edit strategy benchmark",
        "",
        f"- Created: `{report.get('created_at', '')}`",
        f"- Independent repeats per strategy: `{report.get('repeats', 0)}`",
        f"- Production graph: `{report.get('production_workflow', '')}`",
        f"- Overall status: `{report.get('status', '')}`",
        "",
        "A strategy is stable only when every required independent trial passes. A single accepted output is not stability evidence.",
    ]
    for strategy in report.get("strategies", []):
        lines.extend(
            [
                "",
                f"## `{strategy.get('strategy')}` — `{strategy.get('status')}`",
                f"- Trials: `{strategy.get('trial_count')}`; all passed=`{strategy.get('all_trials_passed')}`",
                f"- Score min/mean/stddev: `{strategy.get('score_min')}` / `{strategy.get('score_mean')}` / `{strategy.get('score_population_stddev')}`",
            ]
        )
        reasons = strategy.get("failure_reasons") or []
        if reasons:
            lines.append("- Failure evidence:")
            lines.extend(f"  - {reason}" for reason in reasons)
        for trial in strategy.get("trials", []):
            output = trial.get("output") if isinstance(trial.get("output"), dict) else {}
            lines.append(
                f"- Trial `{trial.get('trial_index')}`: `{trial.get('status')}`; video=`{output.get('video_path', '')}`; receipt=`{trial.get('run_manifest', trial.get('trial_dir', ''))}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent production-style edit strategy benchmarks")
    parser.add_argument("--input", action="append", dest="inputs", required=True, help="Real generated image/video input; repeatable")
    parser.add_argument("--output-root", help="Isolated benchmark output root")
    parser.add_argument("--strategy", action="append", dest="strategies", choices=DEFAULT_STRATEGIES, help="Strategy to benchmark; repeatable")
    parser.add_argument("--repeats", type=int, default=MINIMUM_REPEATS, help="Independent trials per strategy; minimum 3")
    parser.add_argument("--goal", default="Turn generated media into an engaging short-form edit")
    parser.add_argument("--style", default="playful cinematic short-form")
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--target-duration", type=float)
    parser.add_argument("--transition-duration", type=float, default=0.10)
    parser.add_argument(
        "--skip-audio-analysis",
        action="store_true",
        help="Skip loudness/silence analysis for image-only inputs that use a generated silent track",
    )
    parser.add_argument("--minimum-score", type=int, default=DEFAULT_MINIMUM_SCORE)
    parser.add_argument("--max-black-ratio", type=float, default=DEFAULT_MAX_BLACK_RATIO)
    parser.add_argument("--max-frozen-ratio", type=float, default=DEFAULT_MAX_FROZEN_RATIO)
    parser.add_argument("--review-model", help="Explicit OpenRouter free vision model; otherwise first configured free-pool model")
    parser.add_argument("--preserve-input-order", action="store_true", help="Do not rotate the paired input order across trials")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < MINIMUM_REPEATS:
        raise SystemExit(f"--repeats must be at least {MINIMUM_REPEATS}")
    if not 0 <= args.minimum_score <= 100:
        raise SystemExit("--minimum-score must be between 0 and 100")
    inputs = _resolve_inputs(args.inputs)
    _configure_openrouter(args.review_model)
    benchmark_name = f"edit_strategy_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        log_root = output_root.parent / "logs" / output_root.name
    else:
        output_root = REPO_ROOT / "output" / benchmark_name
        log_root = REPO_ROOT / "logs" / benchmark_name
    strategies = tuple(args.strategies or DEFAULT_STRATEGIES)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_type": "independent_production_style_edit_strategy_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_workflow": "image_sequence_edit_v1",
        "output_root": str(output_root),
        "log_root": str(log_root),
        "inputs": [str(path) for path in inputs],
        "repeats": args.repeats,
        "strategies_requested": list(strategies),
        "review_backend": {
            "provider": "openrouter",
            "model": args.review_model or "configured_free_pool_first_model",
            "rotation": False,
        },
        "audio_analysis": not args.skip_audio_analysis,
        "strategies": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for strategy in strategies:
        trials: list[dict[str, Any]] = []
        for trial_index in range(args.repeats):
            trial_inputs = _trial_inputs(inputs, trial_index, args.preserve_input_order)
            print(f"[benchmark] strategy={strategy} trial={trial_index + 1}/{args.repeats}", flush=True)
            receipt = _run_trial(
                strategy=strategy,
                trial_index=trial_index,
                inputs=trial_inputs,
                output_root=output_root,
                log_root=log_root,
                goal=args.goal,
                style=args.style,
                width=args.width,
                height=args.height,
                fps=args.fps,
                target_duration=args.target_duration,
                transition_duration=args.transition_duration,
                analyze_audio=not args.skip_audio_analysis,
                minimum_score=args.minimum_score,
                max_black_ratio=args.max_black_ratio,
                max_frozen_ratio=args.max_frozen_ratio,
            )
            trials.append(receipt)
            print(f"[benchmark] strategy={strategy} trial={trial_index + 1} status={receipt.get('status')}", flush=True)
        report["strategies"].append(_aggregate_strategy(strategy, trials, args.minimum_score))
    report["status"] = "stable_for_fixture" if all(item.get("all_trials_passed") for item in report["strategies"]) else "unstable_or_rejected"
    json_path = _write_json(output_root / "benchmark_report.json", report)
    markdown_path = output_root / "benchmark_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    report["report_paths"] = {"json": str(json_path), "markdown": str(markdown_path)}
    _write_json(json_path, report)
    print(json.dumps({"status": report["status"], "report_paths": report["report_paths"]}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "stable_for_fixture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
