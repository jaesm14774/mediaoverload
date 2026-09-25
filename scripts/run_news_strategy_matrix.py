"""Run every enabled news-driven strategy and retain full prompt evidence.

This is an artifact-generation harness, not a publishing runner. It selects a
fresh database news item for each run, executes the real character workflow,
and copies its LLM/node trace next to the resulting media.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.app.character_requests import (  # noqa: E402
    CharacterGenerationOptions,
    CharacterReviewOptions,
    CharacterRuntimeOptions,
    CharacterWorkflowRequest,
)
from agentic.app.character_workflow import (  # noqa: E402
    CONFIG_MEDIA_TYPE_MAP,
    run_character_workflow,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _enabled_strategies() -> list[str]:
    config = yaml.safe_load((REPO_ROOT / "configs" / "routing.yaml").read_text(encoding="utf-8")) or {}
    routing = dict(config.get("routing") or {})
    if not bool(routing.get("enabled", True)):
        raise RuntimeError("configs/routing.yaml disables routing; refusing to start the strategy matrix")
    candidates = routing.get("strategy_candidates") or []
    strategies = list(dict.fromkeys(str(value).strip() for value in candidates if str(value).strip()))
    if not strategies:
        raise RuntimeError("No enabled routing.strategy_candidates were found")
    unsupported = [strategy for strategy in strategies if strategy not in CONFIG_MEDIA_TYPE_MAP]
    if unsupported:
        raise RuntimeError(f"Unsupported routing strategy candidate(s): {', '.join(unsupported)}")
    return strategies


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _checked_matrix_path(matrix_root: Path, path: Path) -> Path:
    root = Path(matrix_root)
    if not root.is_absolute():
        root = Path(os.path.abspath(root))
    resolved_root = root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Matrix path escapes its root: {candidate}") from exc

    current = candidate
    while True:
        if _is_reparse_point(current):
            raise RuntimeError(f"Matrix path contains a symlink or junction: {current}")
        current_resolved = current.resolve()
        try:
            current_resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(f"Matrix path resolves outside its root: {current}") from exc
        if current_resolved == resolved_root:
            break
        if current.parent == current:
            raise RuntimeError(f"Matrix path does not resolve beneath its root: {candidate}")
        current = current.parent
    return candidate


def _safe_terminal_text(value: Any) -> str:
    text = str(value)
    bidi_controls = {"ALM", "LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
    return "".join(
        character
        if (
            ord(character) >= 32
            and not 127 <= ord(character) <= 159
            and unicodedata.bidirectional(character) not in bidi_controls
        )
        else f"\\u{ord(character):04x}"
        for character in text
    )


def _find_prompt_fields(value: Any, *, prefix: str = "") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, str) and "prompt" in str(key).casefold() and item.strip():
                found.append({"field": path, "text": item})
            else:
                found.extend(_find_prompt_fields(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_prompt_fields(item, prefix=f"{prefix}[{index}]"))
    return found


def _find_named_strings(value: Any, name: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() == name.casefold() and isinstance(item, str) and item.strip():
                found.append(item)
            else:
                found.extend(_find_named_strings(item, name))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_named_strings(item, name))
    return found


def _make_prompt_trace(trace_dir: Path, run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(trace_dir / "run_manifest.json")
    llm_calls: list[dict[str, Any]] = []
    for path in sorted((trace_dir / "llm").glob("*.json")):
        call = _load_json(path)
        llm_calls.append(
            {
                "file": str(path.relative_to(run_dir)),
                "schema_name": call.get("schema_name", ""),
                "attempt": call.get("attempt"),
                "model_role": call.get("model_role", ""),
                "model_id": call.get("model_id", ""),
                "status": call.get("status", ""),
                "messages": call.get("messages", []),
                "raw_response": call.get("raw_response"),
                "parsed_payload": call.get("parsed_payload"),
                "error": call.get("error", ""),
            }
        )

    node_traces: list[dict[str, Any]] = []
    comfy_summaries: list[dict[str, Any]] = []
    media_root = _checked_matrix_path(run_dir, run_dir / "media")
    for path in sorted((trace_dir / "nodes").glob("*.json")):
        node = _load_json(path)
        outputs = node.get("outputs", {})
        node_traces.append(
            {
                "file": str(path.relative_to(run_dir)),
                "node_id": node.get("node_id", ""),
                "skill_name": node.get("skill_name", ""),
                "status": node.get("status", ""),
                "attempt": node.get("attempt"),
                "prompt_fields": _find_prompt_fields(outputs),
                "logs": node.get("logs", []),
            }
        )
        for summary_value in _find_named_strings(outputs, "summary_path"):
            try:
                summary_path = _checked_matrix_path(media_root, Path(summary_value))
            except (OSError, RuntimeError, ValueError):
                continue
            if summary_path.name.endswith("_summary.json") and summary_path.is_file():
                summary = _load_json(summary_path)
                comfy_summaries.append(
                    {
                        "source_node_file": str(path.relative_to(run_dir)),
                        "summary_file": str(summary_path.relative_to(run_dir)),
                        "tool_name": summary.get("tool_name", ""),
                        "workflow_name": summary.get("workflow_name", ""),
                        "saved_files": summary.get("saved_files", []),
                        "render_attempts": summary.get("render_attempts", []),
                        "payload": summary.get("payload", {}),
                    }
                )

    plan = manifest.get("plan") or {}
    prompt_plan_nodes = []
    for node in plan.get("nodes", []) if isinstance(plan, dict) else []:
        if not isinstance(node, dict):
            continue
        prompt_plan_nodes.append(
            {
                "node_id": node.get("node_id", ""),
                "skill_name": node.get("skill_name", ""),
                "stage": node.get("stage", ""),
                "inputs": node.get("inputs", {}),
            }
        )
    result = {
        "run_id": manifest.get("run_id", ""),
        "llm_calls": llm_calls,
        "planned_steps": prompt_plan_nodes,
        "completed_steps": node_traces,
        "comfy_payloads": comfy_summaries,
    }
    _write_json(_checked_matrix_path(run_dir, run_dir / "prompt_trace.json"), result)
    return result


def _news_context(result: dict[str, Any]) -> dict[str, Any]:
    plan = result.get("plan") or {}
    goal = plan.get("goal") if isinstance(plan, dict) else {}
    constraints = goal.get("constraints") if isinstance(goal, dict) else {}
    if isinstance(constraints, dict) and isinstance(constraints.get("news_context"), dict):
        return dict(constraints["news_context"])
    summary = result.get("routing_summary") or {}
    prompt_generation = summary.get("prompt_generation") if isinstance(summary, dict) else {}
    news = prompt_generation.get("news_context") if isinstance(prompt_generation, dict) else {}
    return dict(news) if isinstance(news, dict) else {}


def _capture_trace(run_id: str, run_dir: Path) -> str:
    source = REPO_ROOT / "logs" / "runs" / run_id
    destination = _checked_matrix_path(run_dir, run_dir / "trace")
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return str(destination)
    return ""


def _run_one(
    *,
    strategy: str,
    replicate: int,
    seed: int,
    matrix_root: Path,
    news_history_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resolved_root = matrix_root.resolve()
    run_dir = _checked_matrix_path(
        resolved_root,
        resolved_root / strategy / f"run_{replicate:02d}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        "strategy": strategy,
        "replicate": replicate,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "news_driven": True,
        "news_history_path": str(news_history_path),
        "character_prompt_anchor": "Kirby",
        "subject_mode": "single",
        "stage_probe": True,
        "no_review": True,
        "publish_after_generate": False,
        "comfy_host": args.comfy_host,
        "comfy_port": args.comfy_port,
        "comfy_root": args.comfy_root,
    }
    _write_json(_checked_matrix_path(resolved_root, run_dir / "run_config.json"), config_record)
    started = time.perf_counter()
    before = {path.name for path in (REPO_ROOT / "logs" / "runs").glob("*") if path.is_dir()}
    try:
        loaded_history = json.loads(news_history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded_history = []
    news_history_count_before = len(loaded_history) if isinstance(loaded_history, list) else 0
    result: dict[str, Any] = {}
    error = ""
    try:
        result = run_character_workflow(
            CharacterWorkflowRequest(
                repo_root=REPO_ROOT,
                config_path=REPO_ROOT / "configs" / "characters" / "kirby.yaml",
                generation=CharacterGenerationOptions(
                    prompt="Kirby",
                    preferred_generation_type=strategy,
                    output_dir=str(run_dir / "media"),
                    news_driven=True,
                    news_history_path=str(news_history_path),
                    routing_history_path=str(run_dir / "routing_history.json"),
                    rng=random.Random(seed),
                    selected_character_name="Kirby",
                    seed=seed,
                    subject_mode="single",
                ),
                review=CharacterReviewOptions(
                    publish_after_generate=False,
                    no_review=True,
                    stage_probe=True,
                    enable_review_loop=False,
                ),
                runtime=CharacterRuntimeOptions(
                    comfy_host=args.comfy_host,
                    comfy_port=args.comfy_port,
                    comfy_root=Path(args.comfy_root).resolve(),
                    auto_download_assets=False,
                ),
            )
        )
    except Exception as exc:  # Preserve a failed run as evidence and keep the matrix moving.
        error = f"{type(exc).__name__}: {exc}"

    run_id = str(result.get("run_id") or "")
    trace_path = _capture_trace(run_id, run_dir) if run_id else ""
    if not run_id:
        logs_root = REPO_ROOT / "logs" / "runs"
        created = [path for path in logs_root.glob("*") if path.is_dir() and path.name not in before]
        if len(created) == 1:
            run_id = created[0].name
            trace_path = _capture_trace(run_id, run_dir)
        elif created:
            _write_json(
                _checked_matrix_path(resolved_root, run_dir / "uncorrelated_trace_candidates.json"),
                [str(path) for path in created],
            )

    if result:
        _write_json(_checked_matrix_path(resolved_root, run_dir / "workflow_result.json"), result)
    if trace_path:
        _make_prompt_trace(Path(trace_path), run_dir)

    news = _news_context(result)
    if not news:
        try:
            history = json.loads(news_history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history = []
        if isinstance(history, list) and len(history) > news_history_count_before:
            candidate = history[-1]
            news = candidate if isinstance(candidate, dict) else {}
    media_paths = list((result.get("artifacts") or {}).get("media_paths") or []) if result else []
    status = str(result.get("status") or ("failed" if error else "unknown"))
    record = {
        **config_record,
        "run_id": run_id,
        "status": status,
        "failure_node": str(result.get("failure_node") or ""),
        "failure_skill": str(result.get("failure_skill") or ""),
        "failure_reason": str(result.get("failure_reason") or error),
        "news_title": str(news.get("title") or ""),
        "news_url": str(news.get("url") or ""),
        "news_category": str(news.get("category") or ""),
        "media_paths": media_paths,
        "media_count": len(media_paths),
        "trace_path": trace_path,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(_checked_matrix_path(resolved_root, run_dir / "run_record.json"), record)
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a news-driven matrix for every enabled MediaOverload strategy")
    parser.add_argument(
        "--output-root",
        default="",
        help="Evidence directory. Defaults to a timestamped path under E:/comfyui/_extra/benchmarks/news_strategy_matrix",
    )
    parser.add_argument("--runs-per-strategy", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Continue an existing matrix without rerunning recorded executions")
    parser.add_argument(
        "--strategy",
        dest="selected_strategies",
        action="append",
        help="Limit this matrix to an enabled strategy; repeat the option to select more than one.",
    )
    parser.add_argument("--seed-base", type=int, default=20260923)
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.runs_per_strategy < 1:
        raise SystemExit("--runs-per-strategy must be >= 1")
    enabled_strategies = _enabled_strategies()
    requested_values = list(dict.fromkeys(args.selected_strategies or []))
    requested_strategies = [strategy for strategy in enabled_strategies if strategy in requested_values]
    unsupported = [strategy for strategy in requested_values if strategy not in enabled_strategies]
    if unsupported:
        raise SystemExit(f"Requested strategy is not enabled: {', '.join(unsupported)}")
    runtime_settings = {
        "comfy_host": str(args.comfy_host),
        "comfy_port": int(args.comfy_port),
        "comfy_root": str(Path(args.comfy_root).expanduser().resolve()),
    }
    if args.resume:
        if not args.output_root:
            raise SystemExit("--resume requires --output-root")
        matrix_root = Path(args.output_root).expanduser().resolve()
        if not matrix_root.is_dir():
            raise SystemExit(f"Cannot resume: matrix root does not exist: {matrix_root}")
        matrix_config_path = _checked_matrix_path(matrix_root, matrix_root / "matrix_config.json")
        prior_config = _load_json(matrix_config_path)
        if not prior_config:
            raise SystemExit(f"Cannot resume: matrix_config.json is missing from {matrix_root}")
        strategies = list(prior_config.get("strategies") or [])
        saved_enabled_strategies = list(prior_config.get("enabled_strategies") or [])
        if not saved_enabled_strategies:
            if strategies != enabled_strategies:
                raise SystemExit("Cannot resume because the matrix does not record the full enabled strategy list")
            saved_enabled_strategies = enabled_strategies
        if saved_enabled_strategies != enabled_strategies:
            raise SystemExit("Cannot resume because the enabled strategy list changed")
        if not strategies or len(set(strategies)) != len(strategies) or any(item not in enabled_strategies for item in strategies):
            raise SystemExit("Cannot resume because the saved strategy selection is invalid")
        if requested_strategies and requested_strategies != strategies:
            raise SystemExit("Cannot resume with a different strategy selection")
        if int(prior_config.get("runs_per_strategy", 0)) != args.runs_per_strategy:
            raise SystemExit("Cannot resume with a different --runs-per-strategy value")
        if int(prior_config.get("seed_base", -1)) != args.seed_base:
            raise SystemExit("Cannot resume with a different --seed-base value")
        news_history_path = _checked_matrix_path(matrix_root, matrix_root / "news_history.json")
        if not news_history_path.is_file():
            raise SystemExit("Cannot resume because the shared news history is missing")
        saved_history_path = str(prior_config.get("news_history_path") or "")
        if saved_history_path and Path(saved_history_path).resolve() != news_history_path.resolve():
            raise SystemExit("Cannot resume because the configured news history path changed")

        for key, expected in runtime_settings.items():
            saved = prior_config.get(key)
            if saved is not None:
                actual = str(Path(str(saved)).expanduser().resolve()) if key == "comfy_root" else saved
                if actual != expected:
                    raise SystemExit(f"Cannot resume with a different {key} value")

        progress_path = _checked_matrix_path(matrix_root, matrix_root / "matrix_progress.json")
        prior_progress = _load_json(progress_path)
        prior_runs = prior_progress.get("runs") or []
        run_records: list[dict[str, Any]] = []
        completed_by_key: dict[tuple[str, int], dict[str, Any]] = {}

        def add_completed_record(record: dict[str, Any], *, source: str) -> None:
            strategy = str(record.get("strategy") or "")
            replicate = int(record.get("replicate") or 0)
            key = (strategy, replicate)
            if strategy not in strategies or not 1 <= replicate <= args.runs_per_strategy:
                raise SystemExit(f"Cannot resume because {source} has an out-of-matrix run key: {key}")
            if int(record.get("seed", -1)) != int(args.seed_base) + replicate:
                raise SystemExit(f"Cannot resume because {source} has a mismatched seed for {key}")
            if record.get("publish_after_generate") is not False:
                raise SystemExit(f"Cannot resume because {source} is not artifact-only: {key}")
            prior = completed_by_key.get(key)
            if prior is not None:
                if prior != record:
                    raise SystemExit(f"Cannot resume because duplicate records disagree for {key}")
                return
            completed_by_key[key] = record
            run_records.append(record)

        for record in prior_runs:
            if not isinstance(record, dict):
                raise SystemExit("Cannot resume because matrix progress contains an invalid run record")
            add_completed_record(record, source="matrix progress")

        for strategy in strategies:
            for replicate in range(1, args.runs_per_strategy + 1):
                run_dir = _checked_matrix_path(matrix_root, matrix_root / strategy / f"run_{replicate:02d}")
                if not run_dir.is_dir():
                    if (strategy, replicate) in completed_by_key:
                        raise SystemExit(f"Cannot resume because completed run evidence is missing: {strategy}/{replicate}")
                    continue
                run_config_path = _checked_matrix_path(matrix_root, run_dir / "run_config.json")
                saved_run_config = _load_json(run_config_path)
                if saved_run_config:
                    for key, expected in runtime_settings.items():
                        saved = saved_run_config.get(key)
                        if saved is not None:
                            actual = str(Path(str(saved)).expanduser().resolve()) if key == "comfy_root" else saved
                            if actual != expected:
                                raise SystemExit(f"Cannot resume because {strategy}/{replicate} used a different {key}")
                run_record_path = _checked_matrix_path(matrix_root, run_dir / "run_record.json")
                if run_record_path.is_file():
                    disk_record = _load_json(run_record_path)
                    if not disk_record:
                        raise SystemExit(f"Cannot resume because {run_record_path} is invalid")
                    add_completed_record(disk_record, source=str(run_record_path))
                    if (str(disk_record.get("strategy") or ""), int(disk_record.get("replicate") or 0)) != (
                        strategy,
                        replicate,
                    ):
                        raise SystemExit(f"Cannot resume because {run_record_path} describes a different run")

        matrix_config = {
            **prior_config,
            "enabled_strategies": enabled_strategies,
            **runtime_settings,
        }
    else:
        strategies = (
            [strategy for strategy in enabled_strategies if strategy in requested_strategies]
            if requested_strategies
            else enabled_strategies
        )
        if args.output_root:
            matrix_root = Path(args.output_root).expanduser().resolve()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            matrix_root = Path(r"E:\comfyui\_extra\benchmarks\news_strategy_matrix") / stamp
        matrix_root.mkdir(parents=True, exist_ok=False)
        news_history_path = _checked_matrix_path(matrix_root, matrix_root / "news_history.json")
        run_records = []
        matrix_config = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo_root": str(REPO_ROOT),
            "routing_config": str(REPO_ROOT / "configs" / "routing.yaml"),
            "strategies": strategies,
            "enabled_strategies": enabled_strategies,
            "runs_per_strategy": args.runs_per_strategy,
            "seed_base": args.seed_base,
            "news_history_path": str(news_history_path),
            **runtime_settings,
            "artifact_only": True,
            "review_mode": "stage_probe auto selection; no Discord review request",
            "publish_after_generate": False,
        }

    allowed_roots = [REPO_ROOT, matrix_root]
    allowed_roots.extend(
        Path(item.strip()).expanduser().resolve()
        for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if item.strip()
    )
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path) for path in dict.fromkeys(allowed_roots)
    )
    matrix_config_path = _checked_matrix_path(matrix_root, matrix_root / "matrix_config.json")
    progress_path = _checked_matrix_path(matrix_root, matrix_root / "matrix_progress.json")
    _write_json(_checked_matrix_path(matrix_root, matrix_config_path), matrix_config)
    _write_json(_checked_matrix_path(matrix_root, progress_path), {**matrix_config, "runs": run_records})
    print(f"Matrix root: {matrix_root}", flush=True)
    print(f"Enabled strategies ({len(strategies)}): {', '.join(strategies)}", flush=True)

    total = len(strategies) * args.runs_per_strategy
    completed_keys = set(completed_by_key) if args.resume else set()
    for strategy in strategies:
        for replicate in range(1, args.runs_per_strategy + 1):
            if (strategy, replicate) in completed_keys:
                continue
            run_dir = _checked_matrix_path(matrix_root, matrix_root / strategy / f"run_{replicate:02d}")
            if run_dir.exists():
                archive_dir = _checked_matrix_path(matrix_root, matrix_root / "interrupted_attempts" / strategy)
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = _checked_matrix_path(matrix_root, archive_dir / (
                    f"run_{replicate:02d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                ))
                safe_run_dir = _checked_matrix_path(matrix_root, run_dir)
                safe_archive_path = _checked_matrix_path(matrix_root, archive_path)
                shutil.move(str(safe_run_dir), str(safe_archive_path))
            # Each replicate uses the same seed across strategy families while
            # news is freshly queried and uniquely selected for every run.
            seed = int(args.seed_base) + replicate
            ordinal = len(run_records) + 1
            print(f"[{ordinal}/{total}] {strategy} run {replicate}/{args.runs_per_strategy} seed={seed}", flush=True)
            record = _run_one(
                strategy=strategy,
                replicate=replicate,
                seed=seed,
                matrix_root=matrix_root,
                news_history_path=news_history_path,
                args=args,
            )
            run_records.append(record)
            completed_keys.add((strategy, replicate))
            _write_json(
                _checked_matrix_path(matrix_root, progress_path),
                {**matrix_config, "runs": run_records},
            )
            title = record["news_title"] or "<news selection missing>"
            outcome = record["failure_reason"] or f"{record['media_count']} media artifact(s)"
            print(
                f"[{ordinal}/{total}] {_safe_terminal_text(record['status'])} | "
                f"news={_safe_terminal_text(title)} | {_safe_terminal_text(outcome)}",
                flush=True,
            )

    results_path = _checked_matrix_path(matrix_root, matrix_root / "matrix_results.json")
    _write_json(
        _checked_matrix_path(matrix_root, results_path),
        {**matrix_config, "runs": run_records},
    )
    print(f"Completed {len(run_records)}/{total} runs. Results: {matrix_root / 'matrix_results.json'}", flush=True)
    return 0 if all(item["status"] == "success" for item in run_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
