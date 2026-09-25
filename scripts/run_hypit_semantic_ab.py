"""Run a matched A/B benchmark for semantic cue timing.

The B variant applies the useful Hypit idea of semantic event anchoring while
reusing MediaOverload's existing prompt, Krea2, MiniMax H3, and hard-media-QA
route. Each pair changes only ``semantic_cue_mode``; generated media and
manifests stay outside the repository by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
from agentic.app.character_workflow import run_character_workflow  # noqa: E402
from agentic.runtime.visual_action_contract import (  # noqa: E402
    SEMANTIC_CUE_MODE,
    semantic_cue_timeline,
)


CASES: tuple[dict[str, str], ...] = (
    {
        "case_id": "lunchbox_snap",
        "prompt": "Kirby-like pink hero already reaches toward one oversized lunchbox lid as a tiny snack rolls away; the lid snaps shut, the hero bounces back, catches the snack, and settles into a proud pose.",
    },
    {
        "case_id": "noodle_scarf",
        "prompt": "Kirby-like pink hero already tugs one stubborn noodle with both hands; it suddenly slurps free, the hero wobbles backward, then lands with the noodle wrapped like a silly scarf.",
    },
    {
        "case_id": "bubble_pop",
        "prompt": "Kirby-like pink hero already leans into one giant soap bubble; the bubble stretches around the hero, pops with a harmless wobble, and leaves the hero blinking in the same opening direction.",
    },
    {
        "case_id": "mochi_bounce",
        "prompt": "Kirby-like pink hero already pokes one oversized mochi with a wooden skewer; the mochi rolls away, bumps the hero's feet, and returns as a tiny hat while the hero reacts with a surprised bounce.",
    },
    {
        "case_id": "pebble_puff",
        "prompt": "Kirby-like pink hero already swings one small toy bat toward a round pebble; contact creates a cute puff, the hero recoils, and the pebble lands beside the hero in a settled comedic result.",
    },
)

SINGLE_PROTAGONIST_CONTRACT = (
    "Exactly one visible selected protagonist; no duplicate, clone, reflection, miniature copy, "
    "background character, or second version of the protagonist. Keep one dominant physical mechanism."
)


def stable_seed(case_id: str, seed_base: int) -> int:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 100_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _extract_evidence(result: dict[str, Any], variant_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
    state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
    node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    qa = node_outputs.get("video-qa") if isinstance(node_outputs.get("video-qa"), dict) else {}
    video_paths = sorted(str(path) for path in variant_dir.rglob("*.mp4") if path.is_file())
    technical_pass = bool(qa.get("passed")) and bool(video_paths)
    return {
        "workflow_status": str(result.get("status") or "failed"),
        "technical_pass": technical_pass,
        "qa_passed": bool(qa.get("passed")),
        "qa_errors": [str(item) for item in qa.get("errors", []) if str(item).strip()],
        "contact_sheet_path": str(qa.get("contact_sheet_path") or ""),
        "video_paths": video_paths,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failure_reason": str(result.get("failure_reason") or ""),
    }


def _run_variant(
    *,
    case: dict[str, str],
    variant: str,
    semantic_mode: str,
    seed: int,
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant_dir = case_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    timeline = (
        semantic_cue_timeline(args.duration_seconds, media_type="text2img2video")
        if semantic_mode
        else {}
    )
    controls = {
        "prompt": case["prompt"],
        "single_protagonist_contract": SINGLE_PROTAGONIST_CONTRACT,
        "generation_type": "text2image2video",
        "subject_mode": "single",
        "duration_seconds": int(args.duration_seconds),
        "seed": int(seed),
        "comfy_port": int(args.comfy_port),
        "semantic_cue_mode": semantic_mode,
    }
    source_signature = hashlib.sha256(
        json.dumps({key: value for key, value in controls.items() if key != "semantic_cue_mode"}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run_record = {
        "case_id": case["case_id"],
        "variant": variant,
        "changed_variable": "semantic_cue_mode",
        "semantic_cue_mode": semantic_mode,
        "source_signature": source_signature,
        "controls": controls,
        "semantic_cue_timeline": timeline,
        "manual_review_rubric": [
            "action_readability",
            "semantic_order_and_cause_effect",
            "subject_identity_and_geography",
            "payoff_readability",
            "overall_visual_quality",
        ],
    }
    _json_write(variant_dir / "variant_config.json", run_record)

    started = time.perf_counter()
    try:
        result = run_character_workflow(
            CharacterWorkflowRequest(
                repo_root=REPO_ROOT,
                config_path=REPO_ROOT / "configs" / "characters" / "kirby.yaml",
                generation=CharacterGenerationOptions(
                    prompt=f"{case['prompt']}\n\n{SINGLE_PROTAGONIST_CONTRACT}",
                    preferred_generation_type="text2image2video",
                    duration_seconds=int(args.duration_seconds),
                    output_dir=str(variant_dir),
                    routing_history_path=str(variant_dir / "routing_history.json"),
                    rng=random.Random(seed),
                    seed=seed,
                    semantic_cue_mode=semantic_mode,
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
                    comfy_port=int(args.comfy_port),
                    comfy_root=Path(args.comfy_root).resolve(),
                    auto_download_assets=False,
                ),
            )
        )
        result_dict = result if isinstance(result, dict) else dict(result)
        evidence = _extract_evidence(result_dict, variant_dir, time.perf_counter() - started)
        _json_write(variant_dir / "workflow_result.json", result_dict)
    except Exception as exc:
        evidence = {
            "workflow_status": "failed",
            "technical_pass": False,
            "qa_passed": False,
            "qa_errors": [],
            "contact_sheet_path": "",
            "video_paths": [],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    run_record["evidence"] = evidence
    _json_write(variant_dir / "variant.json", run_record)
    return run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five matched Hypit-inspired semantic cue A/B pairs")
    parser.add_argument(
        "--output-root",
        default=r"E:\comfyui\_extra\benchmarks\hypit_semantic_ab",
        help="Evidence root outside the repository",
    )
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--duration-seconds", type=int, default=6, choices=range(4, 11))
    parser.add_argument("--seed-base", type=int, default=20260921)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    allowed_roots = [run_dir, REPO_ROOT]
    existing_roots = [
        Path(item.strip()).expanduser().resolve()
        for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if item.strip()
    ]
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path) for path in dict.fromkeys([*existing_roots, *allowed_roots])
    )
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hypit_source": "https://github.com/hypit-ai/hypit",
        "route": "existing text2image2video -> Krea2 -> MiniMax H3 I2V -> hard media QA",
        "variant_a": "current_prompt_contract",
        "variant_b": SEMANTIC_CUE_MODE,
        "changed_variable": "semantic_cue_mode",
        "pair_count_target": int(args.limit),
        "duration_seconds": int(args.duration_seconds),
        "seed_base": int(args.seed_base),
        "comfy_host": args.comfy_host,
        "comfy_port": int(args.comfy_port),
        "llm_mode": os.environ.get("AGENTIC_LLM_MODE", "llm"),
        "manual_review_required": True,
        "manual_review_rubric": [
            "action_readability",
            "semantic_order_and_cause_effect",
            "subject_identity_and_geography",
            "payoff_readability",
            "overall_visual_quality",
        ],
    }
    _json_write(run_dir / "run_config.json", run_config)

    pairs: list[dict[str, Any]] = []
    for index, case in enumerate(CASES[: int(args.limit)], start=1):
        seed = stable_seed(case["case_id"], int(args.seed_base))
        case_dir = run_dir / "cases" / case["case_id"]
        print(f"[{index}/{args.limit}] {case['case_id']} | seed={seed}", flush=True)
        control = _run_variant(
            case=case,
            variant="A",
            semantic_mode="",
            seed=seed,
            case_dir=case_dir,
            args=args,
        )
        treatment = _run_variant(
            case=case,
            variant="B",
            semantic_mode=SEMANTIC_CUE_MODE,
            seed=seed,
            case_dir=case_dir,
            args=args,
        )
        pair = {
            "case_id": case["case_id"],
            "seed": seed,
            "source_signature": control["source_signature"],
            "control": control,
            "treatment": treatment,
            "technical_both_passed": bool(
                control["evidence"]["technical_pass"] and treatment["evidence"]["technical_pass"]
            ),
            "creative_winner": "undecided_until_manual_review",
        }
        pairs.append(pair)
        _json_write(
            run_dir / "benchmark_summary.json",
            {**run_config, "pairs_completed": len(pairs), "pairs": pairs},
        )
        print(
            f"[{index}/{args.limit}] done | A={control['evidence']['technical_pass']} | B={treatment['evidence']['technical_pass']}",
            flush=True,
        )

    summary = {
        **run_config,
        "pairs_completed": len(pairs),
        "technical_both_passed_pairs": sum(bool(pair["technical_both_passed"]) for pair in pairs),
        "pairs": pairs,
        "conclusion": "manual_contact_sheet_review_required_before_any_default_change",
    }
    _json_write(run_dir / "benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if len(pairs) == int(args.limit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
