"""Run three matched MediaOverload story-arc video pairs inspired by Vox Director.

Variant A uses the current prompt contract. Variant B adds one topic-matched
arc beat map. The existing local Krea2 -> MiniMax H3 route and hard media QA
are unchanged. Generated media is written outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


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


EXPERIMENT_TEXT_PROVIDER = "ollama"
EXPERIMENT_TEXT_MODEL = "qwen3.8-27b-ud-q2xl-local:latest"
EXPERIMENT_VISION_MODEL = "google/gemma-4-26b-a4b-it:free"
DURATION_SECONDS = 6
SINGLE_PROTAGONIST_CONTRACT = (
    "Exactly one visible selected protagonist; no duplicate, clone, reflection, miniature copy, "
    "background character, or second version of the protagonist. Keep one dominant physical mechanism."
)

CASES: tuple[dict[str, str], ...] = (
    {
        "case_id": "lunchbox_payoff",
        "arc": "hook_payoff",
        "prompt": (
            "Kirby is trying to keep one oversized lunchbox from snapping shut while a tiny cookie "
            "rolls toward the hinge. Create one playful physical-comedy story with a clear ending."
        ),
        "arc_instruction": (
            "Use the hook_payoff arc for this single comic idea. Start in the first fifth with the "
            "lunchbox problem already in motion; use the middle three fifths to let the same lid "
            "mechanism turn the attempt into a surprising complication; use the final fifth to "
            "resolve the opening question, show Kirby's readable reaction, and hold on the proof."
        ),
    },
    {
        "case_id": "windmill_lantern",
        "arc": "how_it_works",
        "prompt": (
            "Kirby wants one small lantern to glow in a dark garden. A paper windmill stands between "
            "a passing gust and the unlit lantern. Show a simple, satisfying physical explanation."
        ),
        "arc_instruction": (
            "Use the how_it_works arc for this visible mechanism. Open in the first fifth on the "
            "unlit lantern and the question of how it will glow; use the middle three fifths to show "
            "the gust turning the windmill and that motion reaching the lantern in causal order; use "
            "the final fifth to show the lantern lit as the mechanism's benefit and hold on that result."
        ),
    },
    {
        "case_id": "paper_boat_recovery",
        "arc": "man_in_hole",
        "prompt": (
            "Kirby wants one paper boat to cross a puddle safely. The first leaf used as a bridge is "
            "already tipping, and the boat must still reach the far bank. Make the recovery physical and clear."
        ),
        "arc_instruction": (
            "Use a compact man_in_hole arc for this recovery story. In the first fifth, establish the "
            "crossing goal while action is already underway; in the next two fifths, let the leaf bridge "
            "fail and make the situation visibly worse; in the fourth fifth, show Kirby notice and use "
            "one physical clue to change tactics; in the final fifth, complete the same crossing and hold "
            "on the earned safer state."
        ),
    },
)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_seed(case_id: str, seed_base: int) -> int:
    offset = int(_sha256(case_id)[:8], 16) % 100_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


def _redact(value: object) -> str:
    text = str(value or "")
    return __import__("re").sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-[REDACTED]", text)


def _queue_state(host: str, port: int) -> tuple[int, int]:
    url = f"http://{host}:{port}/queue"
    try:
        with urlopen(url, timeout=3) as response:
            queue = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ComfyUI queue endpoint is unavailable: {type(exc).__name__}") from exc
    return len(queue.get("queue_running", [])), len(queue.get("queue_pending", []))


def _require_idle_comfy(host: str, port: int) -> None:
    running, pending = _queue_state(host, port)
    if running or pending:
        raise RuntimeError(f"ComfyUI is busy (running={running}, pending={pending}); no benchmark job was submitted")


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    probe = json.loads(result.stdout)
    return {
        "duration_seconds": round(float(probe.get("format", {}).get("duration") or 0), 3),
        "streams": probe.get("streams", []),
    }


def _extract_evidence(result: dict[str, Any], variant_dir: Path, elapsed_seconds: float) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
    state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
    node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    qa = node_outputs.get("video-qa") if isinstance(node_outputs.get("video-qa"), dict) else {}
    prompt_lineage = state.get("prompt_lineage") if isinstance(state.get("prompt_lineage"), list) else []
    idea_brief = next(
        (
            item
            for item in prompt_lineage
            if isinstance(item, dict) and item.get("node_id") == "idea-brief"
        ),
        {},
    )
    backend = idea_brief.get("llm_backend") if isinstance(idea_brief.get("llm_backend"), dict) else {}
    text_candidates = backend.get("openrouter_text_candidates")
    prompt_generation_pass = bool(idea_brief) and not bool(
        idea_brief.get("fallback_reason") or idea_brief.get("manager_error")
    )
    model_pin_pass = (
        backend.get("text_provider") == EXPERIMENT_TEXT_PROVIDER
        and
        backend.get("text_model_raw") == EXPERIMENT_TEXT_MODEL
    )
    video_paths = sorted(path.resolve() for path in variant_dir.rglob("*.mp4") if path.is_file())
    probes: dict[str, Any] = {}
    for path in video_paths:
        try:
            probes[str(path)] = _ffprobe(path)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
            probes[str(path)] = {"error": f"{type(exc).__name__}: {_redact(exc)}"}
    return {
        "workflow_status": str(result.get("status") or "failed"),
        "technical_pass": bool(qa.get("passed")) and bool(video_paths) and prompt_generation_pass and model_pin_pass,
        "qa_passed": bool(qa.get("passed")),
        "prompt_generation_pass": prompt_generation_pass,
        "model_pin_pass": model_pin_pass,
        "prompt_fallback_reason": _redact(idea_brief.get("fallback_reason") or ""),
        "prompt_manager_error": _redact(idea_brief.get("manager_error") or ""),
        "text_provider": str(backend.get("text_provider") or ""),
        "effective_text_model": str(backend.get("text_model_raw") or ""),
        "text_model_rotation": backend.get("openrouter_rotate_text_models"),
        "configured_text_pool": text_candidates if isinstance(text_candidates, list) else [],
        "qa_errors": [_redact(item) for item in qa.get("errors", []) if str(item).strip()],
        "contact_sheet_path": str(qa.get("contact_sheet_path") or ""),
        "video_paths": [str(path) for path in video_paths],
        "ffprobe": probes,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "failure_reason": _redact(result.get("failure_reason") or ""),
    }


def _run_variant(
    *,
    case: dict[str, str],
    variant: str,
    seed: int,
    case_dir: Path,
    args: argparse.Namespace,
    render_signature: str,
) -> dict[str, Any]:
    variant_dir = case_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=False)
    base_prompt = f"{case['prompt']}\n\n{SINGLE_PROTAGONIST_CONTRACT}"
    prompt = (
        base_prompt
        if variant == "A"
        else f"{base_prompt}\n\nTopic-matched narrative arc: {case['arc']}. {case['arc_instruction']}"
    )
    variant_record: dict[str, Any] = {
        "case_id": case["case_id"],
        "variant": variant,
        "changed_variable": "topic_matched_story_arc_instruction",
        "narrative_arc": case["arc"] if variant == "B" else "current_prompt_contract",
        "base_prompt_sha256": _sha256(base_prompt),
        "variant_prompt_sha256": _sha256(prompt),
        "render_signature": render_signature,
        "controls": {
            "generation_type": "text2image2video",
            "subject_mode": "single",
            "selected_character_name": "Kirby",
            "duration_seconds": DURATION_SECONDS,
            "seed": seed,
            "comfy_host": args.comfy_host,
            "comfy_port": int(args.comfy_port),
            "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
            "text_model": EXPERIMENT_TEXT_MODEL,
            "vision_model_provider": "openrouter",
            "vision_model": EXPERIMENT_VISION_MODEL,
        },
        "prompt": prompt,
        "manual_review_scores": {
            "causal_progression": None,
            "opening_readability": None,
            "arc_fit": None,
            "payoff": None,
            "identity_geography_continuity": None,
            "overall_visual_quality": None,
            "notes": "pending rendered-video review",
        },
    }
    _json_write(variant_dir / "variant_config.json", variant_record)

    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    started = time.perf_counter()
    try:
        result = run_character_workflow(
            CharacterWorkflowRequest(
                repo_root=REPO_ROOT,
                config_path=REPO_ROOT / "configs" / "characters" / "kirby.yaml",
                generation=CharacterGenerationOptions(
                    prompt=prompt,
                    preferred_generation_type="text2image2video",
                    duration_seconds=DURATION_SECONDS,
                    output_dir=str(variant_dir),
                    rng=random.Random(seed),
                    seed=seed,
                    subject_mode="single",
                    selected_character_name="Kirby",
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
    except Exception as exc:
        evidence = {
            "workflow_status": "failed",
            "technical_pass": False,
            "qa_passed": False,
            "qa_errors": [],
            "contact_sheet_path": "",
            "video_paths": [],
            "ffprobe": {},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "failure_reason": f"{type(exc).__name__}: {_redact(exc)}",
        }
    variant_record["evidence"] = evidence
    _json_write(variant_dir / "variant.json", variant_record)
    return variant_record


def _pin_experiment_models() -> None:
    """Use one configured text provider and prevent cross-provider fallback."""
    os.environ["AGENTIC_LLM_MODE"] = "llm"
    os.environ["AGENTIC_TEXT_MODEL_PROVIDER"] = EXPERIMENT_TEXT_PROVIDER
    os.environ["AGENTIC_TEXT_MODEL"] = EXPERIMENT_TEXT_MODEL
    os.environ["AGENTIC_OLLAMA_NUM_CTX"] = "8192"
    os.environ["AGENTIC_OLLAMA_NUM_PREDICT"] = "2048"
    os.environ["AGENTIC_OLLAMA_THINK"] = "false"
    os.environ["AGENTIC_VISION_MODEL_PROVIDER"] = "openrouter"
    os.environ["AGENTIC_VISION_MODEL"] = EXPERIMENT_VISION_MODEL
    # Keep the independent OpenRouter vision setting stable in case a review
    # path requests it; the text model is local Ollama and does not rotate.
    os.environ["AGENTIC_OPENROUTER_ROTATE_TEXT_MODELS"] = "false"
    os.environ["AGENTIC_OPENROUTER_ROTATE_VISION_MODELS"] = "false"
    os.environ["AGENTIC_RANDOM_MODELS"] = "false"
    os.environ["AGENTIC_PROVIDER_FALLBACK_ENABLED"] = "false"
    os.environ["AGENTIC_TEXT_ALLOW_FALLBACK"] = "false"
    os.environ["AGENTIC_VISION_ALLOW_FALLBACK"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three matched Vox-inspired story-arc video pairs")
    parser.add_argument(
        "--output-root",
        default=r"E:\comfyui\_extra\benchmarks\vox_director_story_arc_ab",
        help="Evidence root outside the repository",
    )
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--seed-base", type=int, default=20260925)
    parser.add_argument(
        "--case-id",
        action="append",
        choices=[case["case_id"] for case in CASES],
        help="Run selected case(s); omit to run all three matched pairs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root == REPO_ROOT or REPO_ROOT in output_root.parents:
        raise ValueError("Benchmark output must stay outside the repository")
    if not Path(args.comfy_root).is_dir():
        raise FileNotFoundError(f"ComfyUI root does not exist: {args.comfy_root}")

    selected_case_ids = set(args.case_id or (case["case_id"] for case in CASES))
    selected_cases = [case for case in CASES if case["case_id"] in selected_case_ids]
    _pin_experiment_models()
    _require_idle_comfy(args.comfy_host, int(args.comfy_port))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    existing_roots = [
        Path(item.strip()).expanduser().resolve()
        for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if item.strip()
    ]
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path) for path in dict.fromkeys([*existing_roots, REPO_ROOT.resolve(), run_dir.resolve()])
    )
    fixed_controls = {
        "route": "text2image2video -> Krea2 keyframe -> MiniMax H3 I2V -> existing hard video QA",
        "duration_seconds": DURATION_SECONDS,
        "character": "Kirby",
        "subject_mode": "single",
        "text_model_provider": EXPERIMENT_TEXT_PROVIDER,
        "text_model": EXPERIMENT_TEXT_MODEL,
        "vision_model": EXPERIMENT_VISION_MODEL,
        "text_model_rotation": False,
        "vision_model_rotation": False,
        "ollama_num_ctx": 8192,
        "ollama_num_predict": 2048,
        "ollama_think": False,
        "random_models": False,
        "provider_fallback": False,
        "comfy_host": args.comfy_host,
        "comfy_port": int(args.comfy_port),
        "seed_base": int(args.seed_base),
    }
    render_signature = _sha256(json.dumps(fixed_controls, sort_keys=True))
    run_config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_repo": "https://github.com/Alisa0808/vox-director",
        "variant_a": "current_prompt_contract",
        "variant_b": "topic_matched_story_arc_instruction",
        "changed_variable": "topic_matched_story_arc_instruction",
        "pair_count_target": len(selected_cases),
        "case_arc_map": {case["case_id"]: case["arc"] for case in selected_cases},
        "fixed_controls": fixed_controls,
        "render_signature": render_signature,
        "manual_review_required": True,
        "manual_review_threshold": "B must win causal progression in all three pairs with no material regression",
        "publish_after_generate": False,
        "dispatch": False,
    }
    _json_write(run_dir / "run_config.json", run_config)

    pairs: list[dict[str, Any]] = []
    for case in selected_cases:
        index = CASES.index(case) + 1
        seed = stable_seed(case["case_id"], int(args.seed_base))
        case_dir = run_dir / "cases" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=False)
        order = ("A", "B") if index % 2 else ("B", "A")
        variants: dict[str, dict[str, Any]] = {}
        print(f"[{index}/{len(CASES)}] {case['case_id']} arc={case['arc']} seed={seed} order={order}", flush=True)
        for variant in order:
            variants[variant] = _run_variant(
                case=case,
                variant=variant,
                seed=seed,
                case_dir=case_dir,
                args=args,
                render_signature=render_signature,
            )
            _json_write(case_dir / "pair.json", {
                "case_id": case["case_id"],
                "narrative_arc": case["arc"],
                "seed": seed,
                "execution_order": list(order),
                "control_signature": render_signature,
                "technical_both_passed": bool(
                    variants.get("A", {}).get("evidence", {}).get("technical_pass")
                    and variants.get("B", {}).get("evidence", {}).get("technical_pass")
                ),
                "creative_winner": "undecided_until_manual_review",
                "variants": variants,
            })
            if not variants[variant].get("evidence", {}).get("technical_pass"):
                failed_pair = {
                    "case_id": case["case_id"],
                    "narrative_arc": case["arc"],
                    "seed": seed,
                    "execution_order": list(order),
                    "control_signature": render_signature,
                    "technical_both_passed": False,
                    "creative_winner": "not_scored_technical_failure",
                    "variants": variants,
                }
                pairs.append(failed_pair)
                passed_pairs = sum(pair["technical_both_passed"] for pair in pairs)
                _json_write(run_dir / "benchmark_summary.json", {
                    **run_config,
                    "pairs_completed": passed_pairs,
                    "technical_all_pairs_passed": False,
                    "pairs": pairs,
                    "adoption_decision": "not_adopt_due_to_technical_failures",
                })
                print(f"Evidence: {run_dir}", flush=True)
                print(f"Technical A/B pairs passed: {passed_pairs}/{len(pairs)}", flush=True)
                print("Stopped after a technical failure; no remaining variants were submitted.", flush=True)
                return 2
        pair = {
            "case_id": case["case_id"],
            "narrative_arc": case["arc"],
            "seed": seed,
            "execution_order": list(order),
            "control_signature": render_signature,
            "technical_both_passed": bool(
                variants.get("A", {}).get("evidence", {}).get("technical_pass")
                and variants.get("B", {}).get("evidence", {}).get("technical_pass")
            ),
            "creative_winner": "undecided_until_manual_review",
            "variants": variants,
        }
        pairs.append(pair)
        _json_write(run_dir / "benchmark_summary.json", {**run_config, "pairs_completed": len(pairs), "pairs": pairs})
        _require_idle_comfy(args.comfy_host, int(args.comfy_port))

    technical_pass = all(pair["technical_both_passed"] for pair in pairs)
    _json_write(run_dir / "benchmark_summary.json", {
        **run_config,
        "pairs_completed": len(pairs),
        "technical_all_pairs_passed": technical_pass,
        "pairs": pairs,
        "adoption_decision": "pending_manual_video_review" if technical_pass else "not_adopt_due_to_technical_failures",
    })
    print(f"Evidence: {run_dir}", flush=True)
    print(f"Technical A/B pairs passed: {sum(pair['technical_both_passed'] for pair in pairs)}/{len(pairs)}", flush=True)
    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
