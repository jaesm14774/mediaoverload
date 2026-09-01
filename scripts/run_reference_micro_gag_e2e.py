"""Run the collection-informed short-video optimization through the existing flow.

This is an execution benchmark, not a second rendering pipeline. Every case
uses ``run_character_workflow`` with the existing ``text2image2video`` route:
reference-video analysis -> LLM brief -> Krea2 first frame -> MiniMax H3 I2V
-> technical/semantic QA. Retries change only the prompt direction and keep a
stable per-case seed so the result is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
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


CASE_PROMPTS: tuple[str, ...] = (
    "Kirby-like pink hero in a bright tabletop kitchen, already reaching toward one oversized lunchbox lid as a tiny snack rolls away; the lid snaps shut, the hero bounces back, catches the snack, and settles into a proud loopable pose.",
    "Kirby-like pink hero beside one steaming bowl on a clean pastel counter, already tugging a stubborn noodle with both hands; the noodle suddenly slurps free, the hero wobbles backward, then lands with the noodle neatly wrapped like a silly scarf.",
    "Kirby-like pink hero on a soft painted picnic blanket, already leaning into one giant soap bubble; the bubble stretches around the hero, pops with a harmless wobble, and leaves the hero blinking in the same opening direction for a playful loop.",
    "Kirby-like pink hero on a minimal white stage, already squeezing one springy pink cushion; the cushion rebounds into the hero, the body squashes and recoils, and the hero ends hugging it with a clear delighted expression.",
    "Kirby-like pink hero at a cozy dessert table, already poking one oversized mochi with a wooden skewer; the mochi rolls away, bumps the hero's feet, and returns as a tiny hat while the hero reacts with a readable surprised bounce.",
    "Kirby-like pink hero in a sunny meadow, already swinging one small red toy bat toward a round pebble; the contact creates a cute puff, the hero recoils, and the pebble lands beside the hero in a settled comedic result.",
    "Kirby-like pink hero on a colorful game-cartridge-shaped platform with no writing, already reaching for one glowing star-shaped button; a star wipe reveals the button underneath, the hero pops into view, and the final pose mirrors the opening reach.",
    "Kirby-like pink hero underwater beside one oversized coral shell, already pushing it open; a burst of bubbles lifts the hero, the shell reveals a harmless pearl, and the hero floats back into a calm loopable finish.",
    "Kirby-like pink hero in a clean studio with one oversized costume cape, already tangled in the cape while trying to turn; the cape spins the hero once, flops into place, and ends with a proud tiny superhero pose.",
    "Kirby-like pink hero at a cozy food stall tabletop, already balancing one rice ball on a spoon; the rice ball slips, the hero catches it against the belly, and the final pose echoes the opening balance with a triumphant wobble.",
)


REFERENCE_SINGLE_PROTAGONIST_CONTRACT = (
    "Reference-derived micro-gag contract: show exactly one visible selected protagonist, "
    "with no duplicate, clone, reflection, miniature copy, background character, or second "
    "version of the protagonist. Keep the tabletop uncluttered so the one physical gag is "
    "immediately readable."
)


def stable_seed(case_id: str, seed_base: int) -> int:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % 100_000
    return max(1, min(2_147_483_646, int(seed_base) + offset))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 10 collection-informed short videos through the existing MediaOverload route"
    )
    parser.add_argument(
        "--collection-root",
        default=r"C:\Users\jaesm14774\Downloads\收集",
        help="Folder containing reference videos",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "output" / "reference_micro_gag_e2e"),
    )
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=5,
        choices=range(4, 10),
        help="Requested short clip duration; default 5 matches the existing H3 124-frame I2V profile",
    )
    parser.add_argument("--reference-depth", choices=("standard", "deep"), default="deep")
    parser.add_argument("--reference-keyframes", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=20260831)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        choices=range(0, 4),
        help="Maximum prompt rewrites after the initial attempt (0-3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        choices=range(1, 11),
        help="Number of benchmark cases; default and acceptance target are 10",
    )
    return parser.parse_args()


def _load_sources(collection_root: Path) -> list[Path]:
    if not collection_root.is_dir():
        raise SystemExit(f"Collection folder does not exist: {collection_root}")
    sources = sorted(
        path.resolve()
        for path in collection_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    )
    if not sources:
        raise SystemExit(f"No supported reference videos found in: {collection_root}")
    return sources


def _case_sources(sources: list[Path], limit: int) -> list[Path]:
    selected = list(sources)
    # The collection has nine clips; use the strongest tactile-gag source as a
    # second, independently prompted case rather than silently producing only
    # nine videos. The source contributes grammar, never source assets or plot.
    strongest_index = min(4, len(selected) - 1)
    while len(selected) < limit:
        selected.append(selected[strongest_index])
    return selected[:limit]


def _retry_direction(qa: dict[str, Any], attempt: int, failure_reason: str = "") -> str:
    semantic = qa.get("semantic_qa") if isinstance(qa.get("semantic_qa"), dict) else {}
    issues = [str(item).strip() for item in semantic.get("issues", []) if str(item).strip()]
    checks = semantic.get("checks") if isinstance(semantic.get("checks"), dict) else {}
    failed_checks = [
        key
        for key, value in checks.items()
        if value is False and key != "news_anchor_visible"
    ]
    technical = [str(item).strip() for item in qa.get("errors", []) if str(item).strip()]
    failure_text = str(failure_reason or "").strip()
    details = [*issues, *(f"failed visual check: {key}" for key in failed_checks), *technical]
    if failure_text:
        details.append(failure_text)
    if not details:
        details = [
            "make the opening action visible immediately",
            "show one decisive physical cause-and-effect change",
            "finish with a readable cute reaction and settled payoff",
        ]
    unique = list(dict.fromkeys(details))[:5]
    if "stage_probe_quality_gate" in failure_text or "asset_review_hard_gate" in failure_text:
        unique.insert(0, "the pre-video image gate rejected the candidates; enforce exactly one visible protagonist with no duplicate or extra character")
        unique = list(dict.fromkeys(unique))[:5]
    return (
        f"Retry direction for attempt {attempt}: preserve the same protagonist, palette, and one-prop gag. "
        "Change only the weakest visual lever and keep the reference as timing/framing grammar, not copied content. "
        f"Fix these observed problems: {'; '.join(unique)}. "
        "The first frame must already contain the hook, and the final sampled frames must show the completed physical result."
    )


def _extract_attempt_evidence(result: dict[str, Any]) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
    generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
    state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
    node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
    qa = node_outputs.get("video-qa") if isinstance(node_outputs.get("video-qa"), dict) else {}
    semantic = qa.get("semantic_qa") if isinstance(qa.get("semantic_qa"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    video_paths = [
        str(path)
        for path in (artifacts.get("video_paths") or [])
        if Path(str(path)).is_file()
    ]
    technical_pass = bool(qa.get("passed")) and bool(video_paths)
    semantic_status = str(semantic.get("status") or "unavailable")
    semantic_pass = semantic.get("passed") is True
    # If the vision backend is unavailable, technical QA plus later human
    # contact-sheet inspection is the honest acceptance boundary. Never turn
    # an explicit semantic fail into a pass.
    accepted = technical_pass and (semantic_pass or semantic_status == "unavailable")
    return {
        "workflow_status": str(result.get("status") or "failed"),
        "technical_pass": technical_pass,
        "semantic_pass": semantic_pass,
        "semantic_status": semantic_status,
        "semantic_score": semantic.get("score"),
        "semantic_checks": dict(semantic.get("checks") or {}) if isinstance(semantic.get("checks"), dict) else {},
        "semantic_issues": [str(item) for item in semantic.get("issues", []) if str(item)],
        "contact_sheet_path": str(qa.get("contact_sheet_path") or semantic.get("contact_sheet_path") or ""),
        "video_paths": video_paths,
        "accepted_before_manual_review": accepted,
        "failure_reason": str(result.get("failure_reason") or ""),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_case(
    *,
    case_id: str,
    source: Path,
    base_prompt: str,
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = stable_seed(case_id, args.seed_base)
    attempts: list[dict[str, Any]] = []
    prompt = f"{base_prompt}\n\n{REFERENCE_SINGLE_PROTAGONIST_CONTRACT}"
    for attempt in range(1, int(args.max_retries) + 2):
        attempt_dir = case_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        prompt_record = {
            "case_id": case_id,
            "attempt": attempt,
            "prompt": prompt,
            "reference_video": str(source),
            "reference_policy": "borrow_grammar_not_assets",
            "generation_type": "text2image2video",
            "workflow_profile": "reference_micro_gag_v1",
            "duration_seconds": int(args.duration_seconds),
            "seed": seed,
        }
        _write_json(attempt_dir / "prompt.json", prompt_record)
        try:
            result = run_character_workflow(
                CharacterWorkflowRequest(
                    repo_root=REPO_ROOT,
                    config_path=REPO_ROOT / "configs" / "characters" / "kirby.yaml",
                    generation=CharacterGenerationOptions(
                        prompt=prompt,
                        preferred_generation_type="text2image2video",
                        duration_seconds=int(args.duration_seconds),
                        output_dir=str(attempt_dir),
                        reference_video_source=str(source),
                        reference_video_depth=args.reference_depth,
                        reference_video_max_keyframes=int(args.reference_keyframes),
                        # Keep selection stable while a retry changes only
                        # the prompt direction; image/video seeds are passed
                        # separately to the renderers below.
                        rng=random.Random(seed),
                        seed=seed,
                    ),
                    review=CharacterReviewOptions(
                        publish_after_generate=False,
                        no_review=False,
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
            evidence = _extract_attempt_evidence(result)
            _write_json(attempt_dir / "workflow_result.json", result)
            attempt_record = {"attempt": attempt, "prompt": prompt_record, "evidence": evidence}
            attempts.append(attempt_record)
            _write_json(attempt_dir / "attempt.json", attempt_record)
            if evidence["accepted_before_manual_review"]:
                break
            generation = result.get("generation") if isinstance(result.get("generation"), dict) else {}
            generation_result = generation.get("result") if isinstance(generation.get("result"), dict) else {}
            state = generation_result.get("state") if isinstance(generation_result.get("state"), dict) else {}
            node_outputs = state.get("node_outputs") if isinstance(state.get("node_outputs"), dict) else {}
            qa = node_outputs.get("video-qa") if isinstance(node_outputs.get("video-qa"), dict) else {}
            prompt = f"{base_prompt}\n\n{REFERENCE_SINGLE_PROTAGONIST_CONTRACT}\n\n{_retry_direction(qa, attempt, evidence['failure_reason'])}"
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            attempt_record = {"attempt": attempt, "prompt": prompt_record, "error": error}
            attempts.append(attempt_record)
            _write_json(attempt_dir / "attempt.json", attempt_record)
            prompt = (
                f"{base_prompt}\n\n{REFERENCE_SINGLE_PROTAGONIST_CONTRACT}\n\n"
                f"{_retry_direction({}, attempt, str(exc))}"
            )
    winner = next(
        (record for record in reversed(attempts) if record.get("evidence", {}).get("accepted_before_manual_review")),
        attempts[-1] if attempts else {},
    )
    return {
        "case_id": case_id,
        "source_video": str(source),
        "base_prompt": base_prompt,
        "seed": seed,
        "max_retries": int(args.max_retries),
        "attempt_count": len(attempts),
        "winner_attempt": winner.get("attempt"),
        "accepted_before_manual_review": bool(winner.get("evidence", {}).get("accepted_before_manual_review")),
        "attempts": attempts,
    }


def main() -> int:
    args = parse_args()
    collection_root = Path(args.collection_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output = output_root / run_id
    run_output.mkdir(parents=True, exist_ok=True)
    sources = _case_sources(_load_sources(collection_root), int(args.limit))
    allowed_roots = [collection_root, run_output, REPO_ROOT]
    existing_roots = [
        Path(item.strip()).expanduser().resolve()
        for item in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if item.strip()
    ]
    os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(
        str(path) for path in dict.fromkeys([*existing_roots, *allowed_roots])
    )
    cases: list[dict[str, Any]] = []
    _write_json(
        run_output / "run_config.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_root": str(collection_root),
            "output_root": str(run_output),
            "route": "existing text2image2video -> Krea2 -> MiniMax H3 I2V",
            "workflow_profile": "reference_micro_gag_v1",
            "duration_seconds": int(args.duration_seconds),
            "max_retries": int(args.max_retries),
            "case_count": len(sources),
            "seed_base": int(args.seed_base),
            "publish": False,
        },
    )
    for index, source in enumerate(sources, start=1):
        case_id = f"micro-gag-{index:02d}"
        print(f"[{index}/{len(sources)}] {case_id} | reference={source.name}", flush=True)
        case = _run_case(
            case_id=case_id,
            source=source,
            base_prompt=CASE_PROMPTS[index - 1],
            case_dir=run_output / "cases" / case_id,
            args=args,
        )
        cases.append(case)
        _write_json(run_output / "benchmark_summary.json", {"run_id": run_id, "cases": cases})
        print(
            f"[{index}/{len(sources)}] done | attempts={case['attempt_count']} | "
            f"accepted_before_manual_review={case['accepted_before_manual_review']}",
            flush=True,
        )
    accepted = sum(bool(case.get("accepted_before_manual_review")) for case in cases)
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection_root": str(collection_root),
        "output_root": str(run_output),
        "case_count": len(cases),
        "accepted_before_manual_review": accepted,
        "acceptance_ratio_before_manual_review": round(accepted / max(1, len(cases)), 4),
        "manual_contact_sheet_review_required": True,
        "cases": cases,
    }
    _write_json(run_output / "benchmark_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(cases) == int(args.limit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
