"""Run the provider-neutral long-video mix through real ComfyUI.

This is an execution-level smoke test for the shared conditioning planner. It
does not publish, does not use Discord review, and never substitutes another
recipe when a selected recipe fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = REPO_ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.app.main import build_runtime


MIX_WEIGHTS = {
    "anchor_first": 1,
    "anchor_first_last": 3,
    "anchor_last": 2,
    "reference_bundle": 2,
}
EXPECTED_REFERENCE_CANDIDATES = 4


def _qa(tools: Any, video_path: str, output_root: Path, *, width: int, height: int) -> dict[str, Any]:
    contact_sheet = output_root / "qa" / f"{Path(video_path).stem}_contact_sheet.jpg"
    return tools.call(
        "media.video_qa",
        {
            "video_path": video_path,
            "expected_width": width,
            "expected_height": height,
            "expected_fps": 24,
            "fps_tolerance": 0.2,
            "require_audio": True,
            "require_stereo_audio": True,
            "analyze_audio": True,
            "warn_if_no_audio": True,
            "contact_sheet_path": str(contact_sheet),
            "frame_count": 8,
            "columns": 4,
            "scale_width": 320,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared long-video conditioning mix through ComfyUI")
    parser.add_argument("--comfy-root", default=r"D:\ComfyUI_windows_portable")
    parser.add_argument("--output-root", default=r"D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_longvideo_mix_e2e")
    parser.add_argument("--comfy-host", default="127.0.0.1")
    parser.add_argument("--comfy-port", type=int, default=8188)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--length", type=int, default=81)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Use a short 17-frame, 8-step render")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    length = 17 if args.smoke else args.length
    steps = 8 if args.smoke else args.steps
    planner, runner, _memory = build_runtime(
        REPO_ROOT / "agentic",
        output_root=output_root,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=comfy_root,
    )
    goal = planner.create_goal(
        prompt=(
            "Kirby crosses a windy meadow, discovers a glowing seed, protects it from a storm, "
            "and reaches a warm clearing with a clear visual payoff."
        ),
        media_type="long_video",
        duration_seconds=max(10, args.segments * 5),
        style="polished 2D anime cinematic, clear silhouette, coherent camera motion",
        auto_download_assets=False,
        constraints={
            "character": "Kirby",
            "segment_count": max(2, args.segments),
            "longvideo_mix_weights": dict(MIX_WEIGHTS),
            "longvideo_mix_seed": args.seed,
            "longvideo_review_policy": "opening_only",
            "longvideo_width": args.width,
            "longvideo_height": args.height,
            "longvideo_length": length,
            "longvideo_steps": steps,
            "longvideo_model_profile": "q2",
            "longvideo_frame_candidate_count": 1,
            "longvideo_reference_candidate_count": 4,
            "longvideo_reference_selection_limit": 4,
            "require_human_review": False,
            "enable_review_loop": False,
            "use_tts": False,
        },
    )
    plan = planner.build_plan(goal)
    report: dict[str, Any] = {
        "recipe_sequence": plan.metadata.get("recipe_sequence", []),
        "recipe_workflows": plan.metadata.get("recipe_workflows", {}),
        "mix_seed": plan.metadata.get("mix_seed"),
        "mix_weights": plan.metadata.get("mix_weights", {}),
        "output_root": str(output_root),
        "smoke": args.smoke,
    }
    result = runner.run(plan)
    report["workflow_status"] = result.status
    report["segment_results"] = []
    for record in result.records:
        if not record.node_id.startswith("segment-video-"):
            continue
        saved_files = [str(path) for path in record.outputs.get("saved_files", []) if path]
        item: dict[str, Any] = {
            "node_id": record.node_id,
            "status": record.status,
            "attempt": record.attempt,
            "recipe": record.outputs.get("recipe"),
            "workflow_name": record.outputs.get("workflow_name"),
            "memory_retry_count": record.outputs.get("memory_retry_count"),
            "video_path": saved_files[0] if saved_files else "",
        }
        if record.outputs.get("recipe") == "reference_bundle":
            references = record.outputs.get("reference_manifest") or []
            reference_types = [
                str(reference.get("type") or "")
                for reference in references
                if isinstance(reference, dict)
            ]
            item["reference_manifest_count"] = len(references)
            item["reference_mode"] = "+".join(
                f"{reference_types.count(reference_type)} {reference_type}"
                for reference_type in ("image", "video")
                if reference_types.count(reference_type)
            )
            item["reference_contract_passed"] = len(references) == EXPECTED_REFERENCE_CANDIDATES
            if not item["reference_contract_passed"]:
                item["reference_contract_error"] = (
                    f"expected {EXPECTED_REFERENCE_CANDIDATES} reference inputs, received {len(references)}"
                )
        if record.logs:
            item["logs"] = list(record.logs)
        if record.metrics:
            item["metrics"] = dict(record.metrics)
        if saved_files:
            item["qa"] = _qa(runner.tool_registry, saved_files[0], output_root, width=args.width, height=args.height)
        report["segment_results"].append(item)

    final_output = result.state.node_outputs.get("concat-final-video", {})
    final_path = str(final_output.get("video_path") or "") if isinstance(final_output, dict) else ""
    report["final_video_path"] = final_path
    report["final_qa"] = _qa(runner.tool_registry, final_path, output_root, width=args.width, height=args.height) if final_path else {"passed": False, "errors": ["missing final video"]}
    report["passed"] = bool(
        result.status == "success"
        and report["segment_results"]
        and all(item.get("qa", {}).get("passed") for item in report["segment_results"])
        and all(item.get("reference_contract_passed", True) for item in report["segment_results"])
        and report["final_qa"].get("passed") is True
    )
    report_path = output_root / "longvideo_mix_e2e_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
