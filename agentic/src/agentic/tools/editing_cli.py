from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.drama import DramaPlan, compile_drama_plan
from agentic.runtime.editing import EditPlan, build_edit_plan
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import ToolRegistry
from agentic.skills.editing import EditingSkills
from agentic.tools.media_services import register_media_service_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an agent-controlled MediaOverload edit timeline")
    parser.add_argument("--input", action="append", dest="inputs", help="Ordered image or video input; repeatable")
    parser.add_argument("--input-root", action="append", dest="input_roots", help="Approved root for edit inputs; repeatable")
    parser.add_argument("--edit-plan", help="JSON file containing an EditPlan")
    parser.add_argument("--drama-plan", help="JSON file containing a DramaPlan; scenes are compiled into an EditPlan")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument(
        "--profile",
        choices=("baseline_concat", "xfade_clean_v1", "chapter_dip_v1", "editorial_kinetic_v1"),
        default="xfade_clean_v1",
    )
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--target-duration", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transition-duration", type=float, default=0.10)
    parser.add_argument("--goal", default="Create an engaging short-form edit from generated media")
    parser.add_argument("--style", default="fashion short-form")
    parser.add_argument("--creative-review", action="store_true", help="Run the blocking vision-LLM creative review loop")
    parser.add_argument("--creative-review-max-attempts", type=int, default=3, help="Maximum creative-review candidates (1-4)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.edit_plan and args.drama_plan:
        raise ValueError("Use either --edit-plan or --drama-plan, not both")
    raw_drama_plan = None
    if args.drama_plan:
        payload = json.loads(Path(args.drama_plan).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Drama plan JSON must contain an object")
        drama_plan = DramaPlan.from_dict(payload)
        plan = compile_drama_plan(drama_plan)
        raw_drama_plan = drama_plan.to_dict()
    elif args.edit_plan:
        payload = json.loads(Path(args.edit_plan).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Edit plan JSON must contain an object")
        plan = EditPlan.from_dict(payload)
    else:
        if not args.inputs:
            raise ValueError("At least one --input is required when --edit-plan or --drama-plan is not supplied")
        plan = build_edit_plan(
            args.inputs,
            profile=args.profile,
            output_width=args.width,
            output_height=args.height,
            fps=args.fps,
            target_duration_seconds=args.target_duration,
            variant_seed=args.seed,
            transition_duration_seconds=args.transition_duration,
        )
    output = Path(args.output).expanduser().resolve()
    review_enabled = args.creative_review or plan.profile == "editorial_kinetic_v1"
    tool_registry = ToolRegistry()
    register_media_service_tools(
        tool_registry,
        output.parent,
        input_roots=[Path(root) for root in args.input_roots or []],
    )
    prompt_engine = PromptEngine(
        llm_engine=LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    ) if review_enabled else PromptEngine()
    skills = EditingSkills(tool_registry, output.parent, prompt_engine=prompt_engine)
    goal = GoalRequest(prompt=args.goal, media_type="image_sequence_edit", duration_seconds=0, style=args.style)
    node = ExecutionNode(
        node_id="compose-edit",
        skill_name="media.video.compose_timeline",
        inputs={
            "output_path": str(output),
            "contact_sheet_path": str(output.with_suffix(".contact_sheet.jpg")),
            "manifest_path": str(output.with_suffix(".edit_manifest.json")),
            "creative_review": review_enabled,
            "creative_review_max_attempts": args.creative_review_max_attempts,
            "require_audio": plan.profile != "baseline_concat",
            "require_stereo_audio": plan.profile != "baseline_concat",
        },
    )
    if raw_drama_plan is not None:
        node.inputs["drama_plan"] = raw_drama_plan
        node.inputs["drama_plan_path"] = str(output.with_suffix(".drama_plan.json"))
    else:
        node.inputs["edit_plan"] = plan.to_dict()
    context = SkillContext(
        plan=ExecutionPlan(goal=goal, workflow_name="image_sequence_edit_v1", nodes=[node]),
        node=node,
        state=RunState(goal={}, metadata={}),
    )
    result = skills.compose_timeline(context)
    print(json.dumps({"status": result.status, "outputs": result.outputs, "metrics": result.metrics, "logs": result.logs}, indent=2, ensure_ascii=False))
    if result.status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
