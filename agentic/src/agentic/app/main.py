import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.assets.registry import AssetRegistry
from agentic.memory import PortfolioMemory, RunMemory
from agentic.runtime.creativity import FeedbackRanker, IdeaDirector, RetryPolicy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.step_logger import create_run_logger
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.agent_social import register_agent_social_skills
from agentic.skills.comfy_image import register_comfy_image_skills
from agentic.skills.comfy_workflow_skills import register_comfy_workflow_skills
from agentic.skills.longvideo import register_longvideo_skills
from agentic.skills.storyboard import register_storyboard_skills
from agentic.tools.comfy import register_builtin_tools
from agentic.tools.authoring import register_authoring_tools
from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools
from agentic.tools.local import register_local_tools
from agentic.tools.media_services import register_media_service_tools
from agentic.tools.social_services import register_social_service_tools


def build_runtime(
    root: Path,
    output_root: Path | None = None,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
    comfy_root: Path | None = None,
    run_id: str | None = None,
) -> tuple[TaskPlanner, WorkflowRunner, RunMemory]:
    asset_registry = AssetRegistry(
        root / "configs" / "workflow_manifests",
        root,
        asset_root=comfy_root or root.parent,
    )
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    run_memory = RunMemory()
    portfolio_dir = root / "logs"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    portfolio_memory = PortfolioMemory(portfolio_dir / "agentic_portfolio.jsonl")
    resolved_output_root = output_root or root / "output"
    llm_engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    prompt_engine = PromptEngine(llm_engine=llm_engine)

    register_builtin_tools(tool_registry, asset_registry)
    register_authoring_tools(tool_registry, asset_registry, root)
    register_local_tools(tool_registry, resolved_output_root)
    register_comfy_workflow_tools(
        tool_registry,
        asset_registry,
        resolved_output_root,
        comfy_host=comfy_host,
        comfy_port=comfy_port,
    )
    register_media_service_tools(tool_registry, resolved_output_root)
    register_social_service_tools(tool_registry, resolved_output_root)
    register_agent_primitive_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=prompt_engine)
    register_agent_social_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=prompt_engine)
    register_comfy_image_skills(skill_registry, tool_registry, resolved_output_root)
    register_comfy_workflow_skills(skill_registry, tool_registry, resolved_output_root)
    register_longvideo_skills(skill_registry, tool_registry, resolved_output_root)
    register_storyboard_skills(skill_registry, tool_registry)

    idea_director = IdeaDirector()
    retry_policy = RetryPolicy(max_attempts=3)
    feedback_ranker = FeedbackRanker()
    logger = None
    if run_id:
        logger, _ = create_run_logger(root / "logs" / "runs", run_id)

    planner = TaskPlanner(asset_registry=asset_registry, idea_director=idea_director)
    runner = WorkflowRunner(
        skill_registry=skill_registry,
        run_memory=run_memory,
        portfolio_memory=portfolio_memory,
        retry_policy=retry_policy,
        feedback_ranker=feedback_ranker,
        logger=logger,
    )
    runner.tool_registry = tool_registry
    return planner, runner, run_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic media runtime")
    parser.add_argument("--goal", required=True, help="High-level goal for the runtime")
    parser.add_argument("--media-type", default="long_video", help="Target media type")
    parser.add_argument("--duration-seconds", type=int, default=30, help="Target duration")
    parser.add_argument("--style", default="cinematic surreal", help="Preferred visual style")
    parser.add_argument("--execute", action="store_true", help="Execute the generated plan")
    parser.add_argument("--auto-download-assets", action="store_true", help="Allow auto asset preparation")
    parser.add_argument("--output-dir", help="Optional output directory for artifact-producing workflows")
    parser.add_argument("--comfy-host", help="Override ComfyUI host")
    parser.add_argument("--comfy-port", type=int, help="Override ComfyUI port")
    parser.add_argument("--comfy-root", help="Override ComfyUI root for asset checks")
    parser.add_argument("--input-image", help="Input image path for img2img/i2v/upscale workflows")
    parser.add_argument("--input-video", help="Input video path for frame extraction workflows")
    parser.add_argument("--input-dir", help="Input directory for publish/review workflows")
    parser.add_argument("--media-path", action="append", dest="media_paths", help="Explicit media path for publish/review workflows; repeatable")
    parser.add_argument("--text", help="Text payload for TTS-style workflows")
    parser.add_argument("--character", help="Optional character or subject identity for agentic planning")
    parser.add_argument("--platform", action="append", dest="platforms", help="Target publish platform; repeatable")
    parser.add_argument("--hashtag", action="append", dest="hashtags", help="Hashtag for publish captions; repeatable")
    parser.add_argument("--caption-prefix", help="Optional caption prefix for publish workflows")
    parser.add_argument("--selection-limit", type=int, help="Review selection cap for publish/review workflows")
    parser.add_argument("--review-notes", help="Optional review feedback used to trigger refine/regenerate branches")
    parser.add_argument("--enable-review-loop", action="store_true", help="Force planner retry/review branches where supported")
    parser.add_argument("--dry-run-publish", action="store_true", help="Keep publish/review dispatch in dry-run mode")
    parser.add_argument("--use-tts", action="store_true", help="Enable TTS generation for long-video workflows")
    return parser.parse_args()


def _build_prompt_summary(state: dict[str, object]) -> dict[str, object]:
    node_prompt_modes = state.get("node_prompt_modes", {})
    prompt_lineage = state.get("prompt_lineage", [])
    fallback_nodes: dict[str, str] = {}
    llm_backends: dict[str, object] = {}
    if isinstance(prompt_lineage, list):
        for entry in prompt_lineage:
            if not isinstance(entry, dict):
                continue
            node_id = str(entry.get("node_id", "")).strip()
            if not node_id:
                continue
            fallback_reason = entry.get("fallback_reason")
            if isinstance(fallback_reason, str) and fallback_reason:
                fallback_nodes[node_id] = fallback_reason
            llm_backend = entry.get("llm_backend")
            if llm_backend is not None:
                llm_backends[node_id] = llm_backend
    return {
        "node_prompt_modes": node_prompt_modes if isinstance(node_prompt_modes, dict) else {},
        "prompt_lineage": prompt_lineage if isinstance(prompt_lineage, list) else [],
        "fallback_nodes": fallback_nodes,
        "llm_backends": llm_backends,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    planner, runner, run_memory = build_runtime(
        project_root,
        output_root=output_root,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=Path(args.comfy_root).resolve() if args.comfy_root else None,
    )

    goal = planner.create_goal(
        prompt=args.goal,
        media_type=args.media_type,
        duration_seconds=args.duration_seconds,
        style=args.style,
        auto_download_assets=args.auto_download_assets,
        constraints={
            "input_image_path": args.input_image,
            "input_video_path": args.input_video,
            "input_dir": args.input_dir,
            "media_paths": args.media_paths or [],
            "text": args.text,
            "character": args.character,
            "platforms": args.platforms or [],
            "hashtags": args.hashtags or [],
            "caption_prefix": args.caption_prefix,
            "selection_limit": args.selection_limit,
            "review_notes": args.review_notes,
            "enable_review_loop": args.enable_review_loop,
            "dry_run": args.dry_run_publish,
            "output_dir": str(output_root) if output_root else None,
            "use_tts": args.use_tts,
        },
    )
    plan = planner.build_plan(goal)

    print("=== PLAN ===")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))

    if not args.execute:
        return

    print("=== RUN ===")
    result = runner.run(plan)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("=== PROMPTS ===")
    print(json.dumps(_build_prompt_summary(result.to_dict().get("state", {})), indent=2, ensure_ascii=False))
    print("=== MEMORY ===")
    print(json.dumps(run_memory.as_serializable(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
