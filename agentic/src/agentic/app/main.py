import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.assets.registry import AssetRegistry
from agentic.memory import PortfolioMemory, RunMemory
from agentic.runtime.creativity import FeedbackRanker, IdeaDirector, RetryPolicy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.step_logger import create_run_logger
from agentic.runtime.story_service import NativeH3StoryService
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.agent_social import register_agent_social_skills
from agentic.skills.comfy_image import register_comfy_image_skills
from agentic.skills.comfy_workflow_skills import register_comfy_workflow_skills
from agentic.skills.editing import register_editing_skills
from agentic.skills.longvideo import register_longvideo_skills
from agentic.skills.reference_video import register_reference_video_skills
from agentic.skills.storyboard import register_storyboard_skills
from agentic.tools.comfy import register_builtin_tools
from agentic.tools.authoring import register_authoring_tools
from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools
from agentic.tools.context_services import NewsContextService
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
    logger=None,
    recorder: RunRecorder | None = None,
    input_roots: Iterable[Path] = (),
) -> tuple[TaskPlanner, WorkflowRunner, RunMemory]:
    if recorder is None and logger is not None:
        recorder = getattr(logger, "run_recorder", None)
    if run_id and logger is None:
        logger, _ = create_run_logger(root / "logs" / "runs", run_id, recorder=recorder)
        recorder = getattr(logger, "run_recorder", recorder)
    asset_registry = AssetRegistry(root, asset_root=_resolve_comfy_root(root, comfy_root))
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    run_memory = RunMemory()
    portfolio_dir = root / "logs"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    portfolio_memory = PortfolioMemory(portfolio_dir / "agentic_portfolio.jsonl")
    resolved_output_root = output_root or root / "output"
    _allow_runtime_output_for_visual_evidence(resolved_output_root)
    llm_engine = LLMPromptEngine(
        mode=os.environ.get("AGENTIC_LLM_MODE", "llm"),
        recorder=recorder,
    )
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
    register_media_service_tools(tool_registry, resolved_output_root, input_roots=input_roots)
    register_social_service_tools(tool_registry, resolved_output_root)
    register_agent_primitive_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=prompt_engine)
    register_agent_social_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=prompt_engine)
    register_comfy_image_skills(skill_registry, tool_registry, resolved_output_root)
    register_comfy_workflow_skills(skill_registry, tool_registry, resolved_output_root)
    register_editing_skills(
        skill_registry,
        tool_registry,
        resolved_output_root,
        prompt_engine=prompt_engine,
    )
    register_longvideo_skills(
        skill_registry,
        tool_registry,
        resolved_output_root,
        prompt_engine=prompt_engine,
        story_service=NativeH3StoryService(
            llm_engine=llm_engine,
            news_service=NewsContextService(),
        ),
    )
    register_reference_video_skills(skill_registry, tool_registry, resolved_output_root)
    register_storyboard_skills(skill_registry, tool_registry)

    idea_director = IdeaDirector()
    retry_policy = RetryPolicy(max_attempts=3)
    feedback_ranker = FeedbackRanker()
    planner = TaskPlanner(asset_registry=asset_registry, idea_director=idea_director)
    runner = WorkflowRunner(
        skill_registry=skill_registry,
        run_memory=run_memory,
        portfolio_memory=portfolio_memory,
        retry_policy=retry_policy,
        feedback_ranker=feedback_ranker,
        logger=logger,
        recorder=recorder,
    )
    runner.tool_registry = tool_registry
    return planner, runner, run_memory


def _allow_runtime_output_for_visual_evidence(output_root: Path) -> None:
    """Allow the app's own generated media root to reach vision fallbacks.

    Vision providers intentionally reject arbitrary filesystem paths. The
    runtime output directory is an explicit app-owned root, so registering it
    here keeps caption/review fallback functional even when ``--output-dir``
    points outside the repository (for example, a dedicated D: drive).
    """

    resolved = output_root.expanduser().resolve()
    configured = [
        Path(raw.strip()).expanduser().resolve()
        for raw in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if raw.strip()
    ]
    if not any(resolved == root or root in resolved.parents for root in configured):
        configured.append(resolved)
        os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(str(root) for root in configured)


def _resolve_comfy_root(root: Path, explicit_root: Path | None) -> Path:
    if explicit_root:
        return explicit_root.expanduser().resolve()
    configured_root = os.environ.get("COMFYUI_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    container_root = Path("/comfyui")
    if container_root.is_dir():
        return container_root.resolve()
    portable_root = Path(r"D:\ComfyUI_windows_portable")
    if portable_root.is_dir():
        return portable_root.resolve()
    return root.parent.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic media runtime")
    parser.add_argument("--goal", required=True, help="High-level goal for the runtime")
    parser.add_argument("--media-type", default="long_video", help="Target media type")
    parser.add_argument("--duration-seconds", type=int, default=30, help="Target duration")
    parser.add_argument("--style", default="cinematic surreal", help="Preferred visual style")
    parser.add_argument("--execute", action="store_true", help="Execute the generated plan")
    parser.add_argument("--auto-download-assets", action="store_true", help="Allow auto asset preparation")
    parser.add_argument("--output-dir", help="Optional output directory for artifact-producing workflows")
    parser.add_argument("--width", type=int, help="Output video width in pixels for image-to-video workflows")
    parser.add_argument("--height", type=int, help="Output video height in pixels for image-to-video workflows")
    parser.add_argument("--longvideo-steps", type=int, help="Sampling steps for long-video H3 workflows")
    parser.add_argument("--comfy-host", help="Override ComfyUI host")
    parser.add_argument("--comfy-port", type=int, help="Override ComfyUI port")
    parser.add_argument(
        "--comfy-root",
        default=os.environ.get("COMFYUI_ROOT", r"D:\ComfyUI_windows_portable"),
        help="ComfyUI root for asset checks (default: COMFYUI_ROOT or D:\\ComfyUI_windows_portable)",
    )
    parser.add_argument("--input-image", help="Input image path for img2img/i2v/upscale workflows")
    parser.add_argument("--input-video", help="Input video path for frame extraction workflows")
    parser.add_argument(
        "--reference-video",
        help="Reference video path or URL used to extract pacing, shot, and motion evidence for remix planning",
    )
    parser.add_argument(
        "--reference-video-depth",
        choices=("standard", "deep"),
        default="standard",
        help="Reference-video analysis depth (both modes currently use local structural evidence; deep is reserved for richer analysis)",
    )
    parser.add_argument(
        "--reference-keyframes",
        type=int,
        default=12,
        help="Number of evenly spaced reference keyframes to extract (2-20)",
    )
    parser.add_argument("--input-dir", help="Input directory for publish/review workflows")
    parser.add_argument("--media-path", action="append", dest="media_paths", help="Explicit media path for publish/review workflows; repeatable")
    parser.add_argument("--edit-input", action="append", dest="edit_input_paths", help="Ordered image/video input for image_sequence_edit; repeatable")
    parser.add_argument("--edit-input-root", action="append", dest="edit_input_roots", help="Approved root for image_sequence_edit inputs; repeatable")
    parser.add_argument("--drama-plan", help="JSON DramaPlan file for image_sequence_edit; compiles scenes into the deterministic timeline editor")
    parser.add_argument(
        "--edit-profile",
        choices=("baseline_concat", "motion_cut_v1", "xfade_clean_v1", "chapter_dip_v1", "editorial_kinetic_v1"),
        help="Timeline profile for image_sequence_edit",
    )
    parser.add_argument("--edit-transition-duration", type=float, help="Transition overlap in seconds for timeline editing")
    parser.add_argument("--edit-variant-seed", type=int, help="Deterministic variation seed for timeline editing")
    parser.add_argument("--edit-require-audio", action="store_true", help="Require an audio stream in image_sequence_edit QA")
    parser.add_argument("--edit-analyze-audio", action="store_true", help="Run loudness/silence analysis in image_sequence_edit QA")
    parser.add_argument("--edit-creative-review", action="store_true", help="Run a blocking vision-LLM creative review loop for edit candidates")
    parser.add_argument("--edit-creative-review-max-attempts", type=int, help="Maximum deterministic edit candidates reviewed by the vision LLM (1-4)")
    parser.add_argument(
        "--longvideo-edit-profile",
        choices=("baseline_concat", "xfade_clean_v1", "chapter_dip_v1", "editorial_kinetic_v1"),
        help="Use the agent timeline editor to merge non-TTS long-video segments",
    )
    parser.add_argument(
        "--longvideo-production-profile",
        choices=("text2longvideo",),
        help="Use deterministic multi-beat H3 story assembly with rendered-tail continuity and a publish package",
    )
    parser.add_argument("--text", help="Text payload for TTS-style workflows")
    parser.add_argument("--character", help="Optional character or subject identity for agentic planning")
    parser.add_argument("--platform", action="append", dest="platforms", help="Target publish platform; repeatable")
    parser.add_argument(
        "--publish-mode",
        choices=("live", "safe_poc"),
        default="",
        help="Publishing policy: live or safe_poc (YouTube private, Facebook draft, Instagram container-only)",
    )
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
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    planner, runner, run_memory = build_runtime(
        project_root,
        output_root=output_root,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=Path(args.comfy_root).resolve() if args.comfy_root else None,
        input_roots=[Path(root) for root in args.edit_input_roots or []],
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
            "reference_video_source": args.reference_video,
            "reference_video_depth": args.reference_video_depth,
            "reference_video_max_keyframes": args.reference_keyframes,
            "input_dir": args.input_dir,
            "media_paths": args.media_paths or [],
            "edit_input_paths": args.edit_input_paths or [],
            "drama_plan_source": args.drama_plan,
            "edit_profile": args.edit_profile,
            "edit_transition_duration": args.edit_transition_duration,
            "edit_variant_seed": args.edit_variant_seed,
            "edit_require_audio": args.edit_require_audio,
            "edit_analyze_audio": args.edit_analyze_audio,
            "edit_creative_review": args.edit_creative_review,
            "edit_creative_review_max_attempts": args.edit_creative_review_max_attempts,
            "longvideo_edit_profile": args.longvideo_edit_profile,
            "longvideo_production_profile": args.longvideo_production_profile,
            "longvideo_steps": args.longvideo_steps,
            "text": args.text,
            "character": args.character,
            "platforms": args.platforms or [],
            "publish_mode": args.publish_mode,
            "hashtags": args.hashtags or [],
            "caption_prefix": args.caption_prefix,
            "selection_limit": args.selection_limit,
            "review_notes": args.review_notes,
            "enable_review_loop": args.enable_review_loop,
            "dry_run": args.dry_run_publish,
            "output_dir": str(output_root) if output_root else None,
            "width": args.width,
            "height": args.height,
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
