import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.assets.registry import AssetRegistry
from agentic.memory import PortfolioMemory, RunMemory
from agentic.runtime.contracts import (
    DEFAULT_DISCORD_REVIEW_TIMEOUT_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
)
from agentic.runtime.creativity import IdeaDirector, RetryPolicy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.step_logger import create_run_logger
from agentic.runtime.serial_video import SerialVideoStore
from agentic.runtime.story_service import NativeH3StoryService
from agentic.app.character_config import (
    configured_discord_review_timeout_seconds,
    load_character_config,
)
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.agent_social import register_agent_social_skills
from agentic.skills.comfy_image import register_comfy_image_skills
from agentic.skills.comfy_workflow_skills import register_comfy_workflow_skills
from agentic.skills.editing import register_editing_skills
from agentic.skills.longvideo import register_longvideo_skills
from agentic.skills.motion_graphics import register_motion_graphics_skills
from agentic.skills.motion_comic import register_motion_comic_skills
from agentic.skills.reference_video import register_reference_video_skills
from agentic.tools.comfy import register_builtin_tools
from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools
from agentic.tools.comfy_h3 import register_comfy_h3_tools
from agentic.tools.comfy_nvidia_vfx import register_comfy_nvidia_vfx_tools
from agentic.tools.context_services import NewsContextService
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

    register_builtin_tools(tool_registry, asset_registry)
    register_comfy_workflow_tools(
        tool_registry,
        asset_registry,
        resolved_output_root,
        comfy_host=comfy_host,
        comfy_port=comfy_port,
    )
    register_comfy_h3_tools(tool_registry, asset_registry, resolved_output_root, comfy_host, comfy_port)
    register_comfy_nvidia_vfx_tools(tool_registry, comfy_host, comfy_port)
    register_media_service_tools(tool_registry, resolved_output_root, input_roots=input_roots)
    register_social_service_tools(tool_registry, resolved_output_root)
    register_motion_graphics_skills(
        skill_registry,
        tool_registry,
        resolved_output_root,
        prompt_engine=llm_engine,
    )
    register_agent_primitive_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=llm_engine)
    register_motion_comic_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=llm_engine)
    register_agent_social_skills(skill_registry, tool_registry, resolved_output_root, prompt_engine=llm_engine)
    register_comfy_image_skills(skill_registry, tool_registry, resolved_output_root)
    register_comfy_workflow_skills(skill_registry, tool_registry, resolved_output_root)
    register_editing_skills(
        skill_registry,
        tool_registry,
        resolved_output_root,
    )
    register_longvideo_skills(
        skill_registry,
        tool_registry,
        resolved_output_root,
        prompt_engine=llm_engine,
        story_service=NativeH3StoryService(
            llm_engine=llm_engine,
            news_service=NewsContextService(),
        ),
    )
    register_reference_video_skills(skill_registry, tool_registry, resolved_output_root)

    idea_director = IdeaDirector()
    retry_policy = RetryPolicy(max_attempts=3)
    planner = TaskPlanner(asset_registry=asset_registry, idea_director=idea_director)
    runner = WorkflowRunner(
        skill_registry=skill_registry,
        run_memory=run_memory,
        portfolio_memory=portfolio_memory,
        retry_policy=retry_policy,
        logger=logger,
        recorder=recorder,
    )
    runner.tool_registry = tool_registry
    return planner, runner, run_memory


def _allow_runtime_output_for_visual_evidence(
    output_root: Path, *, additional_roots: Iterable[Path] = ()
) -> None:
    """Allow the app's own generated media root to reach vision fallbacks.

    Vision providers intentionally reject arbitrary filesystem paths. The
    runtime output directory and any explicitly registered serial reference
    roots are approved inputs, so they remain available even when
    ``--output-dir`` points outside the repository.
    """

    resolved = output_root.expanduser().resolve()
    configured = [
        Path(raw.strip()).expanduser().resolve()
        for raw in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
        if raw.strip()
    ]
    if not any(resolved == root or root in resolved.parents for root in configured):
        configured.append(resolved)
    for additional_root in additional_roots:
        candidate = additional_root.expanduser().resolve()
        if not any(candidate == root or root in candidate.parents for root in configured):
            configured.append(candidate)
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


def _serial_publish_configuration(
    project_root: Path,
    character: str,
    requested_platforms: Sequence[str] | None,
    character_config_path: str | Path | None = None,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    config_path = (
        Path(character_config_path).expanduser().resolve()
        if character_config_path
        else project_root / "configs" / "characters" / f"{character}.yaml"
    )
    if not config_path.is_file():
        raise ValueError(f"Serial publishing has no character platform config: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    social = payload.get("social_media", {}) if isinstance(payload, dict) else {}
    raw_platforms = social.get("platforms", {}) if isinstance(social, dict) else {}
    if not isinstance(raw_platforms, dict):
        raise ValueError(f"Character platform config is invalid: {config_path}")
    enabled: dict[str, dict[str, object]] = {}
    for name, raw_config in raw_platforms.items():
        if not isinstance(raw_config, dict) or not raw_config.get("enabled", False):
            continue
        platform_config = dict(raw_config)
        raw_folder = str(platform_config.get("config_folder_path") or "").strip()
        if raw_folder.startswith("/app/"):
            local_folder = project_root / raw_folder.removeprefix("/app/")
            if local_folder.exists():
                platform_config["config_folder_path"] = str(local_folder.resolve())
        enabled[str(name)] = platform_config
    platforms = [str(value) for value in (requested_platforms or enabled.keys())]
    unknown = [name for name in platforms if name not in enabled]
    if unknown:
        raise ValueError(
            f"Serial publishing requested disabled or unconfigured platforms for {character}: {', '.join(unknown)}"
        )
    if not platforms:
        raise ValueError(f"Character {character} has no enabled social publishing platforms.")
    return platforms, {name: enabled[name] for name in platforms}


def _serial_video_speed_configuration(
    project_root: Path,
    character: str,
    character_config_path: str | Path | None = None,
) -> dict[str, object] | None:
    config_path = (
        Path(character_config_path).expanduser().resolve()
        if character_config_path
        else project_root / "configs" / "characters" / f"{character.lower()}.yaml"
    )
    if not config_path.is_file():
        raise ValueError(f"Serial video has no character config: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Character config is invalid: {config_path}")
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"Character generation config is invalid: {config_path}")
    video_speed = generation.get("video_speed")
    if video_speed is None:
        return None
    if not isinstance(video_speed, dict):
        raise ValueError(f"generation.video_speed must be a mapping in {config_path}")
    return dict(video_speed)


def _character_discord_review_timeout_configuration(
    project_root: Path,
    character: str,
    character_config_path: str | Path | None = None,
    *,
    serial_mode: bool = False,
) -> int | None:
    if character_config_path:
        config_path = Path(character_config_path).expanduser().resolve()
    elif character:
        config_path = (
            project_root
            / "configs"
            / "characters"
            / f"{character.lower()}.yaml"
        )
    else:
        config_path = None
    if config_path is None or not config_path.is_file():
        if character_config_path:
            raise ValueError(f"Character config not found: {config_path}")
        return None
    payload = load_character_config(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Character config is invalid: {config_path}")
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"Character generation config is invalid: {config_path}")
    return configured_discord_review_timeout_seconds(
        generation,
        serial_mode=serial_mode,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic media runtime")
    parser.add_argument("--goal", help="High-level goal for the runtime")
    parser.add_argument("--media-type", default="long_video", help="Target media type")
    parser.add_argument("--duration-seconds", type=int, default=30, help="Target duration")
    parser.add_argument("--style", default="cinematic surreal", help="Preferred visual style")
    parser.add_argument("--execute", action="store_true", help="Execute the generated plan")
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
    parser.add_argument("--character-config", help="Character YAML that configures serial video and social platforms")
    parser.add_argument("--episode-topic", help="Required short topic shown in serial episode publication copy")
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
    parser.add_argument("--serial-mode", action="store_true", help="Continue a fixed-character episodic video series with per-segment human gates")
    parser.add_argument("--serial-auto-next", action="store_true", help="Resume the latest blocked episode first; otherwise start a new series or continue an eligible one")
    parser.add_argument("--series-id", help="Stable ID for an explicit serial video series")
    parser.add_argument(
        "--serial-character-reference",
        action="append",
        dest="serial_character_references",
        default=[],
        help="Fixed character identity image for a new series; repeat up to three times, and optionally pass the same paths later",
    )
    parser.add_argument("--serial-episode-number", type=int, help="Episode to create or re-render; defaults to the next ready episode")
    parser.add_argument("--serial-publish-episode-number", type=int, help="Run final Discord approval and social dispatch for an already-ready episode")
    parser.add_argument("--serial-resume-episode-number", type=int, help="Resume the current blocked episode after its accepted segments are reviewed")
    parser.add_argument("--serial-review-episode-number", type=int, help="Current episode whose timed-out segment review is being recorded")
    parser.add_argument("--serial-review-segment-number", type=int, help="Segment reviewed by the creator")
    parser.add_argument("--serial-review-decision", choices=("accepted", "rejected"), help="Creator's decision for one QA-passed serial MP4")
    parser.add_argument("--serial-reviewed-video", help="Exact MP4 path inspected by the creator")
    parser.add_argument(
        "--discord-review-timeout-seconds",
        type=int,
        help="Maximum seconds to wait for a required Discord human review",
    )
    args = parser.parse_args(argv)
    if not args.goal and not args.serial_auto_next:
        parser.error("--goal is required unless --serial-auto-next is selected")
    if args.serial_mode:
        if not args.series_id and not args.serial_auto_next:
            parser.error("--serial-mode requires --series-id")
        if args.series_id and args.serial_auto_next:
            parser.error("--serial-auto-next selects the series; omit --series-id")
        if args.media_type.strip().lower() != "long_video":
            parser.error("--serial-mode requires --media-type long_video")
        if not args.character or not args.character.strip():
            parser.error("--serial-mode requires an explicit --character")
        if args.serial_episode_number is not None and args.serial_episode_number < 1:
            parser.error("--serial-episode-number must be positive")
        if args.serial_publish_episode_number is not None and args.serial_publish_episode_number < 1:
            parser.error("--serial-publish-episode-number must be positive")
        if args.serial_resume_episode_number is not None and args.serial_resume_episode_number < 1:
            parser.error("--serial-resume-episode-number must be positive")
        if args.serial_review_episode_number is not None and args.serial_review_episode_number < 1:
            parser.error("--serial-review-episode-number must be positive")
        if args.serial_review_segment_number is not None and args.serial_review_segment_number < 1:
            parser.error("--serial-review-segment-number must be positive")
        lifecycle_modes = sum(
            value is not None
            for value in (
                args.serial_episode_number,
                args.serial_publish_episode_number,
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
            )
        )
        if lifecycle_modes > 1:
            parser.error("Rerender, resume, and manual review episode options are mutually exclusive")
        if (
            args.serial_publish_episode_number is None
            and args.serial_resume_episode_number is None
            and args.serial_review_episode_number is None
            and not str(args.episode_topic or "").strip()
        ):
            if not args.serial_auto_next:
                parser.error("New serial episodes require --episode-topic")
        if args.serial_auto_next and any(
            value is not None
            for value in (
                args.serial_episode_number,
                args.serial_publish_episode_number,
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
            )
        ):
            parser.error("--serial-auto-next cannot be combined with episode-specific lifecycle options")
        has_review_inputs = any(
            value is not None
            for value in (
                args.serial_review_segment_number,
                args.serial_review_decision,
                args.serial_reviewed_video,
            )
        )
        if (has_review_inputs or args.serial_review_episode_number is not None) and not all(
            value is not None
            for value in (
                args.serial_review_episode_number,
                args.serial_review_segment_number,
                args.serial_review_decision,
                args.serial_reviewed_video,
            )
        ):
            parser.error("Manual serial review requires an episode, segment, decision, and exact reviewed MP4 path")
        if args.serial_review_episode_number is not None and args.execute:
            parser.error("Manual serial review records one decision; do not combine it with --execute")
    elif (
        args.series_id
        or args.serial_character_references
        or args.serial_episode_number is not None
        or args.serial_publish_episode_number is not None
        or args.serial_resume_episode_number is not None
        or args.serial_review_episode_number is not None
        or args.serial_auto_next
        or args.serial_review_segment_number is not None
        or args.serial_review_decision is not None
        or args.serial_reviewed_video is not None
    ):
        parser.error("Serial series options require --serial-mode")
    if args.discord_review_timeout_seconds is not None and args.discord_review_timeout_seconds < 1:
        parser.error("--discord-review-timeout-seconds must be positive")
    return args


def _execution_plan_from_dict(payload: dict[str, object]) -> ExecutionPlan:
    goal_payload = payload.get("goal")
    if not isinstance(goal_payload, dict):
        raise ValueError("Persisted serial plan has no goal object.")
    constraints = goal_payload.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("Persisted serial plan goal constraints are invalid.")
    goal = GoalRequest(
        prompt=str(goal_payload.get("prompt") or ""),
        media_type=str(goal_payload.get("media_type") or "long_video"),
        duration_seconds=int(goal_payload.get("duration_seconds") or 30),
        style=str(goal_payload.get("style") or "cinematic surreal"),
        constraints=dict(constraints),
    )
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("Persisted serial plan has no node list.")
    nodes: list[ExecutionNode] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError("Persisted serial plan contains an invalid node.")
        nodes.append(
            ExecutionNode(
                node_id=str(raw_node.get("node_id") or ""),
                skill_name=str(raw_node.get("skill_name") or ""),
                inputs=dict(raw_node.get("inputs") or {}),
                depends_on=[str(value) for value in raw_node.get("depends_on", [])],
                tags=[str(value) for value in raw_node.get("tags", [])],
                tool_name=(str(raw_node["tool_name"]) if raw_node.get("tool_name") else None),
                stage=(str(raw_node["stage"]) if raw_node.get("stage") else None),
            )
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Persisted serial plan metadata is invalid.")
    return ExecutionPlan(
        goal=goal,
        workflow_name=str(payload.get("workflow_name") or "long_video"),
        nodes=nodes,
        metadata=dict(metadata),
        description=(str(payload["description"]) if payload.get("description") else None),
    )


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


def main(argv: Sequence[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(REPO_ROOT / "media_overload.env", override=False)
    configured_review_timeout = _character_discord_review_timeout_configuration(
        REPO_ROOT,
        args.character,
        args.character_config,
        serial_mode=args.serial_mode,
    )
    discord_review_timeout_seconds = (
        configured_review_timeout
        if configured_review_timeout is not None
        else args.discord_review_timeout_seconds
        if args.discord_review_timeout_seconds is not None
        else DEFAULT_DISCORD_REVIEW_TIMEOUT_SECONDS
    )
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    serial_store = None
    serial_context = None
    resume_episode_row = None
    resume_initial_outputs: dict[str, object] = {}
    if args.serial_mode:
        serial_store = SerialVideoStore()
        requested_episode_number = (
            args.serial_resume_episode_number
            if args.serial_resume_episode_number is not None
            else (
                args.serial_review_episode_number
                if args.serial_review_episode_number is not None
                else (
                    args.serial_publish_episode_number
                    if args.serial_publish_episode_number is not None
                    else args.serial_episode_number
                )
            )
        )
        if args.serial_auto_next:
            serial_context = serial_store.prepare_auto_next_context(
                args.character,
                args.serial_character_references,
            )
            args.series_id = str(serial_context["series_id"])
            if serial_context.get("automation_mode") == "resume":
                args.serial_resume_episode_number = int(serial_context["episode_number"])
                args.goal = str(serial_context["episode_prompt"])
                args.episode_topic = str(serial_context["episode_topic"])
                args.duration_seconds = int(serial_context["target_duration_seconds"])
        else:
            serial_context = serial_store.prepare_series_context(
                args.series_id,
                args.character,
                args.serial_character_references,
                requested_episode_number,
            )
        if args.serial_review_episode_number is not None:
            reviewed_episode = serial_store.get_current_episode(
                args.series_id, args.serial_review_episode_number
            )
            if str(reviewed_episode["prompt"]) != str(args.goal).strip():
                raise ValueError("Manual review must use the episode's original story premise in --goal.")
            result = serial_store.record_review(
                str(reviewed_episode["episode_id"]),
                int(args.serial_review_segment_number),
                str(args.serial_review_decision),
                Path(str(args.serial_reviewed_video)).resolve(),
                details={
                    "decision_source": "creator_cli",
                    "reviewer": "operator",
                    "reviewed_video_path": str(Path(str(args.serial_reviewed_video)).resolve()),
                },
            )
            print(
                json.dumps(
                    {
                        "episode_number": args.serial_review_episode_number,
                        "segment_number": args.serial_review_segment_number,
                        **result,
                    },
                    ensure_ascii=False,
                )
            )
            return
        if args.serial_resume_episode_number is not None:
            resume_episode_row = serial_store.get_current_episode(
                args.series_id, args.serial_resume_episode_number
            )
            if resume_episode_row["status"] != "blocked":
                raise ValueError("--serial-resume-episode-number requires a current blocked episode.")
            if str(resume_episode_row["prompt"]) != str(args.goal).strip():
                raise ValueError("Resume the episode with the original story premise in --goal.")
            serial_context["episode_id"] = resume_episode_row["episode_id"]
    planner, runner, run_memory = build_runtime(
        project_root,
        output_root=output_root,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=Path(args.comfy_root).resolve() if args.comfy_root else None,
        input_roots=[Path(root) for root in args.edit_input_roots or []],
    )
    if args.serial_auto_next and (not args.goal or not args.episode_topic):
        idea = LLMPromptEngine(
            mode=os.environ.get("AGENTIC_LLM_MODE", "llm")
        ).generate_serial_episode_idea(
            character=args.character,
            automation_mode=str(serial_context.get("automation_mode") or "new"),
            story_context=str(serial_context.get("story_context") or ""),
        )
        args.goal = args.goal or str(idea["premise"])
        args.episode_topic = args.episode_topic or str(idea["topic"])
    if serial_context is not None:
        serial_reference_roots = [
            Path(str(item["path"])).parent
            for item in serial_context["reference_manifest"]
            if isinstance(item, dict) and item.get("path")
        ]
        _allow_runtime_output_for_visual_evidence(
            output_root or project_root / "output",
            additional_roots=serial_reference_roots,
        )

    if (
        args.serial_publish_episode_number is not None
        and serial_store is not None
        and serial_context is not None
    ):
        episode = serial_store.get_current_episode(
            args.series_id, args.serial_publish_episode_number
        )
        if episode["status"] != "ready" or not episode.get("final_video_path"):
            raise ValueError("Only an assembled, ready serial episode can enter final publication review.")
        if str(episode["prompt"]) != str(args.goal).strip():
            raise ValueError("Final publication must use the episode's original premise in --goal.")
        from agentic.runtime.contracts import GoalRequest

        platforms, platform_configs = _serial_publish_configuration(
            REPO_ROOT, args.character, args.platforms, args.character_config
        )
        platforms = serial_store.platforms_without_successful_receipt(
            str(episode["episode_id"]), platforms
        )
        if not platforms:
            raise ValueError("All requested platforms already have a reported-success receipt for this episode.")
        platform_configs = {platform: platform_configs[platform] for platform in platforms}
        publish_goal = GoalRequest(
            prompt=str(episode["prompt"]),
            media_type="publish_review",
            duration_seconds=int(episode["target_duration_seconds"]),
            style=args.style,
            constraints={
                "serial_mode": True,
                "serial_database_name": serial_store.database_name,
                "serial_episode_id": str(episode["episode_id"]),
                "serial_episode_number": int(episode["episode_number"]),
                "episode_topic": str(episode["episode_topic"]),
                "series_id": str(episode["series_id"]),
                "character": args.character,
                "media_paths": [str(episode["final_video_path"])],
                "review_scope": "final_video",
                "require_human_review": True,
                "platforms": platforms,
                "platform_configs": platform_configs,
                "publish_mode": args.publish_mode or "live",
                "dry_run": bool(args.dry_run_publish),
                "hashtags": args.hashtags or [],
                "selection_limit": 1,
                "output_dir": str(output_root) if output_root else None,
                "discord_review_timeout_seconds": discord_review_timeout_seconds,
            },
        )
        plan = planner.build_plan(publish_goal)
        if args.execute:
            print("=== PLAN ===")
            print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            print("=== RUN ===")
            result = runner.run(plan)
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            print("=== MEMORY ===")
            print(json.dumps(run_memory.as_serializable(), indent=2, ensure_ascii=False))
            return
        print("=== PLAN ===")
        print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
        return

    serial_platforms: list[str] = []
    serial_platform_configs: dict[str, dict[str, object]] = {}
    serial_video_speed: dict[str, object] | None = None
    if serial_context is not None:
        serial_platforms, serial_platform_configs = _serial_publish_configuration(
            REPO_ROOT, args.character, args.platforms, args.character_config
        )
        if resume_episode_row is not None:
            serial_platforms = serial_store.platforms_without_successful_receipt(
                str(resume_episode_row["episode_id"]), serial_platforms
            )
            if not serial_platforms:
                raise ValueError(
                    "The blocked episode already has a reported-success receipt for every requested platform."
                )
            serial_platform_configs = {
                platform: serial_platform_configs[platform]
                for platform in serial_platforms
            }
        serial_video_speed = _serial_video_speed_configuration(
            REPO_ROOT, args.character, args.character_config
        )

    goal = planner.create_goal(
        prompt=args.goal,
        media_type=args.media_type,
        duration_seconds=args.duration_seconds,
        style=args.style,
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
            "longvideo_edit_profile": args.longvideo_edit_profile,
            "longvideo_production_profile": args.longvideo_production_profile,
            "longvideo_steps": args.longvideo_steps,
            "text": args.text,
            "character": args.character,
            "platforms": serial_platforms if serial_context is not None else (args.platforms or []),
            "platform_configs": serial_platform_configs,
            "publish_mode": args.publish_mode,
            "hashtags": args.hashtags or [],
            "caption_prefix": "" if args.serial_mode else args.caption_prefix,
            "selection_limit": args.selection_limit,
            "review_notes": args.review_notes,
            "enable_review_loop": args.enable_review_loop,
            "dry_run": args.dry_run_publish,
            "output_dir": str(output_root) if output_root else None,
            "width": args.width,
            "height": args.height,
            "use_tts": args.use_tts,
            "discord_review_timeout_seconds": discord_review_timeout_seconds,
            **(
                {
                    "serial_mode": True,
                    "series_id": serial_context["series_id"],
                    "serial_episode_number": serial_context["episode_number"],
                    "serial_database_name": serial_store.database_name,
                    "episode_topic": args.episode_topic,
                    "serial_automation_mode": serial_context.get("automation_mode", "manual"),
                    "serial_automation_source_series_id": serial_context.get("automation_source_series_id", ""),
                    "serial_automation_source_episode_id": serial_context.get("automation_source_episode_id", ""),
                    "platforms": serial_platforms,
                    "platform_configs": serial_platform_configs,
                    "publish_mode": args.publish_mode or "live",
                    "serial_story_context": serial_context["story_context"],
                    "reference_manifest": serial_context["reference_manifest"],
                    "video_speed": serial_video_speed,
                    "longvideo_review_policy": "every_segment",
                    "longvideo_reference_max_images": 4,
                    "longvideo_reference_selection_limit": 4,
                    "require_human_review": True,
                    "stage_probe_auto_select": False,
                    "enable_review_loop": False,
                    "video_count": 1,
                }
                if serial_context is not None and serial_store is not None
                else {}
            ),
        },
    )
    if resume_episode_row is not None:
        goal.constraints["serial_episode_id"] = str(resume_episode_row["episode_id"])
    if serial_store is not None and resume_episode_row is not None:
        saved_plan = serial_store.get_plan_payload(str(resume_episode_row["episode_id"]))
        if saved_plan is not None:
            plan = _execution_plan_from_dict(saved_plan)
            if (
                plan.goal.prompt != str(args.goal).strip()
                or plan.goal.constraints.get("series_id") != args.series_id
                or plan.goal.constraints.get("character") != args.character
                or plan.goal.constraints.get("serial_episode_id") != resume_episode_row["episode_id"]
                or plan.goal.constraints.get("serial_database_name") != serial_store.database_name
            ):
                raise ValueError("The saved serial plan does not belong to this episode, series, or output root.")
            plan.goal.constraints[
                "discord_review_timeout_seconds"
            ] = discord_review_timeout_seconds
        else:
            plan = planner.build_plan(goal)
            if args.execute:
                serial_store.store_reconstructed_plan(
                    str(resume_episode_row["episode_id"]), plan.to_dict()
                )
    else:
        plan = planner.build_plan(goal)

    episode = None
    if args.execute and serial_store is not None and serial_context is not None:
        try:
            if resume_episode_row is not None:
                episode = resume_episode_row
                resume_state = serial_store.resume_episode(str(episode["episode_id"]))
                resume_initial_outputs = dict(resume_state["initial_node_outputs"])
                completed_segment_count = int(resume_state["completed_segment_count"])
                for node in plan.nodes:
                    segment_index = node.inputs.get("segment_index")
                    if (
                        isinstance(segment_index, int)
                        and segment_index < completed_segment_count
                        and node.node_id not in resume_initial_outputs
                        and node.skill_name != "media.video.extract_last_frame"
                    ):
                        resume_initial_outputs[node.node_id] = {}
            else:
                episode = serial_store.start_episode(
                    serial_context,
                    args.goal,
                    topic=args.episode_topic,
                    target_duration_seconds=args.duration_seconds,
                )
                goal.constraints["serial_episode_id"] = episode["episode_id"]
                serial_store.record_plan(
                    episode["episode_id"],
                    segment_count=int(plan.metadata["segment_count"]),
                    plan_payload=plan.to_dict(),
                )
        except Exception as exc:
            if episode is not None:
                serial_store.fail_episode(episode["episode_id"], f"{type(exc).__name__}: {exc}")
            raise

    print("=== PLAN ===")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))

    if not args.execute:
        return

    print("=== RUN ===")
    try:
        result = runner.run(plan, initial_node_outputs=resume_initial_outputs)
    except Exception as exc:
        if serial_store is not None and episode is not None:
            serial_store.fail_episode(episode["episode_id"], f"{type(exc).__name__}: {exc}")
        raise
    if serial_store is not None and episode is not None:
        final_qa = result.state.node_outputs.get("longvideo-video-qa", {})
        final_qa_passed = isinstance(final_qa, dict) and bool(final_qa.get("passed"))
        if final_qa_passed:
            final_node = str(plan.metadata.get("final_video_node") or "")
            final_output = result.state.node_outputs.get(final_node, {})
            if isinstance(final_output, dict):
                final_video_path = (
                    final_output.get("final_video_path")
                    or final_output.get("video_path")
                    or ""
                )
                if isinstance(final_video_path, list):
                    final_video_path = next((path for path in final_video_path if path), "")
            else:
                final_video_path = ""
            try:
                serial_store.complete_episode(
                    episode["episode_id"], "success", str(final_video_path) or None
                )
            except Exception as exc:
                serial_store.fail_episode(episode["episode_id"], f"{type(exc).__name__}: {exc}")
                raise
        if not final_qa_passed or result.status != "success":
            reason = (
                "workflow execution failed after final video QA"
                if final_qa_passed
                else "final video QA did not pass"
            )
            serial_store.fail_episode(episode["episode_id"], reason)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("=== PROMPTS ===")
    print(json.dumps(_build_prompt_summary(result.to_dict().get("state", {})), indent=2, ensure_ascii=False))
    print("=== MEMORY ===")
    print(json.dumps(run_memory.as_serializable(), indent=2, ensure_ascii=False))
    if args.serial_auto_next:
        if result.status != "success" or episode is None:
            raise SystemExit(2)
        publication_rows = serial_store.get_episode_publications(str(episode["episode_id"]))
        latest_attempt = max((int(row["attempt_number"]) for row in publication_rows), default=0)
        latest_statuses = {
            str(row["platform"]): str(row["status"])
            for row in publication_rows
            if int(row["attempt_number"]) == latest_attempt
        }
        required_platforms = {
            str(value)
            for value in plan.goal.constraints.get("platforms", [])
        }
        if not required_platforms or any(
            latest_statuses.get(platform) != "reported_success"
            for platform in required_platforms
        ):
            serial_store.fail_episode(
                str(episode["episode_id"]),
                "automatic serial delivery is incomplete: at least one enabled platform has no reported_success receipt",
            )
            raise SystemExit(3)


if __name__ == "__main__":
    main()
