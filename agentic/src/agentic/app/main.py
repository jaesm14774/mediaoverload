import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
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
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.observability import RunRecorder
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.step_logger import create_run_logger
from agentic.runtime.serial_video import SerialVideoStore
from agentic.runtime.serial_story import (
    build_serial_story_prompts,
    serial_story_artifact_path,
)
from agentic.runtime.serial_character import (
    build_serial_character_prompts,
    serial_character_artifact_path,
)
from agentic.runtime.serial_episode_plan import (
    build_serial_episode_outline_prompts,
    serial_episode_outline_artifact_path,
)
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
from agentic.tools.context_services import DiscordHumanReviewService, NewsContextService
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


def _serial_series_selection_weights_configuration(
    project_root: Path,
    character: str,
    character_config_path: str | Path | None = None,
) -> dict[str, object]:
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
        return {"new": 0.5, "continue": 0.5}
    payload = load_character_config(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Character config is invalid: {config_path}")
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"Character generation config is invalid: {config_path}")
    serial = generation.get("serial_video", {})
    if not isinstance(serial, dict):
        raise ValueError(f"generation.serial_video must be a mapping in {config_path}")
    weights = serial.get("series_selection_weights", {"new": 0.5, "continue": 0.5})
    if not isinstance(weights, dict):
        raise ValueError(
            f"generation.serial_video.series_selection_weights must be a mapping in {config_path}"
        )
    return dict(weights)


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
    parser.add_argument("--character-config", help="YAML production profile for serial review, video speed and social platforms; the approved card supplies the actual character identity")
    parser.add_argument(
        "--episode-topic",
        help="Optional topic for serial story planning; it is not added to public captions",
    )
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
    parser.add_argument("--serial-mode", action="store_true", help="Generate or continue a story-first approved series with per-segment human gates")
    parser.add_argument("--serial-auto-next", action="store_true", help="Resume blocked work first; otherwise continue an eligible existing series with an approved episode outline")
    parser.add_argument("--serial-require-delivery-receipts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--series-id", help="Stable ID for an explicit serial video series")
    parser.add_argument(
        "--serial-story-draft",
        action="store_true",
        help="Create a full series story treatment before choosing a character",
    )
    parser.add_argument(
        "--serial-news-inspiration",
        choices=("random", "none"),
        default="random",
        help="Use one random traceable local news item as fictional inspiration, or explicitly disable news",
    )
    parser.add_argument(
        "--serial-story-complexity",
        choices=("simple", "cinematic"),
        default="simple",
        help="Keep serial animation simple and clear by default, or opt into cinematic complexity",
    )
    parser.add_argument(
        "--serial-story-review-plan-id",
        help="Story plan ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-story-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending series story treatment",
    )
    parser.add_argument(
        "--serial-story-reviewed-file",
        help="Exact story file reviewed by the creator; required when accepting",
    )
    parser.add_argument(
        "--serial-story-review-notes",
        default="",
        help="Optional creator notes for the story decision",
    )
    parser.add_argument(
        "--serial-character-draft",
        action="store_true",
        help="Create a character identity card from the latest approved series story",
    )
    parser.add_argument(
        "--serial-character-preferences",
        default="",
        help="Optional creator preferences for the character; omit to let the model invent one",
    )
    parser.add_argument(
        "--serial-character-card-review-id",
        help="Character card ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-character-card-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending character identity card",
    )
    parser.add_argument(
        "--serial-character-card-name",
        help="Final character name to lock when approving a card",
    )
    parser.add_argument(
        "--serial-character-card-reviewed-file",
        help="Exact character card file reviewed by the creator",
    )
    parser.add_argument(
        "--serial-character-card-review-notes",
        default="",
        help="Optional creator notes for the character card decision",
    )
    parser.add_argument(
        "--serial-episode-plan-draft",
        action="store_true",
        help="Draft and review an episode story, scene and shot outline from the approved series materials",
    )
    parser.add_argument(
        "--serial-episode-plan-review-id",
        help="Episode outline ID to accept or reject after reviewing its Markdown file",
    )
    parser.add_argument(
        "--serial-episode-plan-review-decision",
        choices=("accepted", "rejected"),
        help="Human decision for a pending episode outline",
    )
    parser.add_argument(
        "--serial-episode-plan-reviewed-file",
        help="Exact episode outline file reviewed by the creator",
    )
    parser.add_argument(
        "--serial-episode-plan-review-notes",
        default="",
        help="Optional creator notes for the episode outline decision",
    )
    parser.add_argument(
        "--serial-character-reference",
        action="append",
        dest="serial_character_references",
        default=[],
        help="Reference image to lock when approving a character card; serial generation uses the approved paths and hashes",
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
    story_review_mode = bool(args.serial_story_review_plan_id)
    character_review_mode = bool(args.serial_character_card_review_id)
    character_workflow_mode = bool(args.serial_character_draft or character_review_mode)
    episode_outline_review_mode = bool(args.serial_episode_plan_review_id)
    episode_outline_workflow_mode = bool(args.serial_episode_plan_draft or episode_outline_review_mode)
    story_workflow_mode = bool(args.serial_story_draft or story_review_mode)
    onboarding_workflow_mode = (
        story_workflow_mode or character_workflow_mode or episode_outline_workflow_mode
    )
    has_story_review_fields = any(
        value is not None
        for value in (
            args.serial_story_review_decision,
            args.serial_story_reviewed_file,
        )
    ) or bool(args.serial_story_review_notes)
    has_character_review_fields = any(
        value is not None
        for value in (
            args.serial_character_card_review_decision,
            args.serial_character_card_name,
            args.serial_character_card_reviewed_file,
        )
    ) or bool(args.serial_character_references)
    has_character_review_fields = has_character_review_fields or bool(
        args.serial_character_card_review_notes
    )
    has_episode_outline_review_fields = any(
        value is not None
        for value in (
            args.serial_episode_plan_review_decision,
            args.serial_episode_plan_reviewed_file,
        )
    ) or bool(args.serial_episode_plan_review_notes)
    if (
        not args.goal
        and not args.serial_auto_next
        and not onboarding_workflow_mode
        and not (args.serial_mode and args.series_id)
    ):
        parser.error("--goal is required unless --serial-auto-next is selected")
    if sum((
        bool(args.serial_story_draft),
        story_review_mode,
        bool(args.serial_character_draft),
        character_review_mode,
        bool(args.serial_episode_plan_draft),
        episode_outline_review_mode,
    )) > 1:
        parser.error("Run one story-first onboarding stage at a time")
    if onboarding_workflow_mode and args.serial_mode:
        parser.error("Story-first onboarding stages are separate from video generation; omit --serial-mode")
    if character_workflow_mode and not str(args.series_id or "").strip():
        parser.error("Character card stages require --series-id from the approved story")
    if args.serial_character_draft and (has_character_review_fields or args.serial_character_card_review_decision):
        parser.error("Create a character card draft or review a saved card, not both")
    if character_review_mode and args.serial_character_card_review_decision is None:
        parser.error("Character card review requires --serial-character-card-review-decision accepted|rejected")
    if has_character_review_fields and not character_review_mode:
        parser.error("Character card decision, name, file, and references require --serial-character-card-review-id")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not str(args.serial_character_card_reviewed_file or "").strip()
    ):
        parser.error("Approving a character card requires --serial-character-card-reviewed-file")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not str(args.serial_character_card_name or "").strip()
    ):
        parser.error("Approving a character card requires --serial-character-card-name")
    if (
        character_review_mode
        and args.serial_character_card_review_decision == "accepted"
        and not args.serial_character_references
    ):
        parser.error("Approving a character card requires at least one --serial-character-reference image")
    if episode_outline_workflow_mode and not str(args.series_id or "").strip():
        parser.error("Episode outline stages require --series-id from the approved story")
    if args.serial_episode_plan_draft and args.serial_episode_number is None:
        parser.error("Episode outline drafting requires --serial-episode-number")
    if args.serial_episode_plan_draft and args.serial_episode_number is not None and args.serial_episode_number < 1:
        parser.error("--serial-episode-number must be positive")
    if args.serial_episode_plan_draft and not str(args.episode_topic or "").strip():
        parser.error("Episode outline drafting requires --episode-topic")
    if episode_outline_review_mode and args.serial_episode_plan_review_decision is None:
        parser.error("Episode outline review requires --serial-episode-plan-review-decision accepted|rejected")
    if has_episode_outline_review_fields and not episode_outline_review_mode:
        parser.error("Episode outline decision, file, and notes require --serial-episode-plan-review-id")
    if (
        episode_outline_review_mode
        and args.serial_episode_plan_review_decision == "accepted"
        and not str(args.serial_episode_plan_reviewed_file or "").strip()
    ):
        parser.error("Approving an episode outline requires --serial-episode-plan-reviewed-file")
    if args.serial_story_draft and args.serial_news_inspiration == "none" and not str(args.goal or "").strip():
        parser.error("--serial-news-inspiration none requires a creator story idea in --goal")
    if story_review_mode and args.serial_story_review_decision is None:
        parser.error("Story review requires --serial-story-review-decision accepted|rejected")
    if has_story_review_fields and not story_review_mode:
        parser.error("Story review decision, file, and notes require --serial-story-review-plan-id")
    if (
        story_review_mode
        and args.serial_story_review_decision == "accepted"
        and not str(args.serial_story_reviewed_file or "").strip()
    ):
        parser.error("Accepting a story requires --serial-story-reviewed-file")
    if onboarding_workflow_mode and any(
        value is not None
        for value in (
            args.serial_episode_number if not args.serial_episode_plan_draft else None,
            args.serial_publish_episode_number,
            args.serial_resume_episode_number,
            args.serial_review_episode_number,
        )
    ):
        parser.error("Story planning and episode lifecycle options cannot be combined")
    if args.serial_mode:
        if not args.series_id and not args.serial_auto_next:
            parser.error("--serial-mode requires --series-id")
        if args.series_id and args.serial_auto_next:
            parser.error("--serial-auto-next selects the series; omit --series-id")
        if args.media_type.strip().lower() != "long_video":
            parser.error("--serial-mode requires --media-type long_video")
        if not (args.character or args.series_id):
            parser.error("--serial-mode requires --character for auto-next or --series-id for a saved series")
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
            and not args.series_id
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
    elif not onboarding_workflow_mode and (
        (args.series_id and not onboarding_workflow_mode)
        or (args.serial_news_inspiration != "random" and not args.serial_story_draft)
        or (args.serial_character_references and not character_review_mode)
        or args.serial_character_preferences
        or args.serial_character_card_name
        or args.serial_character_card_reviewed_file
        or args.serial_character_card_review_decision is not None
        or args.serial_character_card_review_notes
        or args.serial_episode_plan_draft
        or args.serial_episode_plan_review_id
        or args.serial_episode_plan_review_decision is not None
        or args.serial_episode_plan_reviewed_file
        or args.serial_episode_plan_review_notes
        or (args.serial_episode_number is not None and not episode_outline_workflow_mode)
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


def _run_serial_story_draft(
    args: argparse.Namespace, project_root: Path, *, emit: bool = True
) -> dict[str, object]:
    store = SerialVideoStore()
    series_id = str(args.series_id or f"serial-{uuid.uuid4().hex[:16]}").strip()
    news_context: dict[str, object] | None = None
    if args.serial_news_inspiration == "random":
        selection = NewsContextService().select_news(random_inspiration=True)
        if selection is None:
            raise ValueError(
                "No usable traceable news item is available for story inspiration. "
                "Retry when the news feed is available, or explicitly use "
                "--serial-news-inspiration none with a custom --goal."
            )
        news_context = selection.to_dict()
        news_context["content"] = " ".join(str(news_context.get("content") or "").split())[:1800]
        news_context["selected_at"] = datetime.now().astimezone().isoformat()
    else:
        news_context = {
            "mode": "none",
            "selected_at": datetime.now().astimezone().isoformat(),
        }

    system_prompt, user_prompt = build_serial_story_prompts(
        str(args.goal or ""),
        news_context if args.serial_news_inspiration == "random" else None,
        news_mode=str(args.serial_news_inspiration),
        character_name=str(args.character or ""),
        complexity=str(args.serial_story_complexity or "simple"),
    )
    client = PromptClient(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    treatment = client.generate_text(
        system_prompt,
        user_prompt,
        step_name="serial_series_story_treatment",
        language_contract="請使用自然、易讀的繁體中文。",
    )
    story_text = treatment.rstrip() + "\n"
    story_plan_id = uuid.uuid4().hex
    output_root = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"
    artifact_path = serial_story_artifact_path(output_root, series_id, story_plan_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(story_text, encoding="utf-8")
    prompt_sha256 = hashlib.sha256(
        (system_prompt + "\0" + user_prompt).encode("utf-8")
    ).hexdigest()
    draft = store.save_series_story_draft(
        series_id,
        story_text,
        user_idea=str(args.goal or ""),
        inspiration=news_context,
        prompt_sha256=prompt_sha256,
        model_id=client.current_model_id("text"),
        artifact_path=str(artifact_path),
        story_plan_id=story_plan_id,
    )
    result = {
        "status": "success",
        "workflow_stage": "story_onboarding",
        "series_id": series_id,
        "story_plan_id": str(draft["story_plan_id"]),
        "story_path": str(artifact_path),
        "story_text": story_text,
        "story_sha256": str(draft["story_sha256"]),
        "version": int(draft["version"]),
        "news_inspiration": news_context,
    }
    if emit:
        print(
            json.dumps(
                {
                    "status": draft["status"],
                    "series_id": series_id,
                    "story_plan_id": draft["story_plan_id"],
                    "version": draft["version"],
                    "story_sha256": draft["story_sha256"],
                    "story_path": str(artifact_path),
                    "news_inspiration": news_context,
                    "next_action": "For integrated Discord review and video generation, start with run_media_interface.py --character <name> --generation-type serial_video.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return result


def _run_serial_story_review(args: argparse.Namespace) -> None:
    store = SerialVideoStore()
    plan = store.get_series_story_plan(str(args.serial_story_review_plan_id))
    reviewed_text = ""
    if args.serial_story_reviewed_file:
        reviewed_path = Path(args.serial_story_reviewed_file).resolve(strict=True)
        expected_path = Path(str(plan.get("artifact_path") or "")).resolve()
        if reviewed_path != expected_path:
            raise ValueError(
                "Story review must use the editable Markdown file registered with this story plan."
            )
        if not reviewed_path.is_file():
            raise ValueError("The reviewed serial story file must be a regular file.")
        reviewed_text = reviewed_path.read_text(encoding="utf-8")
    result = store.record_series_story_decision(
        str(args.serial_story_review_plan_id),
        str(args.serial_story_review_decision),
        reviewed_treatment=reviewed_text,
        reviewer="operator",
        notes=str(args.serial_character_card_review_notes or ""),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "series_id": result["series_id"],
                "story_plan_id": result["story_plan_id"],
                "version": result["version"],
                "review_decision": result["review_decision"],
                "approved_sha256": result["approved_sha256"],
                "reviewer": result["reviewer"],
                "review_notes": result["review_notes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_serial_character_draft(
    args: argparse.Namespace, project_root: Path, *, emit: bool = True
) -> dict[str, object]:
    store = SerialVideoStore()
    series_id = str(args.series_id or "").strip()
    story = store.get_latest_approved_series_story(series_id)
    if story is None:
        raise ValueError("Approve a complete series story before creating its character card.")
    system_prompt, user_prompt = build_serial_character_prompts(
        approved_story=str(story.get("approved_treatment") or ""),
        user_preferences=str(args.serial_character_preferences or ""),
        character_name=str(args.character or ""),
        complexity=str(args.serial_story_complexity or "simple"),
    )
    client = PromptClient(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    card_text = client.generate_text(
        system_prompt,
        user_prompt,
        step_name="serial_series_character_card",
        language_contract="請使用自然、易讀的繁體中文。",
    ).rstrip() + "\n"
    card_id = uuid.uuid4().hex
    output_root = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"
    artifact_path = serial_character_artifact_path(output_root, series_id, card_id)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(card_text, encoding="utf-8")
    prompt_sha256 = hashlib.sha256(
        (system_prompt + "\0" + user_prompt).encode("utf-8")
    ).hexdigest()
    draft = store.save_series_character_card_draft(
        series_id,
        str(story["story_plan_id"]),
        card_text,
        user_preferences=str(args.serial_character_preferences or ""),
        prompt_sha256=prompt_sha256,
        model_id=client.current_model_id("text"),
        artifact_path=str(artifact_path),
        character_card_id=card_id,
    )
    result = {
        "status": "success",
        "series_id": series_id,
        "story_plan_id": str(story["story_plan_id"]),
        "character_card_id": str(draft["character_card_id"]),
        "version": int(draft["version"]),
        "card_sha256": str(draft["card_sha256"]),
        "character_card_path": str(artifact_path),
        "character_card_text": card_text,
    }
    if emit:
        print(
            json.dumps(
                {
                    **{key: value for key, value in result.items() if key != "character_card_text"},
                    "reference_image_step": "This standalone draft command does not generate or review reference images.",
                    "next_action": "Use run_media_interface.py --character <name> --generation-type serial_video for the integrated Discord workflow.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return result


def _run_serial_character_review(args: argparse.Namespace) -> None:
    store = SerialVideoStore()
    card = store.get_series_character_card(str(args.serial_character_card_review_id))
    reviewed_text = ""
    if args.serial_character_card_reviewed_file:
        reviewed_path = Path(args.serial_character_card_reviewed_file).resolve(strict=True)
        expected_path = Path(str(card.get("artifact_path") or "")).resolve()
        if reviewed_path != expected_path:
            raise ValueError(
                "Character review must use the editable Markdown file registered with this card."
            )
        if not reviewed_path.is_file():
            raise ValueError("The reviewed character card must be a regular file.")
        reviewed_text = reviewed_path.read_text(encoding="utf-8")
    result = store.record_series_character_card_decision(
        str(args.serial_character_card_review_id),
        str(args.serial_character_card_review_decision),
        character_name=str(args.serial_character_card_name or ""),
        reviewed_card=reviewed_text,
        reference_paths=args.serial_character_references,
        reviewer="operator",
        notes=str(args.serial_story_review_notes or ""),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "series_id": result["series_id"],
                "story_plan_id": result["story_plan_id"],
                "character_card_id": result["character_card_id"],
                "version": result["version"],
                "character_name": result["character_name"],
                "review_decision": result["review_decision"],
                "approved_sha256": result["approved_sha256"],
                "reference_assets": result["reference_assets"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_serial_episode_outline_draft(
    args: argparse.Namespace, project_root: Path, *, emit: bool = True
) -> dict[str, object]:
    store = SerialVideoStore()
    series_id = str(args.series_id or "").strip()
    story = store.get_latest_approved_series_story(series_id)
    character = store.get_latest_approved_character_card(series_id)
    if story is None:
        raise ValueError("Approve a complete series story before planning an episode.")
    if character is None or character.get("story_plan_id") != story.get("story_plan_id"):
        raise ValueError("Approve a character card for the current series story before planning an episode.")
    episode_number = int(args.serial_episode_number)
    previous_summaries: list[str] = []
    for prior_number in range(max(1, episode_number - 2), episode_number):
        prior = store.get_latest_approved_series_episode_outline(series_id, prior_number)
        if prior is None:
            continue
        excerpt = str(prior.get("approved_outline") or "")[-1800:]
        previous_summaries.append(
            f"第 {prior_number} 集（{prior['episode_topic']}）核准內容尾段：\n{excerpt}"
        )
    system_prompt, user_prompt = build_serial_episode_outline_prompts(
        approved_story=str(story.get("approved_treatment") or ""),
        approved_character=str(character.get("approved_card") or ""),
        episode_number=episode_number,
        episode_topic=str(args.episode_topic or ""),
        episode_direction=str(args.goal or ""),
        previous_episode_summaries=previous_summaries,
        complexity=str(args.serial_story_complexity or "simple"),
    )
    client = PromptClient(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    outline_text = client.generate_text(
        system_prompt,
        user_prompt,
        step_name="serial_episode_story_scenes_shots",
        language_contract="請使用自然、易讀的繁體中文。",
    ).rstrip() + "\n"
    outline_id = uuid.uuid4().hex
    output_root = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"
    artifact_path = serial_episode_outline_artifact_path(
        output_root, series_id, episode_number, outline_id
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(outline_text, encoding="utf-8")
    prompt_sha256 = hashlib.sha256(
        (system_prompt + "\0" + user_prompt).encode("utf-8")
    ).hexdigest()
    draft = store.save_series_episode_outline_draft(
        series_id,
        episode_number,
        str(args.episode_topic or ""),
        outline_text,
        episode_direction=str(args.goal or ""),
        prompt_sha256=prompt_sha256,
        model_id=client.current_model_id("text"),
        artifact_path=str(artifact_path),
        outline_id=outline_id,
    )
    result = {
        "status": "success",
        "series_id": series_id,
        "episode_number": episode_number,
        "episode_topic": str(draft["episode_topic"]),
        "outline_id": str(draft["outline_id"]),
        "version": int(draft["version"]),
        "outline_sha256": str(draft["outline_sha256"]),
        "story_plan_id": str(draft["story_plan_id"]),
        "story_sha256": str(draft["story_sha256"]),
        "character_card_id": str(draft["character_card_id"]),
        "character_sha256": str(draft["character_sha256"]),
        "outline_path": str(artifact_path),
        "outline_text": outline_text,
    }
    if emit:
        print(
            json.dumps(
                {
                    **{key: value for key, value in result.items() if key != "outline_text"},
                    "next_action": "Use run_media_interface.py --character <name> --generation-type serial_video for the integrated Discord workflow.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return result


def _run_serial_episode_outline_review(args: argparse.Namespace) -> None:
    store = SerialVideoStore()
    outline = store.get_series_episode_outline(str(args.serial_episode_plan_review_id))
    reviewed_text = ""
    if args.serial_episode_plan_reviewed_file:
        reviewed_path = Path(args.serial_episode_plan_reviewed_file).resolve(strict=True)
        expected_path = Path(str(outline.get("artifact_path") or "")).resolve()
        if reviewed_path != expected_path:
            raise ValueError(
                "Episode outline review must use the editable Markdown file registered with this plan."
            )
        if not reviewed_path.is_file():
            raise ValueError("The reviewed episode outline must be a regular file.")
        reviewed_text = reviewed_path.read_text(encoding="utf-8")
    result = store.record_series_episode_outline_decision(
        str(args.serial_episode_plan_review_id),
        str(args.serial_episode_plan_review_decision),
        reviewed_outline=reviewed_text,
        reviewer="operator",
        notes=str(args.serial_episode_plan_review_notes or ""),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "series_id": result["series_id"],
                "episode_number": result["episode_number"],
                "episode_topic": result["episode_topic"],
                "outline_id": result["outline_id"],
                "review_decision": result["review_decision"],
                "approved_sha256": result["approved_sha256"],
                "story_plan_id": result["story_plan_id"],
                "character_card_id": result["character_card_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _serial_onboarding_blocked(
    *, series_id: str, stage: str, reason: str, status: str = "blocked"
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": status,
        "workflow_stage": "serial_onboarding_blocked",
        "onboarding_stage": stage,
        "series_id": series_id,
        "reason": reason,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _run_serial_new_series_onboarding(
    args: argparse.Namespace,
    project_root: Path,
    serial_store: SerialVideoStore,
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    """Review all story-first materials in Discord, then run the approved episode."""
    from agentic.app.character_requests import (
        CharacterGenerationOptions,
        CharacterReviewOptions,
        CharacterRuntimeOptions,
        CharacterWorkflowRequest,
    )
    from agentic.app.character_workflow import run_character_workflow
    from agentic.app.serial_video_dispatch import build_serial_video_argv

    series_id = str(args.series_id or "").strip()
    character_name = str(args.character or "").strip()
    if not series_id or not character_name:
        raise ValueError("New serial onboarding requires the selected character and series ID.")
    output_root = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"
    review_service = DiscordHumanReviewService(output_root=output_root)

    def review_stage(
        *,
        stage: str,
        summary: str,
        document_path: str,
        media_paths: list[str] | None = None,
    ):
        return review_service.review_candidates(
            text=summary,
            context_text=f"系列：{series_id}｜角色：{character_name}｜複雜度：{args.serial_story_complexity}。完整內容請看附件。",
            document_paths=[document_path],
            media_paths=media_paths or [],
            timeout_seconds=timeout_seconds,
            allow_asset_selection=bool(media_paths),
            allow_text_edit=False,
            selection_mode="single",
            selection_required=bool(media_paths),
            selection_limit=1 if media_paths else None,
            review_scope=f"serial_{stage}",
        )

    def require_acceptance(stage: str, decision) -> dict[str, object] | None:
        if decision.review_mode == "discord" and decision.status == "approved":
            return None
        if decision.status == "rejected":
            return _serial_onboarding_blocked(
                series_id=series_id,
                stage=stage,
                reason="The creator rejected this draft in Discord; no later stage was started.",
            )
        return _serial_onboarding_blocked(
            series_id=series_id,
            stage=stage,
            reason=decision.fallback_reason or "Required Discord review did not complete; the draft remains unapproved.",
        )

    story = _run_serial_story_draft(args, project_root, emit=False)
    story_text = str(story["story_text"])
    story_plan_id = str(story["story_plan_id"])
    story_review = review_stage(
        stage="story",
        summary="【系列故事審核】請閱讀附件。接受後會繼續製作角色、首集大綱並開始影片流程；拒絕或逾時不會生成影片。",
        document_path=str(story["story_path"]),
    )
    blocked = require_acceptance("story", story_review)
    if blocked is not None:
        if story_review.status == "rejected":
            serial_store.record_series_story_decision(
                story_plan_id, "rejected", reviewer=story_review.reviewer
            )
        return blocked
    serial_store.record_series_story_decision(
        story_plan_id,
        "accepted",
        reviewed_treatment=story_text,
        reviewer=story_review.reviewer,
    )

    args.serial_character_preferences = (
        f"主角固定為 {character_name}，不可更換角色或改成一般人類。"
    )
    character = _run_serial_character_draft(args, project_root, emit=False)
    card_text = str(character["character_card_text"])
    card_path = str(character["character_card_path"])

    reference_paths = [str(Path(path).expanduser().resolve()) for path in args.serial_character_references]
    if not reference_paths:
        marker = "角色參考圖提示"
        reference_prompt = card_text.split(marker, 1)[-1].strip(" \r\n：:")
        if not reference_prompt:
            reference_prompt = card_text
        reference_prompt = (
            f"為固定主角 {character_name} 產生單人全身角色參考圖。只呈現角色，不放故事場景、文字或其他人物。\n"
            f"核准的角色外觀描述：\n{reference_prompt}"
        )
        production_config = (
            Path(args.character_config).expanduser().resolve()
            if args.character_config
            else REPO_ROOT / "configs" / "characters" / f"{character_name.lower()}.yaml"
        )
        if not production_config.is_file():
            return _serial_onboarding_blocked(
                series_id=series_id,
                stage="character_reference",
                reason=f"Character production profile does not exist: {production_config}",
                status="failed",
            )
        reference_output = output_root / "serial_series" / series_id / "character_references"
        reference_output.mkdir(parents=True, exist_ok=True)
        reference_request = CharacterWorkflowRequest(
            repo_root=REPO_ROOT,
            config_path=production_config,
            generation=CharacterGenerationOptions(
                prompt=reference_prompt,
                preferred_generation_type="text2img",
                output_dir=str(reference_output),
                selected_character_name=character_name,
                subject_mode="single",
            ),
            review=CharacterReviewOptions(
                publish_after_generate=False,
                no_review=True,
            ),
            runtime=CharacterRuntimeOptions(
                comfy_host=args.comfy_host,
                comfy_port=args.comfy_port,
                comfy_root=args.comfy_root,
            ),
        )
        reference_result = run_character_workflow(reference_request)
        if str(reference_result.get("status") or "").lower() != "success":
            return _serial_onboarding_blocked(
                series_id=series_id,
                stage="character_reference",
                reason=str(reference_result.get("failure_reason") or "Character reference image generation failed."),
                status="failed",
            )
        artifacts = reference_result.get("artifacts", {})
        reference_paths = [
            str(path)
            for path in (artifacts.get("image_paths", []) if isinstance(artifacts, dict) else [])
            if path
        ]
    if not reference_paths:
        return _serial_onboarding_blocked(
            series_id=series_id,
            stage="character_reference",
            reason="No character reference images were available for Discord review.",
            status="failed",
        )

    card_review = review_stage(
        stage="character",
        summary=f"【角色與參考圖審核】請確認附件中的 {character_name} 角色設定，並在下方選擇一張角色參考圖。接受後會繼續首集大綱。",
        document_path=card_path,
        media_paths=reference_paths,
    )
    blocked = require_acceptance("character", card_review)
    if blocked is not None:
        if card_review.status == "rejected":
            serial_store.record_series_character_card_decision(
                str(character["character_card_id"]),
                "rejected",
                reviewer=card_review.reviewer,
            )
        return blocked
    if not card_review.selected_paths:
        return _serial_onboarding_blocked(
            series_id=series_id,
            stage="character",
            reason="Discord accepted the character card without selecting a reference image.",
        )
    serial_store.record_series_character_card_decision(
        str(character["character_card_id"]),
        "accepted",
        character_name=character_name,
        reviewed_card=card_text,
        reference_paths=card_review.selected_paths,
        reviewer=card_review.reviewer,
    )

    episode_idea = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm")).generate_serial_episode_idea(
        character=character_name,
        automation_mode="new",
        story_context=story_text,
    )
    args.serial_episode_number = 1
    args.episode_topic = str(args.episode_topic or episode_idea["topic"])
    args.goal = str(args.goal or episode_idea["premise"])
    outline = _run_serial_episode_outline_draft(args, project_root, emit=False)
    outline_review = review_stage(
        stage="episode_outline",
        summary="【首集大綱審核】請閱讀附件。接受後會直接進入本集影片生成和既有 Discord 片段審核。",
        document_path=str(outline["outline_path"]),
    )
    blocked = require_acceptance("episode_outline", outline_review)
    if blocked is not None:
        if outline_review.status == "rejected":
            serial_store.record_series_episode_outline_decision(
                str(outline["outline_id"]),
                "rejected",
                reviewer=outline_review.reviewer,
            )
        return blocked
    serial_store.record_series_episode_outline_decision(
        str(outline["outline_id"]),
        "accepted",
        reviewed_outline=str(outline["outline_text"]),
        reviewer=outline_review.reviewer,
    )

    config_path = (
        Path(args.character_config).expanduser().resolve()
        if args.character_config
        else REPO_ROOT / "configs" / "characters" / f"{character_name.lower()}.yaml"
    )
    episode_argv = build_serial_video_argv(
        config_path=config_path,
        repo_root=REPO_ROOT,
        episode_index=0,
        prompt=str(outline["outline_text"]),
        episode_topic=str(outline["episode_topic"]),
        duration_seconds=args.duration_seconds,
        output_dir=args.output_dir,
        publish_platforms=args.platforms or (),
        publish_mode=args.publish_mode,
        dry_run_publish=args.dry_run_publish,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=args.comfy_root,
        story_complexity=str(args.serial_story_complexity or "simple"),
    )
    resumed_argv: list[str] = []
    skip_reference = False
    for item in episode_argv:
        if skip_reference:
            skip_reference = False
            continue
        if item == "--serial-auto-next":
            continue
        if item == "--serial-character-reference":
            skip_reference = True
            continue
        resumed_argv.append(item)
    resumed_argv.extend(("--series-id", series_id))
    resumed_argv.append("--serial-require-delivery-receipts")
    result = main(resumed_argv)
    if isinstance(result, dict):
        result["automation_mode"] = "new"
        return result
    return {
        "status": "success",
        "workflow_stage": "serial_episode",
        "automation_mode": "new",
        "series_id": series_id,
        "episode_number": 1,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, object] | None:
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
    if args.serial_story_draft:
        return _run_serial_story_draft(args, project_root)
    if args.serial_story_review_plan_id:
        _run_serial_story_review(args)
        return
    if args.serial_character_draft:
        _run_serial_character_draft(args, project_root)
        return
    if args.serial_character_card_review_id:
        _run_serial_character_review(args)
        return
    if args.serial_episode_plan_draft:
        _run_serial_episode_outline_draft(args, project_root)
        return
    if args.serial_episode_plan_review_id:
        _run_serial_episode_outline_review(args)
        return
    serial_store = None
    serial_context = None
    resume_episode_row = None
    resume_initial_outputs: dict[str, object] = {}
    if args.serial_mode:
        serial_store = SerialVideoStore()
        lifecycle_action = any(
            value is not None
            for value in (
                args.serial_publish_episode_number,
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
            )
        )
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
            selection_weights = _serial_series_selection_weights_configuration(
                REPO_ROOT,
                args.character,
                args.character_config,
            )
            serial_context = serial_store.prepare_auto_next_context(
                args.character,
                args.serial_character_references,
                series_selection_weights=selection_weights,
            )
            args.series_id = str(serial_context["series_id"])
            if serial_context.get("automation_mode") == "new":
                return _run_serial_new_series_onboarding(
                    args,
                    project_root,
                    serial_store,
                    timeout_seconds=discord_review_timeout_seconds,
                )
            if serial_context.get("automation_mode") == "resume":
                args.serial_resume_episode_number = int(serial_context["episode_number"])
                args.goal = str(serial_context["episode_prompt"])
                args.episode_topic = str(serial_context["episode_topic"])
                args.duration_seconds = int(serial_context["target_duration_seconds"])
        else:
            onboarding = serial_store.get_series_onboarding(str(args.series_id))
            if onboarding["is_complete"]:
                approved_name = str(onboarding["character"]["character_name"])
                if args.character and str(args.character).strip() != approved_name:
                    raise ValueError("--character must match the approved serial character card.")
                args.character = approved_name
                approved_refs = [
                    str(item["path"]) for item in onboarding["character"]["reference_assets"]
                ]
                if args.serial_character_references:
                    supplied = serial_store._identity_files(args.serial_character_references)
                    approved = serial_store._identity_files(approved_refs)
                    if supplied != approved:
                        raise ValueError("--serial-character-reference must match the approved character card.")
                args.serial_character_references = approved_refs
            elif not args.character:
                args.character = serial_store.get_production_series_character_name(
                    str(args.series_id)
                ) or ""
            serial_context = serial_store.prepare_series_context(
                args.series_id,
                args.character,
                args.serial_character_references,
                requested_episode_number,
                require_story_first_for_new=True,
                require_episode_outline=not lifecycle_action,
            )
        args.character = str(serial_context["character"])
        onboarding_story = serial_context.get("onboarding_story")
        if onboarding_story is not None and not lifecycle_action:
            lineage = serial_context.get("onboarding_lineage")
            if not isinstance(lineage, dict):
                raise ValueError("An approved episode outline is required before serial video generation.")
            approved_outline = str(lineage["episode_outline"] or "").strip()
            approved_topic = str(lineage["episode_topic"] or "").strip()
            if args.goal and str(args.goal).strip() != approved_outline:
                raise ValueError("Use the exact approved episode outline; omit --goal to load it automatically.")
            if args.episode_topic and str(args.episode_topic).strip() != approved_topic:
                raise ValueError("The episode topic is locked by the approved episode outline.")
            args.goal = approved_outline
            args.episode_topic = approved_topic
        elif not lifecycle_action and not args.serial_auto_next:
            if not str(args.goal or "").strip():
                raise ValueError("An existing series without story-first onboarding still requires --goal.")
            if not str(args.episode_topic or "").strip():
                raise ValueError("An existing series without an approved episode outline requires --episode-topic.")
        if args.serial_review_episode_number is not None:
            reviewed_episode = serial_store.get_current_episode(
                args.series_id, args.serial_review_episode_number
            )
            if not args.goal:
                args.goal = str(reviewed_episode["prompt"])
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
        production_config = (
            Path(args.character_config).expanduser().resolve()
            if args.character_config
            else REPO_ROOT / "configs" / "characters" / f"{args.character.lower()}.yaml"
        )
        if not production_config.is_file():
            raise ValueError(
                "Serial generation needs a YAML production profile for Discord, video speed, and social platforms. "
                "Pass --character-config to an existing profile; its character name does not replace the approved series character."
            )
        if args.serial_resume_episode_number is not None:
            resume_episode_row = serial_store.get_current_episode(
                args.series_id, args.serial_resume_episode_number
            )
            if resume_episode_row["status"] != "blocked":
                raise ValueError("--serial-resume-episode-number requires a current blocked episode.")
            if not args.goal:
                args.goal = str(resume_episode_row["prompt"])
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
    if args.serial_auto_next or args.serial_require_delivery_receipts:
        if result.status != "success" or episode is None:
            if args.serial_auto_next:
                raise SystemExit(2)
            return {
                "status": "blocked",
                "workflow_stage": "serial_episode",
                "series_id": str(serial_context.get("series_id") or ""),
                "reason": "Serial episode workflow did not complete successfully.",
            }
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
            if args.serial_auto_next:
                raise SystemExit(3)
            return {
                "status": "failed",
                "workflow_stage": "serial_episode",
                "series_id": str(serial_context.get("series_id") or ""),
                "episode_id": str(episode["episode_id"]),
                "reason": "Automatic serial delivery is incomplete; at least one enabled platform has no reported_success receipt.",
            }
        return {
            "status": "success",
            "workflow_stage": "serial_episode",
            "automation_mode": str(serial_context.get("automation_mode") or ""),
            "series_id": str(serial_context.get("series_id") or ""),
            "episode_id": str(episode["episode_id"]),
            "episode_number": int(episode["episode_number"]),
        }
    if serial_context is not None and episode is not None:
        return {
            "status": "success" if result.status == "success" else "blocked",
            "workflow_stage": "serial_episode",
            "automation_mode": str(serial_context.get("automation_mode") or "manual"),
            "series_id": str(serial_context.get("series_id") or ""),
            "episode_id": str(episode["episode_id"]),
            "episode_number": int(episode["episode_number"]),
            "reason": "" if result.status == "success" else "Serial episode workflow did not complete successfully.",
        }


if __name__ == "__main__":
    main()
