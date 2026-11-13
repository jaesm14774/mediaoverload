from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.app.cli import parse_args

from agentic.runtime.contracts import (
    DEFAULT_DISCORD_REVIEW_TIMEOUT_SECONDS,
    ExecutionNode,
    ExecutionPlan,
    GoalRequest,
    WorkflowRunResult,
)

if TYPE_CHECKING:
    from agentic.memory import RunMemory
    from agentic.runtime.observability import RunRecorder
    from agentic.runtime.planner import TaskPlanner
    from agentic.runtime.runner import WorkflowRunner
    from agentic.runtime.serial_video import SerialVideoStore
    from agentic.tools.context_services import HumanReviewDecision


@dataclass(slots=True)
class _SerialExecutionState:
    store: SerialVideoStore | None = None
    context: dict[str, object] | None = None
    resume_episode: dict[str, object] | None = None
    resume_initial_outputs: dict[str, object] = field(default_factory=dict)
    finished: bool = False
    result: dict[str, object] | None = None


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
    from agentic.assets.registry import AssetRegistry
    from agentic.memory import PortfolioMemory, RunMemory
    from agentic.runtime.creativity import IdeaDirector, RetryPolicy
    from agentic.runtime.llm_engine import LLMPromptEngine
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
    from agentic.skills.motion_comic import register_motion_comic_skills
    from agentic.skills.motion_graphics import register_motion_graphics_skills
    from agentic.skills.reference_video import register_reference_video_skills
    from agentic.tools.comfy import register_builtin_tools
    from agentic.tools.comfy_h3 import register_comfy_h3_tools
    from agentic.tools.comfy_nvidia_vfx import register_comfy_nvidia_vfx_tools
    from agentic.tools.comfy_workflow_tool import register_comfy_workflow_tools
    from agentic.tools.context_services import NewsContextService
    from agentic.tools.media_services import register_media_service_tools
    from agentic.tools.social_services import register_social_service_tools

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


def _serial_resume_node_outputs(
    plan: ExecutionPlan,
    resume_state: dict[str, object],
) -> dict[str, object]:
    initial_outputs = dict(resume_state["initial_node_outputs"])
    completed_segment_count = int(resume_state["completed_segment_count"])
    for node in plan.nodes:
        segment_index = node.inputs.get("segment_index")
        if (
            isinstance(segment_index, int)
            and segment_index < completed_segment_count
            and node.node_id not in initial_outputs
            and node.skill_name != "media.video.extract_last_frame"
        ):
            initial_outputs[node.node_id] = {}
    return initial_outputs


def _prepare_serial_execution_plan(
    planner: TaskPlanner,
    goal: GoalRequest,
    serial_store: SerialVideoStore | None,
    resume_episode_row: dict[str, object] | None,
    *,
    discord_review_timeout_seconds: int,
) -> tuple[ExecutionPlan, dict[str, object] | None]:
    if serial_store is None or resume_episode_row is None:
        return planner.build_plan(goal), resume_episode_row

    episode_id = str(resume_episode_row["episode_id"])
    saved_plan = serial_store.get_plan_payload(episode_id)
    if saved_plan is None:
        raise ValueError(
            f"Blocked serial episode '{episode_id}' has no saved current-format execution plan."
        )

    plan = _execution_plan_from_dict(saved_plan)
    if (
        plan.goal.prompt != str(goal.prompt).strip()
        or plan.goal.constraints.get("series_id") != goal.constraints.get("series_id")
        or plan.goal.constraints.get("character") != goal.constraints.get("character")
        or plan.goal.constraints.get("serial_episode_id") != episode_id
        or plan.goal.constraints.get("serial_database_name") != serial_store.database_name
    ):
        raise ValueError("The saved serial plan does not belong to this episode, series, or output root.")

    _apply_current_serial_publish_targets(plan, goal)
    plan.goal.constraints["discord_review_timeout_seconds"] = discord_review_timeout_seconds
    return plan, resume_episode_row


def _record_serial_manual_review(
    args: argparse.Namespace,
    serial_store: SerialVideoStore,
) -> dict[str, object]:
    reviewed_episode = serial_store.get_current_episode(
        args.series_id, args.serial_review_episode_number
    )
    if not args.goal:
        args.goal = str(reviewed_episode["prompt"])
    if str(reviewed_episode["prompt"]) != str(args.goal).strip():
        raise ValueError("Manual review must use the episode's original story premise in --goal.")
    reviewed_video = Path(str(args.serial_reviewed_video)).resolve()
    review_result = serial_store.record_review(
        str(reviewed_episode["episode_id"]),
        int(args.serial_review_segment_number),
        str(args.serial_review_decision),
        reviewed_video,
        details={
            "decision_source": "creator_cli",
            "reviewer": "operator",
            "reviewed_video_path": str(reviewed_video),
        },
    )
    return {
        "episode_number": args.serial_review_episode_number,
        "segment_number": args.serial_review_segment_number,
        **review_result,
    }


def _apply_current_serial_publish_targets(
    plan: ExecutionPlan,
    goal: GoalRequest,
) -> None:
    """Use the current character's allowed publication targets when resuming a saved plan."""
    platforms = [str(value) for value in goal.constraints.get("platforms", [])]
    platform_configs = dict(goal.constraints.get("platform_configs", {}) or {})
    plan.goal.constraints["platforms"] = list(platforms)
    plan.goal.constraints["platform_configs"] = dict(platform_configs)
    for node in plan.nodes:
        if "platforms" in node.inputs:
            node.inputs["platforms"] = list(platforms)
        if "platform_configs" in node.inputs:
            node.inputs["platform_configs"] = dict(platform_configs)
        summary_metadata = node.inputs.get("summary_metadata")
        if isinstance(summary_metadata, dict) and "platforms" in summary_metadata:
            summary_metadata["platforms"] = list(platforms)


def _serial_segment_regeneration_requested(plan: ExecutionPlan, result) -> bool:
    for node in plan.nodes:
        if node.inputs.get("review_phase") != "segment_acceptance":
            continue
        outputs = result.state.node_outputs.get(node.node_id)
        if isinstance(outputs, dict) and outputs.get("regeneration_requested") is True:
            return True
    return False


def _required_delivery_receipts_present(
    publication_rows: Sequence[dict[str, object]],
    required_platforms: set[str],
) -> bool:
    if not required_platforms:
        return False
    latest_attempt = max(
        (int(row["attempt_number"]) for row in publication_rows),
        default=0,
    )
    latest_statuses = {
        str(row["platform"]): str(row["status"])
        for row in publication_rows
        if int(row["attempt_number"]) == latest_attempt
    }
    return all(latest_statuses.get(platform) == "reported_success" for platform in required_platforms)


def _serial_completion_result(
    args: argparse.Namespace,
    serial_store: SerialVideoStore | None,
    serial_context: dict[str, object] | None,
    episode: dict[str, object] | None,
    plan: ExecutionPlan,
    result: WorkflowRunResult,
) -> dict[str, object] | None:
    if args.serial_auto_next or args.serial_require_delivery_receipts:
        if result.status != "success" or episode is None:
            return {
                "status": "failed" if args.serial_auto_next else "blocked",
                "generation_type": "serial_video",
                "workflow_stage": "serial_episode",
                "series_id": str((serial_context or {}).get("series_id") or ""),
                "episode_id": str((episode or {}).get("episode_id") or ""),
                "reason": "Serial episode workflow did not complete successfully.",
            }
        if serial_store is None or serial_context is None:
            raise ValueError("Automatic serial completion requires its serial store and context.")
        publication_rows = serial_store.get_episode_publications(str(episode["episode_id"]))
        required_platforms = {
            str(value)
            for value in plan.goal.constraints.get("platforms", [])
        }
        if not _required_delivery_receipts_present(publication_rows, required_platforms):
            serial_store.fail_episode(
                str(episode["episode_id"]),
                "automatic serial delivery is incomplete: at least one enabled platform has no reported_success receipt",
            )
            return {
                "status": "failed",
                "generation_type": "serial_video",
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
    return None


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
    from agentic.app.character_config import (
        configured_discord_review_timeout_seconds,
        load_character_config,
    )

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
    from agentic.app.character_config import load_character_config

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
    from agentic.runtime.llm_client import PromptClient
    from agentic.runtime.serial_video import SerialVideoStore
    from agentic.runtime.serial_story import (
        build_serial_story_prompts,
        serial_story_artifact_path,
    )
    from agentic.tools.context_services import NewsContextService

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
    from agentic.runtime.serial_video import SerialVideoStore

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
    from agentic.runtime.llm_client import PromptClient
    from agentic.runtime.serial_video import SerialVideoStore
    from agentic.runtime.serial_character import (
        build_serial_character_prompts,
        serial_character_artifact_path,
    )

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
    from agentic.runtime.serial_video import SerialVideoStore

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


def _earlier_episode_actions(
    store: SerialVideoStore, series_id: str, episode_number: int
) -> list[str]:
    """One line per approved earlier episode, so a new episode cannot replay it."""
    from agentic.runtime.serial_episode_plan import episode_action_digest

    digests: list[str] = []
    for prior_number in range(max(1, episode_number - 10), episode_number):
        prior = store.get_latest_approved_series_episode_outline(series_id, prior_number)
        if prior is not None:
            digests.append(
                episode_action_digest(
                    prior_number,
                    str(prior["episode_topic"]),
                    str(prior.get("approved_outline") or ""),
                )
            )
    return digests


def _run_serial_episode_outline_draft(
    args: argparse.Namespace, project_root: Path, *, emit: bool = True
) -> dict[str, object]:
    from agentic.runtime.llm_client import PromptClient
    from agentic.runtime.serial_video import SerialVideoStore
    from agentic.runtime.planning.long_video import minimum_segment_count
    from agentic.runtime.serial_episode_plan import (
        build_serial_episode_outline_prompts,
        serial_episode_outline_artifact_path,
    )

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
        earlier_episode_actions=_earlier_episode_actions(store, series_id, episode_number),
        beat_count=minimum_segment_count(args.duration_seconds),
        approved_story_state=str(
            store.get_human_approved_story_state(series_id).get("state_text") or ""
        ),
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
    from agentic.runtime.serial_video import SerialVideoStore

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


def _resolve_serial_character_reference_paths(
    args: argparse.Namespace,
    *,
    output_root: Path,
    series_id: str,
    character_name: str,
    card_text: str,
) -> tuple[list[str], str]:
    reference_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.serial_character_references
    ]
    if reference_paths:
        return reference_paths, ""

    from agentic.app.character_requests import (
        CharacterGenerationOptions,
        CharacterReviewOptions,
        CharacterRuntimeOptions,
        CharacterWorkflowRequest,
    )
    from agentic.app.character_workflow import run_character_workflow

    from agentic.runtime.serial_character import (
        build_serial_character_reference_prompt,
    )
    from agentic.runtime.serial_paths import serial_series_artifact_root

    production_config = (
        Path(args.character_config).expanduser().resolve()
        if args.character_config
        else REPO_ROOT / "configs" / "characters" / f"{character_name.lower()}.yaml"
    )
    if not production_config.is_file():
        return [], f"Character production profile does not exist: {production_config}"
    from agentic.app.character_config import load_character_config
    from agentic.app.serial_look import serial_series_look

    look = serial_series_look(
        dict(load_character_config(production_config).get("generation") or {}),
        series_id,
    )
    style = look["style"]
    reference_prompt = build_serial_character_reference_prompt(
        character_name=character_name,
        character_card=card_text,
        # Only the medium and palette: a catalog style's own composition would pull the
        # character off-centre, which is the one thing an identity image must not do.
        style_direction="; ".join(
            part for part in (style["prompt"], style["palette"]) if part
        ),
    )

    reference_output = (
        serial_series_artifact_root(output_root, series_id) / "character_references"
    )
    reference_output.mkdir(parents=True, exist_ok=True)
    reference_request = CharacterWorkflowRequest(
        repo_root=REPO_ROOT,
        config_path=production_config,
        generation=CharacterGenerationOptions(
            prompt=reference_prompt,
            preferred_generation_type="text2img",
            output_dir=str(reference_output),
            selected_character_name=character_name,
            image_count=6,
            subject_mode="single",
            creative_profile=look,
            identity_reference=True,
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
        return [], str(
            reference_result.get("failure_reason")
            or "Character reference image generation failed."
        )
    artifacts = reference_result.get("artifacts", {})
    reference_paths = [
        str(path)
        for path in (artifacts.get("image_paths", []) if isinstance(artifacts, dict) else [])
        if path
    ]
    if not reference_paths:
        return [], "No character reference images were available for Discord review."
    return reference_paths, ""


def _serial_onboarding_review_blocked(
    serial_store: SerialVideoStore,
    series_id: str,
    stage: str,
    decision: HumanReviewDecision,
) -> dict[str, object] | None:
    if decision.review_mode == "discord" and decision.status == "approved":
        return None
    if str((decision.delivery or {}).get("series_disposition") or "") == "discarded":
        # A series still in onboarding has no production record to close; its rejected
        # draft already keeps it from resuming. Only a started series needs closing.
        if serial_store.get_production_series_character_name(series_id) is not None:
            serial_store.set_series_lifecycle_status(
                series_id,
                "discarded",
                note=f"Creator discarded the series while reviewing the {stage}",
            )
        reason = "The creator discarded this series in Discord; no later stage was started."
    elif decision.status == "rejected":
        reason = "The creator rejected this draft in Discord; no later stage was started."
    else:
        reason = decision.fallback_reason or "Required Discord review did not complete; the draft remains unapproved."
    return _serial_onboarding_blocked(series_id=series_id, stage=stage, reason=reason)


def _record_serial_story_review(
    serial_store: SerialVideoStore,
    series_id: str,
    story_plan_id: str,
    reviewed_story: str,
    decision: HumanReviewDecision,
) -> dict[str, object] | None:
    blocked = _serial_onboarding_review_blocked(
        serial_store, series_id, "story", decision
    )
    if blocked is not None:
        if decision.status == "rejected":
            serial_store.record_series_story_decision(
                story_plan_id, "rejected", reviewer=decision.reviewer
            )
        return blocked
    serial_store.record_series_story_decision(
        story_plan_id,
        "accepted",
        reviewed_treatment=reviewed_story,
        reviewer=decision.reviewer,
    )
    return None


def _record_serial_character_review(
    serial_store: SerialVideoStore,
    series_id: str,
    character_card_id: str,
    character_name: str,
    reviewed_card: str,
    decision: HumanReviewDecision,
) -> dict[str, object] | None:
    blocked = _serial_onboarding_review_blocked(
        serial_store, series_id, "character", decision
    )
    if blocked is not None:
        if decision.status == "rejected":
            serial_store.record_series_character_card_decision(
                character_card_id,
                "rejected",
                reviewer=decision.reviewer,
            )
        return blocked
    if not decision.selected_paths:
        return _serial_onboarding_blocked(
            series_id=series_id,
            stage="character",
            reason="Discord accepted the character card without selecting a reference image.",
        )
    serial_store.record_series_character_card_decision(
        character_card_id,
        "accepted",
        character_name=character_name,
        reviewed_card=reviewed_card,
        reference_paths=decision.selected_paths,
        reviewer=decision.reviewer,
    )
    return None


def _record_serial_episode_outline_review(
    serial_store: SerialVideoStore,
    series_id: str,
    outline_id: str,
    reviewed_outline: str,
    decision: HumanReviewDecision,
) -> dict[str, object] | None:
    blocked = _serial_onboarding_review_blocked(
        serial_store, series_id, "episode_outline", decision
    )
    if blocked is not None:
        if decision.status == "rejected":
            serial_store.record_series_episode_outline_decision(
                outline_id,
                "rejected",
                reviewer=decision.reviewer,
            )
        return blocked
    serial_store.record_series_episode_outline_decision(
        outline_id,
        "accepted",
        reviewed_outline=reviewed_outline,
        reviewer=decision.reviewer,
    )
    return None


def _serial_first_episode_argv(argv: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    skip_reference_value = False
    for item in argv:
        if skip_reference_value:
            skip_reference_value = False
            continue
        if item == "--serial-auto-next":
            continue
        if item == "--serial-character-reference":
            skip_reference_value = True
            continue
        filtered.append(item)
    return filtered


def _canonical_character_look(args: argparse.Namespace, character_name: str) -> str:
    """Return the character library's appearance text for a configured group character."""
    from agentic.app.character_config import load_character_config
    from agentic.tools.context_services import CharacterGroupSelectionService

    config_path = (
        Path(args.character_config).expanduser().resolve()
        if args.character_config
        else REPO_ROOT / "configs" / "characters" / f"{character_name.lower()}.yaml"
    )
    group_name = str(
        (load_character_config(config_path).get("character") or {}).get("group_name") or ""
    ).strip()
    if not group_name:
        return ""
    selection = CharacterGroupSelectionService().select_named_character(
        group_name, character_name
    )
    return str((selection.to_dict().get("selected_profile") or {}).get("role_description") or "").strip()


def _run_serial_new_series_onboarding(
    args: argparse.Namespace,
    project_root: Path,
    serial_store: SerialVideoStore,
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    """Review all story-first materials in Discord, then run the approved episode."""
    from agentic.app.serial_video_dispatch import build_serial_video_argv
    from agentic.runtime.llm_engine import LLMPromptEngine
    from agentic.tools.context_services import DiscordHumanReviewService

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
            allow_series_discard=True,
        )

    story = _run_serial_story_draft(args, project_root, emit=False)
    story_text = str(story["story_text"])
    story_plan_id = str(story["story_plan_id"])
    story_review = review_stage(
        stage="story",
        summary="【系列故事審核】請閱讀附件。接受後會繼續製作角色、首集大綱並開始影片流程；拒絕或逾時不會生成影片。",
        document_path=str(story["story_path"]),
    )
    blocked = _record_serial_story_review(
        serial_store, series_id, story_plan_id, story_text, story_review
    )
    if blocked is not None:
        return blocked

    args.serial_character_preferences = (
        f"主角固定為 {character_name}，不可更換角色或改成一般人類。"
    )
    canonical_look = _canonical_character_look(args, character_name)
    if canonical_look:
        args.serial_character_preferences += (
            f"\n{character_name} 是既有角色，外觀錨點必須逐項沿用以下設定，"
            f"不可重新設計成其他生物、材質或加上新配件：{canonical_look}"
        )
    character = _run_serial_character_draft(args, project_root, emit=False)
    card_text = str(character["character_card_text"])
    card_path = str(character["character_card_path"])

    reference_paths, reference_failure = _resolve_serial_character_reference_paths(
        args,
        output_root=output_root,
        series_id=series_id,
        character_name=character_name,
        card_text=card_text,
    )
    if reference_failure:
        return _serial_onboarding_blocked(
            series_id=series_id,
            stage="character_reference",
            reason=reference_failure,
            status="failed",
        )

    card_review = review_stage(
        stage="character",
        summary=(
            f"【角色與參考圖審核】請對照角色卡，檢查 {character_name} 的輪廓、臉部或頭部特徵、配色、"
            "服裝或身體表面和標誌細節是否清楚一致，且完整入鏡；只選完全符合的一張。"
            "若候選圖都不符合，請拒絕，不要勉強鎖定。接受後，所選圖片的 SHA-256 會成為後續各集固定使用的身份參考，"
            "接著繼續首集大綱。"
        ),
        document_path=card_path,
        media_paths=reference_paths,
    )
    blocked = _record_serial_character_review(
        serial_store,
        series_id,
        str(character["character_card_id"]),
        character_name,
        card_text,
        card_review,
    )
    if blocked is not None:
        return blocked

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
    blocked = _record_serial_episode_outline_review(
        serial_store,
        series_id,
        str(outline["outline_id"]),
        str(outline["outline_text"]),
        outline_review,
    )
    if blocked is not None:
        return blocked

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
    resumed_argv = _serial_first_episode_argv(episode_argv)
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


def _run_serial_onboarding_command(
    args: argparse.Namespace,
    project_root: Path,
) -> tuple[bool, dict[str, object] | None]:
    if args.serial_story_draft:
        return True, _run_serial_story_draft(args, project_root)
    if args.serial_story_review_plan_id:
        _run_serial_story_review(args)
        return True, None
    if args.serial_character_draft:
        _run_serial_character_draft(args, project_root)
        return True, None
    if args.serial_character_card_review_id:
        _run_serial_character_review(args)
        return True, None
    if args.serial_episode_plan_draft:
        _run_serial_episode_outline_draft(args, project_root)
        return True, None
    if args.serial_episode_plan_review_id:
        _run_serial_episode_outline_review(args)
        return True, None
    return False, None


def _prepare_auto_next_serial_context(
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path | None,
    serial_store: SerialVideoStore,
    state: _SerialExecutionState,
    *,
    timeout_seconds: int,
) -> dict[str, object] | None:
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
    automation_mode = serial_context.get("automation_mode")
    if automation_mode == "new":
        state.finished = True
        state.result = _run_serial_new_series_onboarding(
            args,
            project_root,
            serial_store,
            timeout_seconds=timeout_seconds,
        )
        return None
    if automation_mode == "resume":
        args.serial_resume_episode_number = int(serial_context["episode_number"])
        args.goal = str(serial_context["episode_prompt"])
        args.episode_topic = str(serial_context["episode_topic"])
        args.duration_seconds = int(serial_context["target_duration_seconds"])
    return serial_context


def _prepare_existing_serial_context(
    args: argparse.Namespace,
    serial_store: SerialVideoStore,
    *,
    requested_episode_number: int | None,
    lifecycle_action: bool,
) -> dict[str, object]:
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
            supplied = serial_store._identity_paths(args.serial_character_references)
            approved = serial_store._identity_paths(approved_refs)
            if supplied != approved:
                raise ValueError(
                    "--serial-character-reference must match the approved character card."
                )
        args.serial_character_references = approved_refs
    elif not args.character:
        args.character = serial_store.get_production_series_character_name(
            str(args.series_id)
        ) or ""
    return serial_store.prepare_series_context(
        args.series_id,
        args.character,
        args.serial_character_references,
        requested_episode_number,
        require_story_first_for_new=True,
        require_episode_outline=not lifecycle_action,
    )


def _review_automatic_serial_outline(
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path | None,
    serial_store: SerialVideoStore,
    serial_context: dict[str, object],
    state: _SerialExecutionState,
    *,
    timeout_seconds: int,
) -> dict[str, object] | None:
    onboarding_story = serial_context.get("onboarding_story")
    if not (
        args.serial_auto_next
        and serial_context.get("automation_mode") == "continue"
        and onboarding_story is not None
        and serial_context.get("onboarding_lineage") is None
    ):
        return serial_context

    from agentic.runtime.llm_engine import LLMPromptEngine
    from agentic.tools.context_services import DiscordHumanReviewService

    episode_number = int(serial_context["episode_number"])
    args.serial_episode_number = episode_number
    idea_context = "\n\n".join(
        part
        for part in (
            str(onboarding_story.get("approved_treatment") or "").strip(),
            str(serial_context.get("story_context") or "").strip(),
        )
        if part
    )
    episode_idea = LLMPromptEngine(
        mode=os.environ.get("AGENTIC_LLM_MODE", "llm")
    ).generate_serial_episode_idea(
        character=args.character,
        automation_mode="continue",
        story_context=idea_context,
        previous_episodes=_earlier_episode_actions(
            serial_store, str(args.series_id), episode_number
        ),
    )
    args.episode_topic = str(episode_idea["topic"])
    args.goal = str(episode_idea["premise"])
    outline = _run_serial_episode_outline_draft(args, project_root, emit=False)
    review_service = DiscordHumanReviewService(
        output_root=output_root or project_root / "output"
    )
    outline_review = review_service.review_candidates(
        text=(
            f"【第 {episode_number} 集大綱審核】請閱讀附件，確認本集承接已核准的系列主線和上一集狀態。"
            "接受後會開始本集影片生成；故事品質由你判斷。"
        ),
        context_text=f"角色：{args.character}｜第 {episode_number} 集｜接受後開始影片生成。",
        document_paths=[str(outline["outline_path"])],
        media_paths=[],
        timeout_seconds=timeout_seconds,
        allow_asset_selection=False,
        allow_text_edit=False,
        review_scope="serial_episode_outline",
        allow_series_discard=True,
    )
    blocked = _record_serial_episode_outline_review(
        serial_store,
        str(args.series_id),
        str(outline["outline_id"]),
        str(outline["outline_text"]),
        outline_review,
    )
    if blocked is not None:
        state.finished = True
        state.result = blocked
        return None

    prior_context = serial_context
    serial_context = serial_store.prepare_series_context(
        str(args.series_id),
        args.character,
        [
            str(item["path"])
            for item in serial_context.get("identity_references", [])
            if isinstance(item, dict) and item.get("path")
        ],
        episode_number,
        require_story_first_for_new=True,
        require_episode_outline=True,
    )
    for key in (
        "automation_mode",
        "automation_source_series_id",
        "automation_source_episode_id",
    ):
        serial_context[key] = prior_context.get(key, "")
    approved_lineage = serial_context["onboarding_lineage"]
    args.goal = str(approved_lineage["episode_outline"])
    args.episode_topic = str(approved_lineage["episode_topic"])
    return serial_context


def _validate_serial_story_request(
    args: argparse.Namespace,
    serial_context: dict[str, object],
    *,
    lifecycle_action: bool,
) -> None:
    onboarding_story = serial_context.get("onboarding_story")
    if onboarding_story is not None and not lifecycle_action:
        lineage = serial_context.get("onboarding_lineage")
        if not isinstance(lineage, dict):
            raise ValueError(
                "An approved episode outline is required before serial video generation."
            )
        approved_outline = str(lineage["episode_outline"] or "").strip()
        approved_topic = str(lineage["episode_topic"] or "").strip()
        if args.goal and str(args.goal).strip() != approved_outline:
            raise ValueError(
                "Use the exact approved episode outline; omit --goal to load it automatically."
            )
        if args.episode_topic and str(args.episode_topic).strip() != approved_topic:
            raise ValueError("The episode topic is locked by the approved episode outline.")
        args.goal = approved_outline
        args.episode_topic = approved_topic
    elif not lifecycle_action and not args.serial_auto_next:
        if not str(args.goal or "").strip():
            raise ValueError(
                "An existing series without story-first onboarding still requires --goal."
            )
        if not str(args.episode_topic or "").strip():
            raise ValueError(
                "An existing series without an approved episode outline requires --episode-topic."
            )


def _validate_serial_resume_episode(
    args: argparse.Namespace,
    serial_store: SerialVideoStore,
    serial_context: dict[str, object],
) -> dict[str, object] | None:
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
    if args.serial_resume_episode_number is None:
        return None
    from agentic.runtime.serial_video import RESUMABLE_EPISODE_STATUSES

    resume_episode = serial_store.get_current_episode(
        args.series_id, args.serial_resume_episode_number
    )
    if resume_episode["status"] not in RESUMABLE_EPISODE_STATUSES:
        raise ValueError(
            "--serial-resume-episode-number requires a current blocked or interrupted episode."
        )
    if not args.goal:
        args.goal = str(resume_episode["prompt"])
    if str(resume_episode["prompt"]) != str(args.goal).strip():
        raise ValueError("Resume the episode with the original story premise in --goal.")
    serial_context["episode_id"] = resume_episode["episode_id"]
    return resume_episode


def _prepare_serial_execution_state(
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path | None,
    *,
    timeout_seconds: int,
) -> _SerialExecutionState:
    state = _SerialExecutionState()
    if not args.serial_mode:
        return state

    from agentic.runtime.serial_video import SerialVideoStore

    serial_store = SerialVideoStore()
    state.store = serial_store
    lifecycle_action = any(
        value is not None
        for value in (
            args.serial_publish_episode_number,
            args.serial_resume_episode_number,
            args.serial_review_episode_number,
        )
    )
    requested_episode_number = next(
        (
            value
            for value in (
                args.serial_resume_episode_number,
                args.serial_review_episode_number,
                args.serial_publish_episode_number,
                args.serial_episode_number,
            )
            if value is not None
        ),
        None,
    )
    if args.serial_auto_next:
        serial_context = _prepare_auto_next_serial_context(
            args,
            project_root,
            output_root,
            serial_store,
            state,
            timeout_seconds=timeout_seconds,
        )
    else:
        serial_context = _prepare_existing_serial_context(
            args,
            serial_store,
            requested_episode_number=requested_episode_number,
            lifecycle_action=lifecycle_action,
        )
    if state.finished:
        return state
    state.context = serial_context
    args.character = str(serial_context["character"])
    serial_context = _review_automatic_serial_outline(
        args,
        project_root,
        output_root,
        serial_store,
        serial_context,
        state,
        timeout_seconds=timeout_seconds,
    )
    if state.finished:
        return state
    state.context = serial_context
    _validate_serial_story_request(
        args,
        serial_context,
        lifecycle_action=lifecycle_action,
    )
    if args.serial_review_episode_number is not None:
        result = _record_serial_manual_review(args, serial_store)
        print(json.dumps(result, ensure_ascii=False))
        state.finished = True
        return state
    state.resume_episode = _validate_serial_resume_episode(
        args,
        serial_store,
        serial_context,
    )
    return state




def _run_serial_publish_review(
    args: argparse.Namespace,
    serial_store: SerialVideoStore | None,
    serial_context: dict[str, object] | None,
    planner: TaskPlanner,
    runner: WorkflowRunner,
    run_memory: RunMemory,
    output_root: Path | None,
    discord_review_timeout_seconds: int,
) -> bool:
    if (
        args.serial_publish_episode_number is None
        or serial_store is None
        or serial_context is None
    ):
        return False

    from agentic.runtime.serial_video import continuity_state_text

    episode = serial_store.get_current_episode(
        args.series_id, args.serial_publish_episode_number
    )
    if episode["status"] != "ready" or not episode.get("final_video_path"):
        raise ValueError("Only an assembled, ready serial episode can enter final publication review.")
    if str(episode["prompt"]) != str(args.goal).strip():
        raise ValueError("Final publication must use the episode's original premise in --goal.")
    try:
        episode_context = json.loads(str(episode.get("story_context_json") or "{}"))
    except (TypeError, ValueError):
        episode_context = {}
    if not isinstance(episode_context, dict):
        episode_context = {}
    accepted_state_before = episode_context.get("approved_story_state_before")
    if not isinstance(accepted_state_before, dict):
        accepted_state_before = {}
    from agentic.runtime.contracts import GoalRequest

    platforms, platform_configs = _serial_publish_configuration(
        REPO_ROOT, args.character, args.platforms, args.character_config
    )
    platforms = serial_store.platforms_without_successful_receipt(
        str(episode["episode_id"]), platforms
    )
    if not platforms:
        raise ValueError(
            "All requested platforms already have a reported-success receipt for this episode."
        )
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
            "serial_story_review_context": {
                "approved_start_state": str(accepted_state_before.get("state_text") or ""),
                "proposed_end_state": str(
                    episode_context.get("continuity_state_proposal")
                    or continuity_state_text(str(episode["prompt"]))
                ),
            },
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
        return True
    print("=== PLAN ===")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    return True




def _prepare_serial_episode_for_execution(
    args: argparse.Namespace,
    serial_store: SerialVideoStore | None,
    serial_context: dict[str, object] | None,
    resume_episode_row: dict[str, object] | None,
    goal: GoalRequest,
    plan: ExecutionPlan,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    episode = None
    resume_initial_outputs: dict[str, object] = {}
    if not args.execute or serial_store is None or serial_context is None:
        return episode, resume_initial_outputs

    try:
        if resume_episode_row is not None:
            episode = resume_episode_row
            resume_state = serial_store.resume_episode(str(episode["episode_id"]))
            resume_initial_outputs = _serial_resume_node_outputs(plan, resume_state)
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
    return episode, resume_initial_outputs


def _run_workflow_with_serial_regeneration(
    runner: WorkflowRunner,
    plan: ExecutionPlan,
    serial_store: SerialVideoStore | None,
    episode: dict[str, object] | None,
    initial_node_outputs: dict[str, object],
) -> WorkflowRunResult:
    try:
        result = runner.run(plan, initial_node_outputs=initial_node_outputs)
        while (
            serial_store is not None
            and episode is not None
            and _serial_segment_regeneration_requested(plan, result)
        ):
            resume_state = serial_store.resume_episode(str(episode["episode_id"]))
            initial_node_outputs = _serial_resume_node_outputs(plan, resume_state)
            result = runner.run(plan, initial_node_outputs=initial_node_outputs)
    except Exception as exc:
        if serial_store is not None and episode is not None:
            serial_store.fail_episode(episode["episode_id"], f"{type(exc).__name__}: {exc}")
        raise
    return result


def _record_serial_workflow_completion(
    serial_store: SerialVideoStore | None,
    episode: dict[str, object] | None,
    plan: ExecutionPlan,
    result: WorkflowRunResult,
) -> None:
    if serial_store is None or episode is None:
        return
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


def _print_workflow_result(result: WorkflowRunResult, run_memory: RunMemory) -> None:
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("=== PROMPTS ===")
    print(
        json.dumps(
            _build_prompt_summary(result.to_dict().get("state", {})),
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=== MEMORY ===")
    print(json.dumps(run_memory.as_serializable(), indent=2, ensure_ascii=False))


def _configure_cli_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def _prepare_missing_auto_next_idea(
    args: argparse.Namespace,
    serial_context: dict[str, object] | None,
    serial_store: SerialVideoStore | None,
) -> None:
    if args.serial_auto_next and (not args.goal or not args.episode_topic):
        from agentic.runtime.llm_engine import LLMPromptEngine

        idea = LLMPromptEngine(
            mode=os.environ.get("AGENTIC_LLM_MODE", "llm")
        ).generate_serial_episode_idea(
            character=args.character,
            automation_mode=str(serial_context.get("automation_mode") or "new"),
            story_context=str(serial_context.get("story_context") or ""),
            previous_episodes=_earlier_episode_actions(
                serial_store,
                str(serial_context["series_id"]),
                int(serial_context["episode_number"]),
            ),
        )
        args.goal = args.goal or str(idea["premise"])
        args.episode_topic = args.episode_topic or str(idea["topic"])


def _apply_series_look(
    args: argparse.Namespace, serial_context: dict[str, object]
) -> None:
    """Give every episode of a series the canvas and illustration style it was born with."""
    from agentic.app.character_config import load_character_config
    from agentic.app.serial_look import serial_series_look

    config_path = (
        Path(args.character_config).expanduser().resolve()
        if args.character_config
        else REPO_ROOT / "configs" / "characters" / f"{str(args.character).lower()}.yaml"
    )
    look = serial_series_look(
        dict(load_character_config(config_path).get("generation") or {}),
        str(serial_context["series_id"]),
    )
    args.style = look["style"]["label"]
    args.width = int(look["canvas"]["width"])
    args.height = int(look["canvas"]["height"])
    args.serial_style_profile = look["style"]


def _allow_serial_reference_evidence(
    serial_context: dict[str, object] | None,
    project_root: Path,
    output_root: Path | None,
) -> None:
    if serial_context is None:
        return
    serial_reference_roots = [
        Path(str(item["path"])).parent
        for item in serial_context["reference_manifest"]
        if isinstance(item, dict) and item.get("path")
    ]
    _allow_runtime_output_for_visual_evidence(
        output_root or project_root / "output",
        additional_roots=serial_reference_roots,
    )


def _resolve_cli_serial_targets(
    args: argparse.Namespace,
    serial_store: SerialVideoStore | None,
    serial_context: dict[str, object] | None,
    resume_episode_row: dict[str, object] | None,
) -> tuple[list[str], dict[str, dict[str, object]]]:
    serial_platforms: list[str] = []
    serial_platform_configs: dict[str, dict[str, object]] = {}
    if serial_context is None:
        return serial_platforms, serial_platform_configs
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
            platform: serial_platform_configs[platform] for platform in serial_platforms
        }
    return serial_platforms, serial_platform_configs


def _build_cli_goal(
    planner: TaskPlanner,
    args: argparse.Namespace,
    serial_store: SerialVideoStore | None,
    serial_context: dict[str, object] | None,
    serial_platforms: list[str],
    serial_platform_configs: dict[str, dict[str, object]],
    resume_episode_row: dict[str, object] | None,
    output_root: Path | None,
    discord_review_timeout_seconds: int,
) -> GoalRequest:
    constraints: dict[str, object] = {
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
        "discord_review_timeout_seconds": discord_review_timeout_seconds,
    }
    if (args.width is None) != (args.height is None):
        raise ValueError("Canvas width and height must be provided together")
    if args.width is not None and args.height is not None:
        constraints["canvas_width"] = args.width
        constraints["canvas_height"] = args.height
        constraints["story_card_width"] = args.width
        constraints["story_card_height"] = args.height
    else:
        character_config_path = None
        if args.character_config:
            character_config_path = Path(args.character_config).expanduser().resolve()
        elif args.character:
            character_config_path = (
                REPO_ROOT
                / "configs"
                / "characters"
                / f"{str(args.character).strip().lower()}.yaml"
            )
        if character_config_path is not None and character_config_path.is_file():
            from agentic.app.character_config import load_character_config
            from agentic.app.creative_profiles import resolve_creative_profile
            from agentic.runtime.story_cards import story_card_canvas

            character_config = load_character_config(character_config_path)
            generation_config = dict(character_config.get("generation", {}) or {})
            canvas_profile = resolve_creative_profile(
                dict(generation_config.get("creative_profile") or {})
            )["canvas"]
            constraints["canvas_profile"] = canvas_profile
            constraints["canvas_aspect_ratio"] = str(canvas_profile["aspect_ratio"])
            constraints["canvas_width"] = int(canvas_profile["width"])
            constraints["canvas_height"] = int(canvas_profile["height"])
            (
                constraints["story_card_width"],
                constraints["story_card_height"],
            ) = story_card_canvas(generation_config.get("story_card"))
    if serial_context is not None and serial_store is not None:
        from agentic.runtime.serial_video import continuity_state_text

        constraints.update(
            {
                "serial_mode": True,
                "series_id": serial_context["series_id"],
                "serial_episode_number": serial_context["episode_number"],
                "serial_database_name": serial_store.database_name,
                "episode_topic": args.episode_topic,
                "serial_automation_mode": serial_context.get("automation_mode", "manual"),
                "serial_automation_source_series_id": serial_context.get(
                    "automation_source_series_id", ""
                ),
                "serial_automation_source_episode_id": serial_context.get(
                    "automation_source_episode_id", ""
                ),
                "platforms": serial_platforms,
                "platform_configs": serial_platform_configs,
                "publish_mode": args.publish_mode or "live",
                "visual_style_profile": args.serial_style_profile,
                "serial_story_context": serial_context["story_context"],
                "serial_story_review_context": {
                    "approved_start_state": str(
                        (serial_context.get("story_state") or {}).get("state_text") or ""
                    ),
                    "proposed_end_state": continuity_state_text(str(args.goal or "")),
                },
                "reference_manifest": serial_context["reference_manifest"],
                "longvideo_review_policy": "every_segment",
                "longvideo_reference_max_images": 4,
                "longvideo_reference_selection_limit": 4,
                "require_human_review": True,
                "stage_probe_auto_select": False,
                "enable_review_loop": False,
                "video_count": 1,
            }
        )
    if resume_episode_row is not None:
        constraints["serial_episode_id"] = str(resume_episode_row["episode_id"])
    return planner.create_goal(
        prompt=args.goal,
        media_type=args.media_type,
        duration_seconds=args.duration_seconds,
        style=args.style,
        constraints=constraints,
    )
def main(argv: Sequence[str] | None = None) -> dict[str, object] | None:
    _configure_cli_output_streams()

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
    handled, onboarding_result = _run_serial_onboarding_command(args, project_root)
    if handled:
        return onboarding_result
    serial_execution = _prepare_serial_execution_state(
        args,
        project_root,
        output_root,
        timeout_seconds=discord_review_timeout_seconds,
    )
    if serial_execution.finished:
        return serial_execution.result
    serial_store = serial_execution.store
    serial_context = serial_execution.context
    resume_episode_row = serial_execution.resume_episode
    resume_initial_outputs = serial_execution.resume_initial_outputs
    planner, runner, run_memory = build_runtime(
        project_root,
        output_root=output_root,
        comfy_host=args.comfy_host,
        comfy_port=args.comfy_port,
        comfy_root=Path(args.comfy_root).resolve() if args.comfy_root else None,
        input_roots=[Path(root) for root in args.edit_input_roots or []],
    )
    if serial_context is not None:
        _apply_series_look(args, serial_context)
    _prepare_missing_auto_next_idea(args, serial_context, serial_store)

    _allow_serial_reference_evidence(
        serial_context, project_root, output_root
    )

    if _run_serial_publish_review(
        args,
        serial_store,
        serial_context,
        planner,
        runner,
        run_memory,
        output_root,
        discord_review_timeout_seconds,
    ):
        return

    serial_platforms, serial_platform_configs = _resolve_cli_serial_targets(
        args, serial_store, serial_context, resume_episode_row
    )

    goal = _build_cli_goal(
        planner,
        args,
        serial_store,
        serial_context,
        serial_platforms,
        serial_platform_configs,
        resume_episode_row,
        output_root,
        discord_review_timeout_seconds,
    )

    plan, resume_episode_row = _prepare_serial_execution_plan(
        planner,
        goal,
        serial_store,
        resume_episode_row,
        discord_review_timeout_seconds=discord_review_timeout_seconds,
    )
    episode, resume_initial_outputs = _prepare_serial_episode_for_execution(
        args,
        serial_store,
        serial_context,
        resume_episode_row,
        goal,
        plan,
    )

    print("=== PLAN ===")
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))

    if not args.execute:
        return

    print("=== RUN ===")
    result = _run_workflow_with_serial_regeneration(
        runner,
        plan,
        serial_store,
        episode,
        resume_initial_outputs,
    )
    _record_serial_workflow_completion(serial_store, episode, plan, result)
    _print_workflow_result(result, run_memory)

    completion = _serial_completion_result(
        args,
        serial_store,
        serial_context,
        episode,
        plan,
        result,
    )
    if completion is not None:
        return completion


if __name__ == "__main__":
    completion = main()
    if (
        isinstance(completion, dict)
        and completion.get("generation_type") == "serial_video"
        and completion.get("status") == "failed"
    ):
        raise SystemExit(2)
