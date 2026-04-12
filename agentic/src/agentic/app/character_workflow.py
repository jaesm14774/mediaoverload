from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import yaml

from agentic.app.main import _build_prompt_summary, build_runtime
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.step_logger import create_run_logger
from agentic.tools.context_services import NewsContextService

SUPPORTED_PUBLISH_PLATFORMS = {"twitter", "facebook", "instagram_graph"}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".avi", ".webm"}

CONFIG_MEDIA_TYPE_MAP = {
    "text2img": "image",
    "text2video": "text2video",
    "text2image2video": "text2img2video",
    "text2longvideo": "long_video",
    "image2image": "image",
    "text2image2image": "text2img2img",
    "sticker_pack": "sticker_pack",
}


def load_character_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_global_social_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "social_media" / "platforms.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, dict) else {}


def choose_media_type(
    config: dict[str, Any],
    preferred_generation_type: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    generation = dict(config.get("generation", {}) or {})
    weights = dict(generation.get("generation_type_weights", {}) or {})
    if preferred_generation_type:
        config_generation_type = preferred_generation_type
    else:
        config_generation_type = _weighted_choice(weights, rng=rng)
    agentic_media_type = CONFIG_MEDIA_TYPE_MAP.get(config_generation_type, "long_video")
    return config_generation_type, agentic_media_type


def build_goal_payload_from_character_config(
    repo_root: Path,
    config_path: str | Path,
    *,
    prompt: str = "",
    temperature: float = 1.0,
    preferred_generation_type: str | None = None,
    dry_run_publish: bool = False,
    publish_after_generate: bool = True,
    output_dir: str | None = None,
    enable_review_loop: bool = False,
    review_notes: str = "",
    rng: random.Random | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = load_character_config(path)
    character = dict(config.get("character", {}) or {})
    generation = dict(config.get("generation", {}) or {})
    additional_params = dict(config.get("additional_params", {}) or {})
    strategies = dict(additional_params.get("strategies", {}) or {})
    social_media = _merge_social_media_config(
        dict(load_global_social_config(repo_root).get("social_media", {}) or {}),
        dict(config.get("social_media", {}) or {}),
    )

    style = _pick_primary_style(generation.get("style_weights"))
    character_name = str(character.get("name") or character.get("group_name") or path.stem)
    routing = _route_generation_from_character_config(
        repo_root,
        config,
        character_name=character_name,
        style=style,
        prompt=prompt,
        preferred_generation_type=preferred_generation_type,
        rng=rng,
    )
    config_generation_type = str(routing["generation_type"])
    agentic_media_type = CONFIG_MEDIA_TYPE_MAP.get(config_generation_type, "long_video")
    autonomous_prompt = _resolve_autonomous_prompt(
        prompt=prompt,
        character_name=character_name,
        style=style,
        media_type=agentic_media_type,
        generation=generation,
    )
    resolved_prompt = str(autonomous_prompt["prompt"]).strip() or _build_default_prompt(character_name, agentic_media_type, style)
    resolved_output_dir = _resolve_output_dir(
        repo_root,
        output_dir or generation.get("output_dir"),
        character_name,
    )
    duration_seconds = _resolve_duration_seconds(
        config_generation_type,
        strategies,
        routed_segment_count=routing.get("count_plan", {}).get("segment_count"),
    )
    platform_configs, platform_aliases, skipped_platforms = _normalize_platform_configs(
        repo_root,
        social_media.get("platforms"),
    )
    hashtags = [str(tag) for tag in (social_media.get("default_hashtags") or []) if tag]
    use_tts = bool(
        (
            dict(strategies.get("text2longvideo", {}) or {})
            .get("longvideo_config", {})
            or {}
        ).get("use_tts", False)
    )

    constraints = {
        "character": character_name,
        "hashtags": hashtags,
        "platforms": list(platform_configs.keys()),
        "platform_configs": platform_configs,
        "dry_run": dry_run_publish,
        "selection_limit": routing.get("count_plan", {}).get("review_selection_limit"),
        "enable_review_loop": enable_review_loop,
        "review_notes": review_notes,
        "output_dir": str(resolved_output_dir),
        "use_tts": use_tts if agentic_media_type == "long_video" else False,
        "source_temperature": temperature,
        "source_config_path": str(path),
        "source_generation_type": config_generation_type,
        "workflow_name": routing.get("workflow_name", ""),
        "image_workflow_name": routing.get("workflow_plan", {}).get("image_workflow_name", ""),
        "video_workflow_name": routing.get("workflow_plan", {}).get("video_workflow_name", ""),
        "refine_workflow_name": routing.get("workflow_plan", {}).get("refine_workflow_name", ""),
        "transition_workflow_name": routing.get("workflow_plan", {}).get("transition_workflow_name", ""),
        "upscale_workflow_name": routing.get("workflow_plan", {}).get("upscale_workflow_name", ""),
        "image_count": routing.get("count_plan", {}).get("image_count"),
        "video_count": routing.get("count_plan", {}).get("video_count"),
        "segment_count": routing.get("count_plan", {}).get("segment_count"),
        "review_selection_limit": routing.get("count_plan", {}).get("review_selection_limit"),
        "sticker_expression_count": routing.get("count_plan", {}).get("sticker_expression_count"),
        "images_per_prompt": routing.get("count_plan", {}).get("images_per_prompt"),
        "routing_reason": routing.get("reason", ""),
        "routing_prompt_mode": routing.get("prompt_mode", ""),
        "workflow_stage_candidates": routing.get("workflow_stage_candidates", {}),
        "generation_type_candidates": routing.get("generation_type_candidates", []),
        "count_policies": routing.get("count_policies", {}),
        "publish_after_generate": publish_after_generate,
        "platform_aliases": platform_aliases,
        "skipped_platforms": skipped_platforms,
        "prompt_source": autonomous_prompt.get("source", "user"),
        "prompt_mode": autonomous_prompt.get("prompt_mode", "user"),
        "creative_seed": autonomous_prompt.get("creative_seed", ""),
        "news_context": autonomous_prompt.get("news_context", {}),
        "fallback_reason": autonomous_prompt.get("fallback_reason", ""),
        "enable_stage_review": bool(
            os.getenv("discord_review_bot_token")
            and os.getenv("discord_review_channel_id")
            and agentic_media_type in {"text2video", "text2img2video", "long_video"}
        ),
    }
    return {
        "prompt": resolved_prompt,
        "media_type": agentic_media_type,
        "duration_seconds": duration_seconds,
        "style": style,
        "auto_download_assets": False,
        "constraints": constraints,
        "source_generation_type": config_generation_type,
        "selected_workflow_name": routing.get("workflow_name", ""),
        "character_name": character_name,
        "resolved_output_dir": str(resolved_output_dir),
        "routing": routing,
        "prompt_generation": autonomous_prompt,
        "character_config_summary": _summarize_character_config(
            path=path,
            character=character,
            generation=generation,
            social_media=social_media,
            strategies=strategies,
            platform_configs=platform_configs,
        ),
        "routing_summary": _build_routing_summary(
            character_name=character_name,
            prompt=resolved_prompt,
            source_generation_type=config_generation_type,
            agentic_media_type=agentic_media_type,
            duration_seconds=duration_seconds,
            output_dir=str(resolved_output_dir),
            routing=routing,
        ),
    }


def run_character_workflow(
    repo_root: Path,
    config_path: str | Path,
    *,
    prompt: str = "",
    temperature: float = 1.0,
    preferred_generation_type: str | None = None,
    dry_run_publish: bool = False,
    publish_after_generate: bool = True,
    output_dir: str | None = None,
    enable_review_loop: bool = False,
    review_notes: str = "",
    comfy_host: str | None = None,
    comfy_port: int | None = None,
    comfy_root: str | None = None,
    auto_download_assets: bool = False,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    logger, log_path = create_run_logger(repo_root / "agentic" / "logs" / "runs", run_id)
    os.environ["AGENTIC_RUN_LOGGER_NAME"] = logger.name
    logger.info("workflow.start | run_id=%s | config=%s", run_id, Path(config_path).resolve())
    logger.info("character.config.load | path=%s", Path(config_path).resolve())
    payload = build_goal_payload_from_character_config(
        repo_root,
        config_path,
        prompt=prompt,
        temperature=temperature,
        preferred_generation_type=preferred_generation_type,
        dry_run_publish=dry_run_publish,
        publish_after_generate=publish_after_generate,
        output_dir=output_dir,
        enable_review_loop=enable_review_loop,
        review_notes=review_notes,
        rng=rng,
    )
    logger.info(
        "routing.selected | strategy=%s | media_type=%s | workflow=%s | prompt_source=%s",
        payload["source_generation_type"],
        payload["media_type"],
        payload["selected_workflow_name"],
        payload["constraints"].get("prompt_source", ""),
    )
    logger.info("character.config.summary | payload=%s", json.dumps(payload.get("character_config_summary", {}), ensure_ascii=False))
    if payload["constraints"].get("news_context"):
        logger.info("news.context | payload=%s", json.dumps(payload["constraints"]["news_context"], ensure_ascii=False))
    planner, runner, run_memory = build_runtime(
        repo_root / "agentic",
        output_root=Path(payload["resolved_output_dir"]),
        comfy_host=comfy_host,
        comfy_port=comfy_port,
        comfy_root=Path(comfy_root).resolve() if comfy_root else None,
        run_id=run_id,
    )
    goal = planner.create_goal(
        prompt=payload["prompt"],
        media_type=payload["media_type"],
        duration_seconds=int(payload["duration_seconds"]),
        style=str(payload["style"]),
        auto_download_assets=auto_download_assets,
        constraints=dict(payload["constraints"]),
    )
    plan = planner.build_plan(goal)
    logger.info("plan.built | workflow=%s | node_count=%s", plan.workflow_name, len(plan.nodes))
    generation_result = runner.run(plan)
    generation_dict = generation_result.to_dict()
    logger.info("generation.finished | status=%s", generation_result.status)

    publish_dict: dict[str, Any] | None = None
    if publish_after_generate and generation_result.status == "success":
        media_paths = collect_media_paths_from_run_result(generation_dict)
        platform_configs = payload["constraints"].get("platform_configs", {})
        if media_paths and isinstance(platform_configs, dict) and platform_configs:
            logger.info("publish.start | media_count=%s | platforms=%s", len(media_paths), list(platform_configs.keys()))
            publish_goal = planner.create_goal(
                prompt=payload["prompt"],
                media_type="publish_review",
                duration_seconds=0,
                style=str(payload["style"]),
                auto_download_assets=False,
                constraints={
                    "media_paths": media_paths,
                    "platforms": list(platform_configs.keys()),
                    "platform_configs": dict(platform_configs),
                    "hashtags": list(payload["constraints"].get("hashtags", [])),
                    "character": payload["character_name"],
                    "dry_run": dry_run_publish,
                    "output_dir": str(Path(payload["resolved_output_dir"]) / "publish_ready"),
                    "selection_limit": 4,
                    "review_notes": review_notes,
                },
            )
            publish_plan = planner.build_plan(publish_goal)
            publish_result = runner.run(publish_plan)
            publish_result_dict = publish_result.to_dict()
            logger.info("publish.finished | status=%s", publish_result.status)
            publish_dict = {
                "plan": publish_plan.to_dict(),
                "result": publish_result_dict,
                "prompt_summary": _build_prompt_summary(publish_result_dict.get("state", {})),
            }
    logger.info("workflow.end | run_id=%s | status=%s | log_path=%s", run_id, generation_result.status, log_path)

    return {
        "status": "success" if generation_result.status == "success" else "failed",
        "run_id": run_id,
        "log_path": str(log_path),
        "config_path": str(Path(config_path).resolve()),
        "source_generation_type": payload["source_generation_type"],
        "agentic_media_type": payload["media_type"],
        "character": payload["character_name"],
        "routing_summary": payload.get("routing_summary", {}),
        "plan": plan.to_dict(),
        "generation": {
            "result": generation_dict,
            "prompt_summary": _build_prompt_summary(generation_dict.get("state", {})),
        },
        "routing": payload.get("routing", {}),
        "character_config_summary": payload.get("character_config_summary", {}),
        "publish": publish_dict,
        "memory": run_memory.as_serializable(),
    }


def collect_media_paths_from_run_result(run_result: dict[str, Any]) -> list[str]:
    state = dict(run_result.get("state", {}) or {})
    node_outputs = dict(state.get("node_outputs", {}) or {})
    collected: list[str] = []
    for outputs in node_outputs.values():
        if not isinstance(outputs, dict):
            continue
        for key in ("saved_files", "media_paths", "video_path", "gif_path", "frame_path"):
            value = outputs.get(key)
            if isinstance(value, list):
                collected.extend(str(item) for item in value if _is_media_file(str(item)))
            elif isinstance(value, str) and _is_media_file(value):
                collected.append(value)
    return list(dict.fromkeys(collected))


def dumps_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _pick_primary_style(style_weights: Any) -> str:
    if not isinstance(style_weights, dict) or not style_weights:
        return "cinematic surreal"
    ranked = [
        (str(style), float(weight))
        for style, weight in style_weights.items()
        if str(style).strip()
    ]
    if not ranked:
        return "cinematic surreal"
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0][0]


def _weighted_choice(weights: dict[str, Any], rng: random.Random | None = None) -> str:
    filtered = [(str(name), float(weight)) for name, weight in weights.items() if float(weight) > 0]
    if not filtered:
        return "text2longvideo"
    chooser = rng or random
    total = sum(weight for _, weight in filtered)
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for name, weight in filtered:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return filtered[-1][0]


def _route_generation_from_character_config(
    repo_root: Path,
    config: dict[str, Any],
    *,
    character_name: str,
    style: str,
    prompt: str,
    preferred_generation_type: str | None,
    rng: random.Random | None,
) -> dict[str, Any]:
    del rng
    generation = dict(config.get("generation", {}) or {})
    additional_params = dict(config.get("additional_params", {}) or {})
    strategies = dict(additional_params.get("strategies", {}) or {})
    routing_config = _load_global_routing_config(repo_root)
    routing_overrides = dict(generation.get("routing_overrides", {}) or {})
    merged_routing = _merge_routing_config(routing_config, routing_overrides)
    if merged_routing.get("enabled", True) is False:
        raise ValueError("Global routing is disabled. LLM routing is required for this workflow.")
    generation_type_candidates = _collect_generation_type_candidates(merged_routing, preferred_generation_type)
    workflow_stage_candidates = _collect_workflow_stage_candidates(
        repo_root,
        generation,
        strategies,
        merged_routing,
        generation_type_candidates,
    )
    count_policies = _collect_count_policies(merged_routing, generation_type_candidates)
    if preferred_generation_type:
        selected_generation_type = str(preferred_generation_type).strip()
        workflow_plan = {
            stage_key: (workflow_stage_candidates.get(selected_generation_type, {}).get(stage_key, [""])[0] or "")
            for stage_key in (
                "image_workflow_name",
                "video_workflow_name",
                "refine_workflow_name",
                "transition_workflow_name",
                "upscale_workflow_name",
            )
        }
        count_plan = {
            count_key: _pick_policy_value(policy)
            for count_key, policy in dict(count_policies.get(selected_generation_type, {}) or {}).items()
        }
        return {
            "generation_type": selected_generation_type,
            "workflow_name": _primary_workflow_name(selected_generation_type, workflow_plan),
            "workflow_plan": workflow_plan,
            "count_plan": count_plan,
            "reason": "preferred_generation_type override",
            "prompt_mode": "override",
            "llm_backend": {},
            "workflow_stage_candidates": workflow_stage_candidates,
            "generation_type_candidates": generation_type_candidates,
            "count_policies": count_policies,
        }
    route_prompt = str(prompt).strip() or character_name
    engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    routing_hints = merged_routing.get("routing_hints", {})
    result = engine.route_generation_strategy(
        prompt=route_prompt,
        character=character_name,
        style=style,
        generation_type_candidates=generation_type_candidates,
        workflow_stage_candidates=workflow_stage_candidates,
        count_policies=count_policies,
        routing_hints=routing_hints if isinstance(routing_hints, dict) else {},
        preferred_generation_type=preferred_generation_type,
    )
    selected_generation_type = str(result["generation_type"]).strip()
    workflow_plan = {str(key): str(value or "").strip() for key, value in dict(result["workflow_plan"]).items()}
    count_plan = {str(key): int(value) for key, value in dict(result["count_plan"]).items()}
    return {
        "generation_type": selected_generation_type,
        "workflow_name": _primary_workflow_name(selected_generation_type, workflow_plan),
        "workflow_plan": workflow_plan,
        "count_plan": count_plan,
        "reason": str(result.get("reason") or "").strip(),
        "prompt_mode": str(result["prompt_mode"]),
        "llm_backend": result.get("llm_backend") or engine.backend_info(),
        "workflow_stage_candidates": workflow_stage_candidates,
        "generation_type_candidates": generation_type_candidates,
        "count_policies": count_policies,
    }


def _collect_generation_type_candidates(
    routing_config: dict[str, Any],
    preferred_generation_type: str | None,
) -> list[str]:
    explicit = routing_config.get("strategy_candidates") or routing_config.get("allowed_generation_types") or []
    candidates: list[str] = []
    if preferred_generation_type:
        candidates.append(str(preferred_generation_type).strip())
    elif isinstance(explicit, list):
        candidates.extend(str(item).strip() for item in explicit if str(item).strip())
    ordered: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in ordered:
            continue
        ordered.append(candidate)
    if not ordered:
        raise ValueError("Routing config must provide at least one strategy candidate.")
    return ordered


def _collect_workflow_stage_candidates(
    repo_root: Path,
    generation: dict[str, Any],
    strategies: dict[str, Any],
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
) -> dict[str, dict[str, list[str]]]:
    workflow_stage_candidates: dict[str, dict[str, list[str]]] = {}
    explicit_candidates = routing_config.get("workflow_stage_candidates", {}) or routing_config.get("workflow_candidates", {})
    if not isinstance(explicit_candidates, dict):
        explicit_candidates = {}
    generation_workflows = dict(generation.get("workflows", {}) or {})

    for generation_type in generation_type_candidates:
        stage_candidates: dict[str, list[str]] = {}
        explicit = explicit_candidates.get(generation_type, {})
        strategy = dict(strategies.get(generation_type, {}) or {})
        if isinstance(explicit, list):
            explicit = {"image_workflow_name": explicit}
        if not isinstance(explicit, dict):
            explicit = {}

        inferred: dict[str, list[Any]] = {}
        if generation_type == "sticker_pack":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "static_config", "workflow_name"),
                    _nested_get(strategy, "static_config", "workflow_path"),
                    generation_workflows.get("sticker_pack"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "animated_config", "workflow_name"),
                    _nested_get(strategy, "animated_config", "i2v_workflow_path"),
                ],
            }
        elif generation_type == "text2image2video":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "t2i_workflow_path"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "video", "workflow_name"),
                    _nested_get(strategy, "video", "i2v_workflow_path"),
                    generation_workflows.get("text2image2video"),
                ],
                "upscale_workflow_name": [
                    _nested_get(strategy, "first_stage", "upscale_workflow_name"),
                    _nested_get(strategy, "first_stage", "upscale_workflow_path"),
                ],
            }
        elif generation_type == "text2image2image":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "refine_workflow_name": [
                    _nested_get(strategy, "second_stage", "workflow_name"),
                    _nested_get(strategy, "second_stage", "workflow_path"),
                ],
            }
        elif generation_type == "text2longvideo":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "video_generation", "workflow_name"),
                    _nested_get(strategy, "video_generation", "workflow_path"),
                    generation_workflows.get("text2video"),
                ],
                "transition_workflow_name": [
                    _nested_get(strategy, "frame_transition", "workflow_name"),
                    _nested_get(strategy, "frame_transition", "workflow_path"),
                ],
            }
        elif generation_type == "image2image":
            inferred = {
                "refine_workflow_name": [
                    _nested_get(strategy, "workflow_name"),
                    _nested_get(strategy, "workflow_path"),
                    generation_workflows.get("image2image"),
                ],
            }
        elif generation_type == "text2video":
            inferred = {
                "image_workflow_name": [
                    generation_workflows.get("text2img"),
                ],
            }
        else:
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "workflow_name"),
                    _nested_get(strategy, "workflow_path"),
                    generation_workflows.get(generation_type),
                ],
            }

        for stage_key in (
            "image_workflow_name",
            "video_workflow_name",
            "refine_workflow_name",
            "transition_workflow_name",
            "upscale_workflow_name",
        ):
            stage_references: list[Any] = []
            explicit_stage = explicit.get(stage_key, [])
            if explicit_stage:
                if isinstance(explicit_stage, list):
                    stage_references.extend(explicit_stage)
                else:
                    stage_references.append(explicit_stage)
            stage_references.extend(inferred.get(stage_key, []))
            resolved = _resolve_workflow_references(repo_root, stage_references)
            if resolved:
                workflow_stage_candidates.setdefault(generation_type, {})[stage_key] = resolved
        workflow_stage_candidates.setdefault(generation_type, {})
    return workflow_stage_candidates


def _collect_count_policies(
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    raw_policies = routing_config.get("count_policies", {})
    if not isinstance(raw_policies, dict):
        raw_policies = {}
    policies: dict[str, dict[str, dict[str, int]]] = {}
    for generation_type in generation_type_candidates:
        strategy_policy = dict(raw_policies.get(generation_type, {}) or {})
        normalized: dict[str, dict[str, int]] = {}
        for count_key in (
            "image_count",
            "video_count",
            "segment_count",
            "review_selection_limit",
            "sticker_expression_count",
            "images_per_prompt",
        ):
            raw = strategy_policy.get(count_key, {})
            if not isinstance(raw, dict):
                continue
            minimum = int(raw.get("min", 1))
            maximum = int(raw.get("max", minimum))
            normalized[count_key] = {"min": minimum, "max": max(minimum, maximum)}
        policies[generation_type] = normalized
    return policies


def _pick_policy_value(policy: dict[str, Any]) -> int:
    if not isinstance(policy, dict):
        return 1
    minimum = int(policy.get("min", 1))
    maximum = int(policy.get("max", minimum))
    return minimum if minimum <= maximum else maximum


def _load_global_routing_config(repo_root: Path) -> dict[str, Any]:
    routing_path = repo_root / "configs" / "routing.yaml"
    if not routing_path.exists():
        raise FileNotFoundError(f"Global routing config not found: {routing_path}")
    data = yaml.safe_load(routing_path.read_text(encoding="utf-8")) or {}
    if "routing" in data and isinstance(data["routing"], dict):
        return dict(data["routing"])
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid routing config structure: {routing_path}")


def _merge_routing_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_routing_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _workflow_definitions_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "workflow"


def _list_workflow_stems(repo_root: Path) -> set[str]:
    wf_dir = _workflow_definitions_dir(repo_root)
    if not wf_dir.is_dir():
        return set()
    return {path.stem for path in wf_dir.glob("*.json")}


def _resolve_workflow_references(repo_root: Path, references: list[Any]) -> list[str]:
    stems = _list_workflow_stems(repo_root)
    wf_dir = _workflow_definitions_dir(repo_root)
    resolved: list[str] = []
    for reference in references:
        name = _resolve_workflow_name_from_reference(repo_root, stems, wf_dir, reference)
        if name and name not in resolved:
            resolved.append(name)
    return resolved


def _resolve_workflow_name_from_reference(
    repo_root: Path,
    stems: set[str],
    wf_dir: Path,
    reference: Any,
) -> str | None:
    normalized = str(reference or "").strip()
    if not normalized:
        return None
    if normalized in stems:
        return normalized
    resolved_reference = _resolve_repo_path(repo_root, normalized).resolve()
    if not wf_dir.is_dir():
        return None
    for path in sorted(wf_dir.glob("*.json")):
        if path.resolve() == resolved_reference:
            return path.stem
    return None


def _primary_workflow_name(generation_type: str, workflow_plan: dict[str, str]) -> str:
    stage_priority = {
        "text2img": ("image_workflow_name",),
        "text2video": ("video_workflow_name", "image_workflow_name"),
        "text2image2video": ("image_workflow_name", "video_workflow_name", "upscale_workflow_name"),
        "text2image2image": ("image_workflow_name", "refine_workflow_name"),
        "text2longvideo": ("image_workflow_name", "video_workflow_name", "transition_workflow_name"),
        "sticker_pack": ("image_workflow_name", "video_workflow_name"),
        "image2image": ("refine_workflow_name",),
    }
    for key in stage_priority.get(generation_type, ("image_workflow_name", "video_workflow_name", "refine_workflow_name")):
        value = str(workflow_plan.get(key) or "").strip()
        if value:
            return value
    return ""


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_duration_seconds(
    config_generation_type: str,
    strategies: dict[str, Any],
    routed_segment_count: Any | None = None,
) -> int:
    if config_generation_type != "text2longvideo":
        return 30
    longvideo = dict(strategies.get("text2longvideo", {}) or {})
    longvideo_config = dict(longvideo.get("longvideo_config", {}) or {})
    segment_count = int(routed_segment_count or longvideo_config.get("segment_count", 3))
    segment_duration = int(longvideo_config.get("segment_duration", 5))
    return max(10, segment_count * segment_duration)


def _build_routing_summary(
    *,
    character_name: str,
    prompt: str,
    source_generation_type: str,
    agentic_media_type: str,
    duration_seconds: int,
    output_dir: str,
    routing: dict[str, Any],
) -> dict[str, Any]:
    workflow_plan = {
        key: str(value).strip()
        for key, value in dict(routing.get("workflow_plan", {}) or {}).items()
        if str(value).strip()
    }
    count_plan = {
        key: int(value)
        for key, value in dict(routing.get("count_plan", {}) or {}).items()
        if value not in {None, ""}
    }
    backend = dict(routing.get("llm_backend", {}) or {})
    backend_summary = {
        "mode": str(backend.get("mode", "")),
        "text_provider": str(backend.get("text_provider", "")),
        "text_model": str(backend.get("text_model", "")),
        "vision_provider": str(backend.get("vision_provider", "")),
        "vision_model": str(backend.get("vision_model", "")),
    }
    return {
        "character": character_name,
        "prompt": prompt,
        "strategy": source_generation_type,
        "agentic_media_type": agentic_media_type,
        "primary_workflow": str(routing.get("workflow_name", "")),
        "workflow_plan": workflow_plan,
        "count_plan": count_plan,
        "reason": str(routing.get("reason", "")),
        "prompt_mode": str(routing.get("prompt_mode", "")),
        "llm_backend": backend_summary,
        "duration_seconds": int(duration_seconds),
        "output_dir": output_dir,
    }


def _build_default_prompt(character_name: str, media_type: str, style: str) -> str:
    media_label = media_type.replace("_", " ")
    return f"{character_name} in a {style} {media_label} concept"


def _resolve_autonomous_prompt(
    *,
    prompt: str,
    character_name: str,
    style: str,
    media_type: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    explicit_prompt = str(prompt).strip()
    if explicit_prompt:
        return {
            "prompt": explicit_prompt,
            "source": "user",
            "prompt_mode": "user",
            "creative_seed": "",
            "news_context": {},
        }

    news_context: dict[str, Any] = {}
    try:
        selected_news = NewsContextService().get_random_news()
        if selected_news is not None:
            news_context = selected_news.to_dict()
    except Exception as exc:
        news_context = {"error": f"{type(exc).__name__}: {exc}"}

    bundle = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm")).generate_autonomous_scene_prompt(
        character=character_name,
        style=style,
        media_type=media_type,
        news_context=news_context,
    )
    bundle.setdefault("prompt_mode", "template")
    bundle.setdefault("source", "autonomous_fallback")
    return bundle


def _normalize_platform_configs(repo_root: Path, raw_platforms: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    if not isinstance(raw_platforms, dict):
        return {}, {}, []
    configs: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    skipped: list[str] = []
    for platform_name, platform_config in raw_platforms.items():
        if not isinstance(platform_config, dict) or not bool(platform_config.get("enabled", True)):
            continue
        normalized = str(platform_name).lower().strip()
        if normalized == "instagram":
            aliases[normalized] = "instagram_graph"
            normalized = "instagram_graph"
        if normalized not in SUPPORTED_PUBLISH_PLATFORMS:
            skipped.append(str(platform_name))
            continue
        resolved = dict(platform_config)
        if "config_folder_path" in resolved:
            resolved["config_folder_path"] = str(_resolve_repo_path(repo_root, str(resolved["config_folder_path"])))
        configs[normalized] = resolved
    return configs, aliases, skipped


def _merge_social_media_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_social_media_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _summarize_character_config(
    *,
    path: Path,
    character: dict[str, Any],
    generation: dict[str, Any],
    social_media: dict[str, Any],
    strategies: dict[str, Any],
    platform_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    longvideo = dict(strategies.get("text2longvideo", {}) or {})
    longvideo_config = dict(longvideo.get("longvideo_config", {}) or {})
    return {
        "config_path": str(path),
        "character_name": str(character.get("name") or character.get("group_name") or path.stem),
        "group_name": str(character.get("group_name") or ""),
        "output_dir": str(generation.get("output_dir") or ""),
        "generation_type_weights": dict(generation.get("generation_type_weights", {}) or {}),
        "style_count": len(dict(generation.get("style_weights", {}) or {})),
        "default_hashtags": [str(tag) for tag in (social_media.get("default_hashtags") or []) if tag],
        "enabled_platforms": list(platform_configs.keys()),
        "platform_config_folders": {
            name: str(config.get("config_folder_path", ""))
            for name, config in platform_configs.items()
        },
        "longvideo": {
            "segment_count": int(longvideo_config.get("segment_count", 0) or 0),
            "segment_duration": int(longvideo_config.get("segment_duration", 0) or 0),
            "use_tts": bool(longvideo_config.get("use_tts", False)),
            "tts_voice": str(longvideo_config.get("tts_voice", "")),
        },
    }


def _resolve_output_dir(repo_root: Path, configured_output_dir: Any, character_name: str) -> Path:
    base = _resolve_repo_path(repo_root, str(configured_output_dir or (repo_root / "agentic" / "output")))
    return Path(base) / character_name.lower()


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        return repo_root / "agentic" / "output"
    if normalized.startswith("/app/"):
        return repo_root / normalized.removeprefix("/app/")
    if normalized == "/app":
        return repo_root
    path = Path(normalized)
    if path.is_absolute():
        return path
    return repo_root / path


def _is_media_file(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS
