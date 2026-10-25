"""Generation Routing responsibilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any
import yaml
from agentic.assets.registry import AssetRegistry
from agentic.h3_reference import normalize_reference_manifest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.runtime.prompt_requests import GenerationRoutingRequest
from agentic.runtime.route_selection import select_weighted_route
from agentic.runtime.strategies import (
    CONFIG_MEDIA_TYPE_MAP,
    NATIVE_STORY_GENERATION_TYPES,
    WORKFLOW_STAGE_KEYS,
    active_count_policy,
    media_type_for_generation,
    primary_workflow_name,
)
from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    normalize_video_duration,
)
from agentic.runtime.news_grounding import (
    is_brand_safe_news_selection,
    is_usable_news_selection,
)
from agentic.tools.context_services import NewsContextService
from agentic.app.character_config import _normalize_generation_type, _resolve_repo_path


def _resolve_routing_asset_root(
    repo_root: Path, explicit_root: str | None
) -> Path | None:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    configured_root = os.environ.get("COMFYUI_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    container_root = Path("/comfyui")
    if container_root.is_dir():
        return container_root.resolve()
    portable_root = Path(r"D:\ComfyUI_windows_portable")
    if portable_root.is_dir():
        return portable_root.resolve()
    return None


def _weighted_choice(weights: dict[str, Any], rng: random.Random | None = None) -> str:
    filtered = [
        (str(name), float(weight))
        for name, weight in weights.items()
        if float(weight) > 0
    ]
    if not filtered:
        raise ValueError("Routing requires at least one positive strategy weight")
    chooser = rng or random
    total = sum(weight for _, weight in filtered)
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for name, weight in filtered:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return filtered[-1][0]


def _weighted_candidate_choice(
    weights: dict[str, Any],
    candidates: list[str],
    *,
    rng: random.Random | None = None,
) -> str:
    """Pick a configured strategy without selecting an unavailable route."""
    allowed = {
        str(candidate).strip() for candidate in candidates if str(candidate).strip()
    }
    normalized_weights: dict[str, float] = {}
    for name, raw_weight in weights.items():
        normalized = str(name).strip()
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if normalized and normalized in allowed and weight > 0:
            normalized_weights[normalized] = weight
    if not normalized_weights:
        normalized_weights = {
            candidate: 1.0 for candidate in candidates if str(candidate).strip()
        }
    return select_weighted_route(
        normalized_weights,
        list(normalized_weights),
        rng=rng,
    )


def _select_workflow_candidate(
    candidates: list[str],
    *,
    weights: dict[str, Any],
    rng: random.Random | None,
    asset_registry: AssetRegistry | None,
) -> str:
    if not candidates:
        return ""
    qualified = _asset_qualified_workflow_candidates(candidates, asset_registry)
    if rng is None:
        return qualified[0]
    positive_weights: dict[str, Any] = {
        candidate: weights.get(candidate, 1)
        for candidate in qualified
        if _positive_weight(weights.get(candidate, 1))
    }
    return _weighted_choice(
        positive_weights or {candidate: 1 for candidate in qualified}, rng=rng
    )


def _positive_weight(value: Any) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _asset_qualified_workflow_candidates(
    candidates: list[str],
    asset_registry: AssetRegistry | None,
) -> list[str]:
    if asset_registry is None:
        return candidates
    qualified: list[str] = []
    for candidate in candidates:
        try:
            statuses = asset_registry.ensure_workflow_ready(candidate).get(
                "asset_status", []
            )
        except (KeyError, OSError, ValueError):
            continue
        # An empty asset manifest means "unverified", not "ready". Treating
        # it as ready made stage weights select workflow JSONs whose model
        # files are not installed on the active ComfyUI host.
        if not isinstance(statuses, list) or not statuses:
            continue
        if all(
            isinstance(item, dict) and str(item.get("status") or "").lower() == "ready"
            for item in statuses
        ):
            qualified.append(candidate)
    # If no candidate is verified, keep one deterministic fallback so the
    # runtime asset-check node can emit the precise missing-asset diagnostic;
    # do not let an unverified list re-enter weighted selection.
    return qualified or candidates[:1]


def _route_generation_from_character_config(
    repo_root: Path,
    config: dict[str, Any],
    *,
    character_name: str,
    style: str,
    prompt: str,
    preferred_generation_type: str | None,
    requested_duration_seconds: int | None,
    rng: random.Random | None,
    news_driven: bool = False,
    asset_root: Path | None = None,
) -> dict[str, Any]:
    generation = dict(config.get("generation", {}) or {})
    additional_params = dict(config.get("additional_params", {}) or {})
    strategies = dict(additional_params.get("strategies", {}) or {})
    routing_config = _load_global_routing_config(repo_root)
    preferred_generation_type = _normalize_generation_type(preferred_generation_type)
    routing_overrides = dict(generation.get("routing_overrides", {}) or {})
    merged_routing = _merge_routing_config(routing_config, routing_overrides)
    if merged_routing.get("enabled", True) is False:
        raise ValueError(
            "Global routing is disabled. LLM routing is required for this workflow."
        )
    routing_runtime_context = _build_h3_reference_runtime_context(repo_root, generation)
    if not preferred_generation_type:
        serial_video_config = generation.get("serial_video", {}) or {}
        if not isinstance(serial_video_config, dict):
            raise ValueError("generation.serial_video must be a mapping")
        if not bool(serial_video_config.get("enabled", False)):
            merged_routing = dict(merged_routing)
            for candidate_key in ("strategy_candidates", "allowed_generation_types"):
                candidates = merged_routing.get(candidate_key)
                if isinstance(candidates, list):
                    merged_routing[candidate_key] = [
                        candidate for candidate in candidates if candidate != "serial_video"
                    ]
    generation_type_candidates = _collect_generation_type_candidates(
        merged_routing,
        preferred_generation_type,
        routing_runtime_context=routing_runtime_context,
    )
    workflow_stage_candidates = _collect_workflow_stage_candidates(
        repo_root,
        generation,
        strategies,
        merged_routing,
        generation_type_candidates,
    )
    count_policies = _collect_count_policies(
        merged_routing,
        generation_type_candidates,
        workflow_stage_candidates=workflow_stage_candidates,
    )
    asset_registry = (
        AssetRegistry(repo_root / "agentic", asset_root=asset_root)
        if asset_root is not None
        else None
    )
    workflow_selection_weights = dict(
        merged_routing.get("workflow_selection_weights", {}) or {}
    )

    def build_fixed_route(
        selected_generation_type: str,
        reason: str,
        *,
        selection_source: str,
        prompt_mode: str,
    ) -> dict[str, Any]:
        workflow_plan: dict[str, str] = {}
        for stage_key in (
            "image_workflow_name",
            "video_workflow_name",
            "refine_workflow_name",
            "transition_workflow_name",
            "upscale_workflow_name",
        ):
            candidates = list(
                workflow_stage_candidates.get(selected_generation_type, {}).get(
                    stage_key, []
                )
            )
            workflow_plan[stage_key] = _select_workflow_candidate(
                candidates,
                weights=dict(workflow_selection_weights.get(stage_key, {}) or {}),
                # An explicit generation-type override fixes only the route
                # family. Keep stage workflow selection weighted when the
                # scheduler supplied an RNG; otherwise every explicit I2V
                # request silently falls back to the first candidate (Krea).
                rng=rng,
                asset_registry=asset_registry,
            )
        count_plan = {
            count_key: _pick_policy_value(policy)
            for count_key, policy in dict(
                count_policies.get(selected_generation_type, {}) or {}
            ).items()
        }
        return {
            "generation_type": selected_generation_type,
            "workflow_name": primary_workflow_name(
                selected_generation_type, workflow_plan
            ),
            "workflow_plan": workflow_plan,
            "count_plan": count_plan,
            "reason": reason,
            "prompt_mode": prompt_mode,
            "selection_source": selection_source,
            "workflow_selection_mode": "weighted_random"
            if rng is not None
            else "first_qualified",
            "llm_backend": {},
            "workflow_stage_candidates": workflow_stage_candidates,
            "generation_type_candidates": generation_type_candidates,
            "count_policies": count_policies,
            "routing_runtime_context": routing_runtime_context,
            "pre_video_review": dict(merged_routing.get("pre_video_review", {}) or {}),
            "strategy_context": _strategy_context_for_route(
                merged_routing, selected_generation_type
            ),
        }

    if preferred_generation_type:
        selected_generation_type = str(preferred_generation_type).strip()
        return build_fixed_route(
            selected_generation_type,
            "preferred_generation_type override",
            selection_source="explicit_override",
            prompt_mode="override",
        )

    duration_strategy = _duration_strategy_override(
        requested_duration_seconds,
        generation_type_candidates,
    )
    if duration_strategy:
        return build_fixed_route(
            duration_strategy,
            f"duration policy selected {duration_strategy} for {int(requested_duration_seconds)} seconds",
            selection_source="duration_policy",
            prompt_mode="duration_policy",
        )

    if rng is not None:
        selected_generation_type = _weighted_candidate_choice(
            dict(generation.get("generation_type_weights", {}) or {}),
            generation_type_candidates,
            rng=rng,
        )
        return build_fixed_route(
            selected_generation_type,
            f"weighted random selected {selected_generation_type} from character generation_type_weights",
            selection_source="weighted_random",
            prompt_mode="weighted_random",
        )

    route_prompt = str(prompt).strip() or character_name
    engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    raw_routing_hints = merged_routing.get("routing_hints", {})
    routing_hints = (
        dict(raw_routing_hints) if isinstance(raw_routing_hints, dict) else {}
    )
    routing_hints["runtime_context"] = routing_runtime_context
    result = engine.route_generation_strategy(
        GenerationRoutingRequest(
            prompt=route_prompt,
            character=character_name,
            style=style,
            generation_type_candidates=tuple(generation_type_candidates),
            workflow_stage_candidates=workflow_stage_candidates,
            count_policies=count_policies,
            routing_hints=routing_hints if isinstance(routing_hints, dict) else {},
            preferred_generation_type=preferred_generation_type,
        )
    )
    selected_generation_type = str(result["generation_type"]).strip()
    workflow_plan = {
        str(key): str(value or "").strip()
        for key, value in dict(result["workflow_plan"]).items()
    }
    count_plan = {
        str(key): int(value) for key, value in dict(result["count_plan"]).items()
    }
    return {
        "generation_type": selected_generation_type,
        "workflow_name": primary_workflow_name(selected_generation_type, workflow_plan),
        "workflow_plan": workflow_plan,
        "count_plan": count_plan,
        "reason": str(result.get("reason") or "").strip(),
        "prompt_mode": str(result["prompt_mode"]),
        "selection_source": "llm",
        "llm_backend": result.get("llm_backend") or engine.backend_info(),
        "workflow_stage_candidates": workflow_stage_candidates,
        "generation_type_candidates": generation_type_candidates,
        "count_policies": count_policies,
        "routing_runtime_context": routing_runtime_context,
        "pre_video_review": dict(merged_routing.get("pre_video_review", {}) or {}),
        "strategy_context": _strategy_context_for_route(
            merged_routing, selected_generation_type
        ),
    }


def _build_h3_reference_runtime_context(
    repo_root: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Expose only safe reference availability facts to the routing LLM.

    Ref2VA can be selected explicitly and receive assets later from the
    planner's review stage, but automatic routing must not treat an empty or
    unreadable character-config manifest as an available reference input.
    """

    recipe = dict(generation.get("native_h3_ref2va", {}) or {})
    manifest = recipe.get("reference_manifest")
    if manifest is None:
        manifest = generation.get("reference_manifest")
    image_paths = recipe.get("reference_image_paths")
    if image_paths is None:
        image_paths = generation.get("reference_image_paths")
    video_paths = recipe.get("reference_video_paths")
    if video_paths is None:
        video_paths = generation.get("reference_video_paths")
    max_images = int(
        recipe.get("reference_max_images")
        or generation.get("reference_max_images")
        or 9
    )
    max_videos = int(
        recipe.get("reference_max_videos")
        or generation.get("reference_max_videos")
        or 3
    )

    def resolve_path(value: Any) -> str:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return str(path)

    def resolve_manifest(values: Any) -> Any:
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            return values
        resolved: list[Any] = []
        for item in values:
            if isinstance(item, dict):
                record = dict(item)
                key = "path" if record.get("path") is not None else "source_path"
                if record.get(key):
                    record[key] = resolve_path(record[key])
                resolved.append(record)
            else:
                resolved.append(resolve_path(item))
        return resolved

    resolved_manifest = resolve_manifest(manifest)
    resolved_image_paths = [resolve_path(item) for item in (image_paths or [])]
    resolved_video_paths = [resolve_path(item) for item in (video_paths or [])]
    has_configured_values = bool(
        resolved_manifest or resolved_image_paths or resolved_video_paths
    )
    base_context = {
        "source": "character_config",
        "reference_manifest_available": False,
        "reference_image_count": 0,
        "reference_video_count": 0,
        # An empty manifest is not a dead end: the automatic Ref2VA route can
        # create six T2I candidates and ask Discord to select references.
        "automatic_ref2va_eligible": not has_configured_values,
        "reference_manifest_error_code": "auto_generation_available"
        if not has_configured_values
        else "unverified",
    }
    if not has_configured_values:
        return base_context

    try:
        references = normalize_reference_manifest(
            resolved_manifest,
            image_paths=resolved_image_paths,
            video_paths=resolved_video_paths,
            require_files=True,
            max_images=max_images,
            max_videos=max_videos,
        )
    except FileNotFoundError:
        base_context["reference_manifest_error_code"] = "missing_file"
        return base_context
    except (TypeError, ValueError):
        base_context["reference_manifest_error_code"] = "invalid"
        return base_context

    image_count = sum(1 for reference in references if reference.get("type") == "image")
    video_count = sum(1 for reference in references if reference.get("type") == "video")
    return {
        **base_context,
        "reference_manifest_available": True,
        "reference_image_count": image_count,
        "reference_video_count": video_count,
        "automatic_ref2va_eligible": True,
        "reference_manifest_error_code": "",
    }


def _collect_generation_type_candidates(
    routing_config: dict[str, Any],
    preferred_generation_type: str | None,
    routing_runtime_context: dict[str, Any] | None = None,
) -> list[str]:
    explicit = (
        routing_config.get("strategy_candidates")
        or routing_config.get("allowed_generation_types")
        or []
    )
    candidates: list[str] = []
    if preferred_generation_type:
        candidate = str(preferred_generation_type).strip()
        if candidate not in CONFIG_MEDIA_TYPE_MAP:
            raise ValueError(f"Unsupported generation type: {candidate}")
        candidates.append(candidate)
    elif isinstance(explicit, list):
        candidates.extend(str(item).strip() for item in explicit if str(item).strip())
    if not preferred_generation_type and isinstance(routing_runtime_context, dict):
        if not bool(routing_runtime_context.get("automatic_ref2va_eligible", False)):
            candidates = [
                candidate for candidate in candidates if candidate != "native_h3_ref2va"
            ]
    ordered: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in ordered:
            continue
        ordered.append(candidate)
    if not ordered:
        raise ValueError("Routing config must provide at least one strategy candidate.")
    return ordered


def _strategy_context_for_route(
    routing_config: dict[str, Any],
    generation_type: str,
) -> dict[str, Any]:
    """Return the selected strategy's runtime creative context."""

    contexts = routing_config.get("strategy_contexts", {}) or {}
    if not isinstance(contexts, dict):
        return {}
    context = contexts.get(generation_type, {})
    return dict(context) if isinstance(context, dict) else {}


def _collect_workflow_stage_candidates(
    repo_root: Path,
    generation: dict[str, Any],
    strategies: dict[str, Any],
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
) -> dict[str, dict[str, list[str]]]:
    workflow_stage_candidates: dict[str, dict[str, list[str]]] = {}
    explicit_candidates = routing_config.get(
        "workflow_stage_candidates", {}
    ) or routing_config.get("workflow_candidates", {})
    if not isinstance(explicit_candidates, dict):
        explicit_candidates = {}
    generation_workflows = dict(generation.get("workflows", {}) or {})

    for generation_type in generation_type_candidates:
        explicit = explicit_candidates.get(generation_type, {})
        strategy = dict(strategies.get(generation_type, {}) or {})
        if isinstance(explicit, list):
            explicit = {"image_workflow_name": explicit}
        if not isinstance(explicit, dict):
            explicit = {}

        inferred: dict[str, list[Any]] = {}
        if generation_type == "serial_video":
            inferred = {}
        elif generation_type == "story_card":
            story_card_config = dict(generation.get("story_card", {}) or {})
            page_stage = dict(story_card_config.get("page_stage", {}) or {})
            inferred = {
                "image_workflow_name": [
                    page_stage.get("workflow_name"),
                    page_stage.get("workflow_path"),
                    generation_workflows.get("text2img"),
                ],
            }
        elif generation_type == "sticker_pack":
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
        elif generation_type == "game_sprite":
            sprite_config = dict(generation.get("game_sprite", {}) or {})
            inferred = {
                "image_workflow_name": [
                    sprite_config.get("keyframe_workflow_name"),
                    sprite_config.get("image_workflow_name"),
                ],
                "video_workflow_name": [
                    sprite_config.get("video_workflow_name"),
                    sprite_config.get("workflow_name"),
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
        elif generation_type == "native_h3_story":
            native_recipe = dict(generation.get("native_h3_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_story"),
                    generation_workflows.get("text2image2video"),
                ],
            }
        elif generation_type == "native_h3_t2v_story":
            native_recipe = dict(generation.get("native_h3_t2v_story", {}) or {})
            inferred = {
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_t2v_story"),
                    generation_workflows.get("text2video"),
                ],
            }
        elif generation_type == "native_h3_fl2va_story":
            native_recipe = dict(generation.get("native_h3_fl2va_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_fl2va_story"),
                    generation_workflows.get("text2image2video"),
                ],
                "refine_workflow_name": [
                    native_recipe.get("identity_refine_workflow_name"),
                    generation.get("identity_refine_workflow_name"),
                ],
            }
        elif generation_type == "native_h3_l2va_story":
            native_recipe = dict(generation.get("native_h3_l2va_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_l2va_story"),
                    generation_workflows.get("text2image2video"),
                ],
                "refine_workflow_name": [
                    native_recipe.get("identity_refine_workflow_name"),
                    generation.get("identity_refine_workflow_name"),
                ],
            }
        elif generation_type == "native_h3_ref2va":
            native_recipe = dict(generation.get("native_h3_ref2va", {}) or {})
            inferred = {
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_ref2va"),
                ],
            }
        elif generation_type == "text2image2native_h3_ref2va":
            native_recipe = dict(
                generation.get("text2image2native_h3_ref2va", {}) or {}
            )
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    native_recipe.get("image_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("text2image2native_h3_ref2va"),
                    generation_workflows.get("native_h3_ref2va"),
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

        for stage_key in WORKFLOW_STAGE_KEYS:
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
                workflow_stage_candidates.setdefault(generation_type, {})[stage_key] = (
                    resolved
                )
        workflow_stage_candidates.setdefault(generation_type, {})
    return workflow_stage_candidates


def _collect_count_policies(
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
    workflow_stage_candidates: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, dict[str, int]]]:
    raw_policies = routing_config.get("count_policies", {})
    if not isinstance(raw_policies, dict):
        raw_policies = {}
    policies: dict[str, dict[str, dict[str, int]]] = {}
    for generation_type in generation_type_candidates:
        strategy_policy = dict(raw_policies.get(generation_type, {}) or {})
        normalized: dict[str, dict[str, int]] = {}
        for count_key, raw in active_count_policy(
            generation_type,
            workflow_stage_candidates.get(generation_type, {}),
            strategy_policy,
        ).items():
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


def _merge_routing_config(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
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
        name = _resolve_workflow_name_from_reference(
            repo_root, stems, wf_dir, reference
        )
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


def _native_recipe_for_generation(
    generation: dict[str, Any], generation_type: str
) -> dict[str, Any]:
    if generation_type == "story_card":
        return dict(generation.get("story_card", {}) or {})
    if generation_type == "motion_comic":
        return dict(generation.get("motion_comic", {}) or {})
    if generation_type in NATIVE_STORY_GENERATION_TYPES:
        selected = dict(generation.get(generation_type, {}) or {})
        if selected:
            return selected
        return dict(generation.get("native_h3_story", {}) or {})
    if generation_type == "game_sprite":
        return dict(generation.get("game_sprite", {}) or {})
    return dict(generation.get("native_h3_story", {}) or {})


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_duration_seconds(
    config_generation_type: str,
    generation: dict[str, Any] | None = None,
    requested_duration_seconds: int | None = None,
) -> int:
    requested = (
        max(1, int(requested_duration_seconds))
        if requested_duration_seconds is not None
        else None
    )
    if config_generation_type in NATIVE_STORY_GENERATION_TYPES:
        native_recipe = _native_recipe_for_generation(
            dict(generation or {}), config_generation_type
        )
        native_duration = max(
            MIN_VIDEO_DURATION_SECONDS,
            int(native_recipe.get("duration_seconds", 15)),
        )
        if requested is not None and requested != native_duration:
            raise ValueError(
                f"{config_generation_type} supports duration_seconds={native_duration}; "
                f"received {requested}."
            )
        return native_duration
    if config_generation_type == "text2image2video":
        return max(MIN_VIDEO_DURATION_SECONDS, requested or 10)
    if config_generation_type == "game_sprite":
        sprite_config = dict((generation or {}).get("game_sprite", {}) or {})
        configured_duration = int(sprite_config.get("duration_seconds", 10))
        return max(MIN_VIDEO_DURATION_SECONDS, requested or configured_duration)
    if config_generation_type == "motion_comic":
        comic_config = dict((generation or {}).get("motion_comic", {}) or {})
        return max(
            MIN_VIDEO_DURATION_SECONDS,
            requested or int(comic_config.get("duration_seconds", 10)),
        )
    return normalize_video_duration(
        media_type_for_generation(config_generation_type), requested or 30
    )


def _duration_strategy_override(
    requested_duration_seconds: int | None,
    generation_type_candidates: list[str],
) -> str:
    """Keep requests through ten seconds on the single-action route."""
    if requested_duration_seconds is None:
        return ""
    duration = max(1, int(requested_duration_seconds))
    candidates = set(generation_type_candidates)
    if duration <= MIN_VIDEO_DURATION_SECONDS and "text2image2video" in candidates:
        return "text2image2video"
    if duration == 15 and "native_h3_story" in candidates:
        return "native_h3_story"
    return ""


def _duration_profile(duration_seconds: int | None) -> str:
    if duration_seconds is None:
        return "config_default"
    duration = max(1, int(duration_seconds))
    if duration <= MIN_VIDEO_DURATION_SECONDS:
        return "single_action"
    if duration <= 15:
        return "compact_story"
    return "extended_story"


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
        "selection_source": str(routing.get("selection_source", "")),
        "routing_runtime_context": dict(
            routing.get("routing_runtime_context", {}) or {}
        ),
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
    generation_type: str = "",
    news_driven: bool = False,
    news_context_override: dict[str, Any] | None = None,
    recorder: RunRecorder | None = None,
) -> dict[str, Any]:
    explicit_prompt = str(prompt).strip()
    news_context = dict(news_context_override or {})
    resolved_news_driven = bool(news_driven)
    if news_context:
        title = str(news_context.get("title") or "")
        keyword = str(news_context.get("keyword") or "")
        if not is_usable_news_selection(title, keyword):
            raise ValueError(
                "Explicit news context requires a usable title."
            )
        if not is_brand_safe_news_selection(title, keyword):
            raise ValueError("Explicit news context is not brand safe.")
    if explicit_prompt and not news_driven:
        return {
            "prompt": explicit_prompt,
            "source": "user",
            "prompt_mode": "user",
            "creative_seed": "",
            "news_context": news_context,
            "news_driven": False,
            "news_grounding_required": False,
            "source_anchors": [],
        }

    if not news_context:
        try:
            news_service = NewsContextService()
            selected_news = news_service.select_news()
            if selected_news is not None:
                selected_context = selected_news.to_dict()
                selected_title = str(selected_context.get("title") or "")
                selected_keyword = str(selected_context.get("keyword") or "")
                if (
                    is_usable_news_selection(
                        selected_title, selected_keyword
                    )
                    and is_brand_safe_news_selection(
                        selected_title, selected_keyword
                    )
                ):
                    news_context = selected_context
                    resolved_news_driven = True
        except Exception as exc:
            if news_driven:
                raise
            news_context = {"error": f"{type(exc).__name__}: {exc}"}

    news_grounding_required = resolved_news_driven
    if resolved_news_driven and not is_usable_news_selection(
        news_context.get("title", ""), news_context.get("keyword", "")
    ):
        raise RuntimeError(
            "News-driven generation did not receive a usable news context."
        )

    if generation_type in ({"story_card"} | NATIVE_STORY_GENERATION_TYPES):
        return {
            "prompt": "",
            "source": "news",
            "prompt_mode": "news",
            "creative_seed": "",
            "news_context": news_context,
            "news_driven": resolved_news_driven,
            "news_grounding_required": news_grounding_required,
            "source_anchors": [],
        }

    bundle = LLMPromptEngine(
        mode=os.environ.get("AGENTIC_LLM_MODE", "llm"),
        recorder=recorder,
    ).generate_autonomous_scene_prompt(
        character=character_name,
        style=style,
        media_type=media_type,
        news_context=news_context,
        news_grounding_required=news_grounding_required,
    )
    bundle.setdefault("prompt_mode", "template")
    bundle.setdefault("source", "autonomous_fallback")
    bundle["news_context"] = news_context
    bundle["news_driven"] = resolved_news_driven
    bundle["news_grounding_required"] = news_grounding_required
    bundle.setdefault("source_anchors", [])
    return bundle
