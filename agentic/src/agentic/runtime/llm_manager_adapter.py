from __future__ import annotations

import random
from typing import Any

from agentic.runtime.model_backends import (
    AgenticLLMManager,
    FallbackChatModel,
    MAX_OPENROUTER_MODELS_PER_CALL,
    ModelConfig,
    OpenRouterModelCatalog,
    OpenRouterModel,
    OpenRouterRotatingModel,
    build_model,
    provider_credentials_present,
    provider_default_model,
    static_openrouter_model_modes,
    static_openrouter_models,
)

MAX_AUXILIARY_FALLBACKS = 3


def _build_openrouter_text_candidates(backend: dict[str, Any]) -> list[str]:
    text_provider = str(backend["text_provider"])
    if text_provider.lower() != "openrouter":
        return [str(backend["text_model"]).strip() or OpenRouterModel.FREE_TEXT_MODELS[0]]

    pool_mode = bool(backend.get("openrouter_text_pool_mode"))
    rotate = bool(backend.get("openrouter_rotate_text_models", True))
    text_raw = str(backend.get("text_model_raw", "")).strip()
    pool = _discover_pool(backend, "text")

    if pool_mode:
        shuffled = pool[:]
        if bool(backend.get("random_models", True)):
            random.shuffle(shuffled)
        if not rotate:
            return [shuffled[0]]
        backend["openrouter_text_candidates"] = shuffled
        return shuffled

    primary = text_raw or pool[0]
    if not rotate:
        return [primary]
    rest = [m for m in pool if m != primary]
    random.shuffle(rest)
    return [primary, *rest]


def _build_openrouter_vision_candidates(backend: dict[str, Any]) -> list[str]:
    vision_provider = str(backend["vision_provider"])
    if vision_provider.lower() != "openrouter":
        return [str(backend["vision_model"]).strip() or OpenRouterModel.FREE_VISION_MODELS[0]]

    pool_mode = bool(backend.get("openrouter_vision_pool_mode"))
    rotate = bool(backend.get("openrouter_rotate_vision_models", True))
    vision_raw = str(backend.get("vision_model_raw", "")).strip()
    pool = _discover_pool(backend, "vision")

    if pool_mode:
        shuffled = pool[:]
        if bool(backend.get("random_models", True)):
            random.shuffle(shuffled)
        if not rotate:
            return [shuffled[0]]
        backend["openrouter_vision_candidates"] = shuffled
        return shuffled

    primary = vision_raw or pool[0]
    if not rotate:
        return [primary]
    rest = [m for m in pool if m != primary]
    random.shuffle(rest)
    return [primary, *rest]


def _discover_pool(backend: dict[str, Any], modality: str) -> list[str]:
    explicit = backend.get(f"openrouter_{modality}_models")
    if isinstance(explicit, list) and explicit:
        pool = [str(item).strip() for item in explicit if str(item).strip()]
        if bool(backend.get(f"openrouter_{modality}_pool_mode")):
            verified = set(static_openrouter_models(modality))
            filtered = [model for model in pool if model in verified]
            if filtered:
                pool = filtered
                backend[f"openrouter_{modality}_pool_source"] = "env_static_list_filtered"
            else:
                pool = static_openrouter_models(modality)
                backend[f"openrouter_{modality}_pool_source"] = "static_config_fallback"
        else:
            backend[f"openrouter_{modality}_pool_source"] = "env_static_list"
    elif not bool(backend.get("openrouter_discover_models", False)):
        pool = static_openrouter_models(modality)
        backend[f"openrouter_{modality}_pool_source"] = "static_config"
    else:
        limit = int(backend.get("openrouter_free_pool_size") or 0)
        pool = OpenRouterModelCatalog.candidates(modality, limit=limit or None)
        backend[f"openrouter_{modality}_pool_source"] = "live_catalog_opt_in"
    if not pool:
        raise RuntimeError(f"No eligible OpenRouter free {modality} models are configured")
    backend[f"openrouter_{modality}_candidates"] = list(pool)
    return pool


def _max_models_per_call(backend: dict[str, Any], *, vision: bool) -> int | None:
    key = "openrouter_max_vision_models_per_call" if vision else "openrouter_max_text_models_per_call"
    raw = str(backend.get(key, "") or "").strip()
    if raw.isdigit():
        value = int(raw)
        return min(value, MAX_OPENROUTER_MODELS_PER_CALL) if value > 0 else None
    return None


def _csv_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _fallback_pairs(backend: dict[str, Any], modality: str) -> list[tuple[str, str]]:
    prefix = "vision" if modality == "vision" else "text"
    providers = _csv_values(backend.get(f"{prefix}_fallback_providers"))
    models = _csv_values(backend.get(f"{prefix}_fallback_models"))

    if providers and models and len(providers) != len(models):
        backend[f"{prefix}_fallback_config_error"] = (
            f"{prefix}_fallback_providers and {prefix}_fallback_models must have the same length"
        )
        return []

    pairs: list[tuple[str, str]] = []
    for index, provider in enumerate(providers):
        if index < len(models):
            model = models[index]
        else:
            try:
                model = provider_default_model(provider, modality)
            except ValueError:
                model = ""
        pair = (provider, model)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _add_auxiliary_fallbacks(
    primary: Any,
    backend: dict[str, Any],
    modality: str,
) -> Any:
    allowed = bool(backend.get("allow_vision_fallback" if modality == "vision" else "allow_text_fallback"))
    if not allowed:
        return primary

    auxiliary: list[Any] = []
    skipped: list[str] = []
    fallback_pairs = _fallback_pairs(backend, modality)[:MAX_AUXILIARY_FALLBACKS]
    for provider, model_name in fallback_pairs:
        normalized_provider = provider.lower()
        if normalized_provider == "openrouter":
            skipped.append(f"{provider}:primary_provider")
            continue
        if not model_name:
            skipped.append(f"{provider}:no_{modality}_model")
            continue
        try:
            if not provider_credentials_present(normalized_provider):
                skipped.append(f"{provider}:missing_api_key")
                continue
            auxiliary.append(build_model(normalized_provider, ModelConfig(model_name=model_name, temperature=0.3)))
        except (KeyError, ValueError) as exc:
            skipped.append(f"{provider}:{type(exc).__name__}")

    backend[f"{modality}_fallback_skipped"] = skipped
    backend[f"{modality}_fallback_candidates"] = [
        f"{provider}:{model_name}" for provider, model_name in fallback_pairs if model_name
    ]
    if not auxiliary:
        return primary
    return FallbackChatModel(primary, *auxiliary)


def _wrap_openrouter_text(
    backend: dict[str, Any],
    candidates: list[str],
) -> Any:
    max_models = _max_models_per_call(backend, vision=False)
    limited = candidates
    if max_models is not None:
        limited = candidates[:max_models]
    cfg = ModelConfig(model_name=limited[0], temperature=0.3)
    modes = static_openrouter_model_modes("text")
    if len(limited) == 1 and not bool(backend.get("openrouter_text_pool_mode")):
        primary: Any = OpenRouterModel(cfg)
    else:
        primary = OpenRouterRotatingModel(
            cfg,
            limited,
            max_models_per_call=len(limited),
            model_modes=modes,
            random_each_call=bool(backend.get("random_models", True)),
        )
    return _add_auxiliary_fallbacks(primary, backend, "text")


def _wrap_openrouter_vision(backend: dict[str, Any], candidates: list[str]) -> Any:
    max_models = _max_models_per_call(backend, vision=True)
    limited = candidates
    if max_models is not None:
        limited = candidates[:max_models]
    cfg = ModelConfig(model_name=limited[0], temperature=0.3)
    modes = static_openrouter_model_modes("vision")
    if len(limited) == 1 and not bool(backend.get("openrouter_vision_pool_mode")):
        primary: Any = OpenRouterModel(cfg)
    else:
        primary = OpenRouterRotatingModel(
            cfg,
            limited,
            max_models_per_call=len(limited),
            model_modes=modes,
            random_each_call=bool(backend.get("random_models", True)),
        )
    return _add_auxiliary_fallbacks(primary, backend, "vision")


def build_llm_manager(backend: dict[str, Any]) -> Any:
    text_provider = str(backend["text_provider"])
    vision_provider = str(backend["vision_provider"])

    text_candidates = _build_openrouter_text_candidates(backend)
    vision_candidates = _build_openrouter_vision_candidates(backend)

    if text_provider.lower() == "openrouter":
        text_model = _wrap_openrouter_text(backend, text_candidates)
    else:
        raw = str(backend.get("text_model_raw") or "").strip()
        if not raw:
            raw = str(backend.get("text_model") or "").strip()
        if not raw:
            raw = provider_default_model(text_provider, "text")
        text_model = build_model(
            text_provider,
            ModelConfig(model_name=raw),
        )
        text_model = _add_auxiliary_fallbacks(text_model, backend, "text")

    if vision_provider.lower() == "openrouter":
        vision_model = _wrap_openrouter_vision(backend, vision_candidates)
    else:
        vraw = str(backend.get("vision_model_raw") or "").strip()
        if not vraw:
            vraw = str(backend.get("vision_model") or "").strip()
        if not vraw:
            vraw = provider_default_model(vision_provider, "vision")
        if not vraw:
            raise ValueError(f"Provider '{vision_provider}' has no configured vision model")
        vision_model = build_model(
            vision_provider,
            ModelConfig(model_name=vraw),
        )
        vision_model = _add_auxiliary_fallbacks(vision_model, backend, "vision")

    return AgenticLLMManager(
        text_model=text_model,
        vision_model=vision_model,
    )
