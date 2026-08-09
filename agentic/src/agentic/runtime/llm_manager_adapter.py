from __future__ import annotations

import random
from typing import Any

from agentic.runtime.model_backends import (
    AgenticLLMManager,
    FallbackChatModel,
    ModelConfig,
    OpenRouterModelCatalog,
    OpenRouterModel,
    OpenRouterRotatingModel,
    build_model,
    static_openrouter_model_modes,
    static_openrouter_models,
)


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
        return value if value > 0 else None
    return None


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

    fb_provider = str(backend.get("text_fallback_provider", "")).strip().lower()
    fb_model = str(backend.get("text_fallback_model", "")).strip()
    if bool(backend.get("allow_text_fallback", False)) and fb_provider == "gemini" and fb_model:
        fallback = build_model("gemini", ModelConfig(model_name=fb_model, temperature=0.3))
        return FallbackChatModel(primary, fallback)
    return primary


def _wrap_openrouter_vision(backend: dict[str, Any], candidates: list[str]) -> Any:
    max_models = _max_models_per_call(backend, vision=True)
    limited = candidates
    if max_models is not None:
        limited = candidates[:max_models]
    cfg = ModelConfig(model_name=limited[0], temperature=0.3)
    modes = static_openrouter_model_modes("vision")
    if len(limited) == 1 and not bool(backend.get("openrouter_vision_pool_mode")):
        return OpenRouterModel(cfg)
    return OpenRouterRotatingModel(
        cfg,
        limited,
        max_models_per_call=len(limited),
        model_modes=modes,
        random_each_call=bool(backend.get("random_models", True)),
    )


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
        text_model = build_model(
            text_provider,
            ModelConfig(model_name=raw or OpenRouterModel.FREE_TEXT_MODELS[0]),
        )

    if vision_provider.lower() == "openrouter":
        vision_model = _wrap_openrouter_vision(backend, vision_candidates)
    else:
        vraw = str(backend.get("vision_model_raw") or "").strip()
        if not vraw:
            vraw = str(backend.get("vision_model") or "").strip()
        vision_model = build_model(
            vision_provider,
            ModelConfig(model_name=vraw or OpenRouterModel.FREE_VISION_MODELS[0]),
        )

    return AgenticLLMManager(
        text_model=text_model,
        vision_model=vision_model,
    )
