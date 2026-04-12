from __future__ import annotations

import random
from typing import Any

from agentic.runtime.model_backends import (
    AgenticLLMManager,
    FallbackChatModel,
    ModelConfig,
    OpenRouterModel,
    OpenRouterRotatingModel,
    build_model,
)


def _build_openrouter_text_candidates(backend: dict[str, Any]) -> list[str]:
    pool = list(OpenRouterModel.FREE_TEXT_MODELS)
    text_provider = str(backend["text_provider"])
    if text_provider.lower() != "openrouter":
        return [str(backend["text_model"]).strip() or pool[0]]

    pool_mode = bool(backend.get("openrouter_text_pool_mode"))
    random_models = bool(backend.get("random_models"))
    rotate = bool(backend.get("openrouter_rotate_text_models", True))
    text_raw = str(backend.get("text_model_raw", "")).strip()

    if pool_mode:
        shuffled = pool[:]
        random.shuffle(shuffled)
        if not rotate:
            return [shuffled[0]]
        return shuffled

    primary = text_raw or "qwen/qwen3.6-plus:free"
    if random_models:
        primary = OpenRouterModel.get_random_free_text_model()
    if not rotate:
        return [primary]
    rest = [m for m in pool if m != primary]
    random.shuffle(rest)
    return [primary, *rest]


def _build_openrouter_vision_candidates(backend: dict[str, Any]) -> list[str]:
    pool = list(OpenRouterModel.FREE_VISION_MODELS)
    vision_provider = str(backend["vision_provider"])
    if vision_provider.lower() != "openrouter":
        return [str(backend["vision_model"]).strip() or pool[0]]

    pool_mode = bool(backend.get("openrouter_vision_pool_mode"))
    random_models = bool(backend.get("random_models"))
    rotate = bool(backend.get("openrouter_rotate_vision_models", True))
    vision_raw = str(backend.get("vision_model_raw", "")).strip()

    if pool_mode:
        shuffled = pool[:]
        random.shuffle(shuffled)
        if not rotate:
            return [shuffled[0]]
        return shuffled

    primary = vision_raw or "qwen/qwen3.6-plus:free"
    if random_models:
        primary = OpenRouterModel.get_random_free_vision_model()
    if not rotate:
        return [primary]
    rest = [m for m in pool if m != primary]
    random.shuffle(rest)
    return [primary, *rest]


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
    if len(limited) == 1:
        primary: Any = OpenRouterModel(cfg)
    else:
        primary = OpenRouterRotatingModel(cfg, limited, max_models_per_call=len(limited))

    fb_provider = str(backend.get("text_fallback_provider", "")).strip().lower()
    fb_model = str(backend.get("text_fallback_model", "")).strip()
    if fb_provider == "gemini" and fb_model:
        fallback = build_model("gemini", ModelConfig(model_name=fb_model, temperature=0.3))
        return FallbackChatModel(primary, fallback)
    return primary


def _wrap_openrouter_vision(backend: dict[str, Any], candidates: list[str]) -> Any:
    max_models = _max_models_per_call(backend, vision=True)
    limited = candidates
    if max_models is not None:
        limited = candidates[:max_models]
    cfg = ModelConfig(model_name=limited[0], temperature=0.3)
    if len(limited) == 1:
        return OpenRouterModel(cfg)
    return OpenRouterRotatingModel(cfg, limited, max_models_per_call=len(limited))


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
            ModelConfig(model_name=raw or "qwen/qwen3.6-plus:free"),
        )

    if vision_provider.lower() == "openrouter":
        vision_model = _wrap_openrouter_vision(backend, vision_candidates)
    else:
        vraw = str(backend.get("vision_model_raw") or "").strip()
        if not vraw:
            vraw = str(backend.get("vision_model") or "").strip()
        vision_model = build_model(
            vision_provider,
            ModelConfig(model_name=vraw or "qwen/qwen3.6-plus:free"),
        )

    return AgenticLLMManager(
        text_model=text_model,
        vision_model=vision_model,
    )
