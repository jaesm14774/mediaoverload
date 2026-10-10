"""Resolve configured LLM providers separately from creative tasks."""

from __future__ import annotations

import os
from typing import Any
from agentic.runtime.model_backends import _load_project_env, provider_default_model


def resolve_backend_info(mode: str) -> dict[str, Any]:
    _load_project_env()
    text_provider = (
        str(
            os.environ.get("AGENTIC_TEXT_MODEL_PROVIDER", "openrouter") or "openrouter"
        ).strip()
        or "openrouter"
    )
    text_model_raw = str(os.environ.get("AGENTIC_TEXT_MODEL", "") or "").strip()
    vision_provider = (
        str(
            os.environ.get("AGENTIC_VISION_MODEL_PROVIDER", text_provider)
            or text_provider
        ).strip()
        or text_provider
    )
    vision_model_raw = str(os.environ.get("AGENTIC_VISION_MODEL", "") or "").strip()
    raw_reasoning_effort = os.environ.get("AGENTIC_REASONING_EFFORT", "xhigh")
    reasoning_effort = str(raw_reasoning_effort or "xhigh").strip().lower() or "xhigh"
    random_models = os.environ.get("AGENTIC_RANDOM_MODELS", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    text_strategy = (
        os.environ.get("AGENTIC_OPENROUTER_TEXT_MODEL_STRATEGY", "").strip().lower()
    )
    openrouter_text_pool_mode = text_strategy == "free_pool" or (
        text_provider.lower() == "openrouter" and text_model_raw.strip() == ""
    )
    vision_strategy = (
        os.environ.get("AGENTIC_OPENROUTER_VISION_MODEL_STRATEGY", "").strip().lower()
    )
    openrouter_vision_pool_mode = vision_strategy == "free_pool" or (
        vision_provider.lower() == "openrouter" and vision_model_raw.strip() == ""
    )

    if openrouter_text_pool_mode:
        text_model_display = "free_pool"
    elif text_provider.lower() == "openrouter":
        text_model_display = text_model_raw.strip() or "free_pool"
    else:
        text_model_display = (
            text_model_raw.strip()
            or provider_default_model(text_provider, "text")
            or "unconfigured"
        )

    if openrouter_vision_pool_mode:
        vision_model_display = "free_pool"
    elif vision_provider.lower() == "openrouter":
        vision_model_display = vision_model_raw.strip() or "free_pool"
    else:
        # Local providers such as Ollama do not have a catalog default;
        # an explicitly configured model is the source of truth. Keep the
        # catalog fallback for providers that define one.
        vision_model_display = (
            vision_model_raw.strip()
            or provider_default_model(vision_provider, "vision")
            or "unconfigured"
        )

    rotate_text = os.environ.get(
        "AGENTIC_OPENROUTER_ROTATE_TEXT_MODELS", "true"
    ).lower() in {"1", "true", "yes"}
    rotate_vision = os.environ.get(
        "AGENTIC_OPENROUTER_ROTATE_VISION_MODELS", "true"
    ).lower() in {
        "1",
        "true",
        "yes",
    }

    max_text_s = os.environ.get(
        "AGENTIC_OPENROUTER_MAX_TEXT_MODELS_PER_CALL", ""
    ).strip()
    max_vision_s = os.environ.get(
        "AGENTIC_OPENROUTER_MAX_VISION_MODELS_PER_CALL", ""
    ).strip()
    max_text_models = int(max_text_s) if max_text_s.isdigit() else 0
    max_vision_models = int(max_vision_s) if max_vision_s.isdigit() else 0
    discover_models = os.environ.get(
        "AGENTIC_OPENROUTER_DISCOVER_MODELS", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
    }
    text_models = [
        item.strip()
        for item in os.environ.get("AGENTIC_OPENROUTER_TEXT_MODELS", "").split(",")
        if item.strip()
    ]
    vision_models = [
        item.strip()
        for item in os.environ.get("AGENTIC_OPENROUTER_VISION_MODELS", "").split(",")
        if item.strip()
    ]
    free_pool_s = os.environ.get("AGENTIC_OPENROUTER_FREE_POOL_SIZE", "5").strip()
    free_pool_size = int(free_pool_s) if free_pool_s.isdigit() else 5
    cache_ttl_s = os.environ.get(
        "AGENTIC_OPENROUTER_MODEL_CACHE_TTL_SECONDS", "21600"
    ).strip()
    cache_ttl_seconds = int(cache_ttl_s) if cache_ttl_s.isdigit() else 21600

    text_fallback_providers = [
        item.strip()
        for item in os.environ.get("AGENTIC_TEXT_FALLBACK_PROVIDERS", "").split(",")
        if item.strip()
    ]
    text_fallback_models = [
        item.strip()
        for item in os.environ.get("AGENTIC_TEXT_FALLBACK_MODELS", "").split(",")
        if item.strip()
    ]
    vision_fallback_providers = [
        item.strip()
        for item in os.environ.get("AGENTIC_VISION_FALLBACK_PROVIDERS", "").split(",")
        if item.strip()
    ]
    vision_fallback_models = [
        item.strip()
        for item in os.environ.get("AGENTIC_VISION_FALLBACK_MODELS", "").split(",")
        if item.strip()
    ]
    provider_fallback_enabled = os.environ.get(
        "AGENTIC_PROVIDER_FALLBACK_ENABLED", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
    }
    allow_text_fallback = provider_fallback_enabled or os.environ.get(
        "AGENTIC_TEXT_ALLOW_FALLBACK", "false"
    ).lower() in {
        "1",
        "true",
        "yes",
    }
    allow_vision_fallback = provider_fallback_enabled or os.environ.get(
        "AGENTIC_VISION_ALLOW_FALLBACK", "false"
    ).lower() in {"1", "true", "yes"}

    return {
        "mode": mode,
        "text_provider": text_provider,
        "text_model": text_model_display,
        "text_model_raw": text_model_raw,
        "vision_provider": vision_provider,
        "vision_model": vision_model_display,
        "vision_model_raw": vision_model_raw,
        "reasoning_effort": reasoning_effort,
        "random_models": random_models,
        "openrouter_text_pool_mode": openrouter_text_pool_mode,
        "openrouter_vision_pool_mode": openrouter_vision_pool_mode,
        "openrouter_rotate_text_models": rotate_text,
        "openrouter_rotate_vision_models": rotate_vision,
        "openrouter_max_text_models_per_call": max_text_models,
        "openrouter_max_vision_models_per_call": max_vision_models,
        "openrouter_discover_models": discover_models,
        "openrouter_text_models": text_models,
        "openrouter_vision_models": vision_models,
        "openrouter_free_pool_size": max(0, free_pool_size),
        "openrouter_model_cache_ttl_seconds": max(0, cache_ttl_seconds),
        "text_fallback_providers": text_fallback_providers,
        "text_fallback_models": text_fallback_models,
        "vision_fallback_providers": vision_fallback_providers,
        "vision_fallback_models": vision_fallback_models,
        "provider_fallback_enabled": provider_fallback_enabled,
        "allow_text_fallback": allow_text_fallback,
        "allow_vision_fallback": allow_vision_fallback,
    }
