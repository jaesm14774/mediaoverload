from __future__ import annotations

from typing import Any

from agentic.runtime.model_backends import AgenticLLMManager, ModelConfig, OpenRouterModel, build_model


def build_llm_manager(backend: dict[str, Any]) -> Any:
    text_provider = str(backend["text_provider"])
    text_model = str(backend["text_model"])
    vision_provider = str(backend["vision_provider"])
    vision_model = str(backend["vision_model"])
    random_models = bool(backend["random_models"])

    if random_models and text_provider.lower() == "openrouter":
        text_model = OpenRouterModel.get_random_free_text_model()
    if random_models and vision_provider.lower() == "openrouter":
        vision_model = OpenRouterModel.get_random_free_vision_model()

    return AgenticLLMManager(
        text_model=build_model(text_provider, ModelConfig(model_name=text_model)),
        vision_model=build_model(vision_provider, ModelConfig(model_name=vision_model)),
    )
