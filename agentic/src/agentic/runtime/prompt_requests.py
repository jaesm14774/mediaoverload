from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationRoutingRequest:
    """Inputs required to choose one generation route and its bounded plan."""

    prompt: str
    character: str
    style: str
    generation_type_candidates: tuple[str, ...]
    workflow_stage_candidates: dict[str, dict[str, list[str]]]
    count_policies: dict[str, dict[str, Any]]
    routing_hints: dict[str, Any] = field(default_factory=dict)
    preferred_generation_type: str | None = None


@dataclass(frozen=True, slots=True)
class JsonChatRequest:
    """One JSON-producing request for an LLM call."""

    manager: Any
    system_prompt: str
    user_prompt: str
    schema_name: str
    schema: dict[str, Any]
    model: str = "text"
    images: list[str] | None = None
    recorder: Any = None
    max_retries: int | None = None
    request_timeout: float | None = None
    max_models_per_call: int | None = None
    repair_attempts: int = 2
    use_response_format: bool = True
