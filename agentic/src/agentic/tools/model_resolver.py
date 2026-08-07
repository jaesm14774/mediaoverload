from __future__ import annotations

from agentic.runtime.registry import ToolRegistry


def register_model_resolver_tools(
    tool_registry: ToolRegistry,
    *,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
) -> None:
    del comfy_host, comfy_port

    def resolve_model(payload: dict[str, object]) -> dict[str, object]:
        return {"status": "noop", "payload": dict(payload or {})}

    tool_registry.register(
        "model.resolve",
        resolve_model,
        "Resolve model metadata when a workflow requests it.",
    )
