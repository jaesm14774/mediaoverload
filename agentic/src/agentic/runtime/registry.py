from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic.runtime.contracts import SkillHandler, ToolHandler


@dataclass(slots=True)
class SkillDefinition:
    name: str
    handler: SkillHandler
    description: str
    stage: str = "runtime"
    tags: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    handler: ToolHandler
    description: str
    kind: str = "generic"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(
        self,
        name: str,
        handler: SkillHandler,
        description: str,
        *,
        stage: str = "runtime",
        tags: tuple[str, ...] = (),
        tool_names: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._skills[name] = SkillDefinition(
            name=name,
            handler=handler,
            description=description,
            stage=stage,
            tags=tags,
            tool_names=tool_names,
            metadata=metadata or {},
        )

    def get(self, name: str) -> SkillDefinition:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    def list_names(self) -> list[str]:
        return sorted(self._skills)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "stage": item.stage,
                "tags": list(item.tags),
                "tool_names": list(item.tool_names),
                "metadata": item.metadata,
            }
            for item in sorted(self._skills.values(), key=lambda entry: entry.name)
        ]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        description: str,
        *,
        kind: str = "generic",
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            description=description,
            kind=kind,
            tags=tags,
            metadata=metadata or {},
        )

    def call(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(payload)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "kind": item.kind,
                "tags": list(item.tags),
                "metadata": item.metadata,
            }
            for item in sorted(self._tools.values(), key=lambda entry: entry.name)
        ]

