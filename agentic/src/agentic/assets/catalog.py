from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowToolSpec:
    name: str
    tool_name: str
    workflow_name: str
    description: str
    media_type: str
    mode: str = "generate"
    defaults: dict[str, Any] = field(default_factory=dict)
    required_inputs: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowToolSpec":
        return cls(
            name=data["name"],
            tool_name=data["tool_name"],
            workflow_name=data["workflow_name"],
            description=data["description"],
            media_type=data["media_type"],
            mode=data.get("mode", "generate"),
            defaults=data.get("defaults", {}),
            required_inputs=data.get("required_inputs", []),
            output_keys=data.get("output_keys", []),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tool_name": self.tool_name,
            "workflow_name": self.workflow_name,
            "description": self.description,
            "media_type": self.media_type,
            "mode": self.mode,
            "defaults": self.defaults,
            "required_inputs": self.required_inputs,
            "output_keys": self.output_keys,
            "tags": self.tags,
        }


@dataclass(slots=True)
class LegacyCapability:
    capability: str
    current_entrypoints: list[str]
    dependencies: list[str]
    target_skills: list[str]
    target_tools: list[str]
    migration_stage: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LegacyCapability":
        return cls(
            capability=data["capability"],
            current_entrypoints=data.get("current_entrypoints", []),
            dependencies=data.get("dependencies", []),
            target_skills=data.get("target_skills", []),
            target_tools=data.get("target_tools", []),
            migration_stage=data.get("migration_stage", "planned"),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "current_entrypoints": self.current_entrypoints,
            "dependencies": self.dependencies,
            "target_skills": self.target_skills,
            "target_tools": self.target_tools,
            "migration_stage": self.migration_stage,
            "notes": self.notes,
        }


class CatalogLoader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load_workflow_tool_specs(self) -> list[WorkflowToolSpec]:
        spec_path = self.root / "configs" / "tool_specs" / "workflow_tools.json"
        if not spec_path.exists():
            return []
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return [WorkflowToolSpec.from_dict(item) for item in data]

    def load_legacy_capabilities(self) -> list[LegacyCapability]:
        migration_path = self.root / "configs" / "migrations" / "legacy_capabilities.json"
        if not migration_path.exists():
            return []
        data = json.loads(migration_path.read_text(encoding="utf-8"))
        return [LegacyCapability.from_dict(item) for item in data]
