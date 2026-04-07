from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic.tools.comfy_backend import AgenticMediaGenerator, AgenticNodeManager


class ComfyAdapter:
    def __init__(self) -> None:
        self.node_manager = AgenticNodeManager()

    def resolve_alias(self, workflow_path: str, alias_name: str) -> str | None:
        return self.node_manager.resolve_alias(workflow_path, alias_name)

    def generate_updates(
        self,
        workflow: dict[str, Any],
        updates_config: list[dict[str, Any]] | None = None,
        description: str | None = None,
        seed: int | None = None,
        workflow_path: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.node_manager.generate_updates(
            workflow=workflow,
            updates_config=updates_config or [],
            description=description,
            seed=seed,
            workflow_path=workflow_path,
        )

    def build_generator(self, host: str | None = None, port: int | None = None) -> Any:
        return AgenticMediaGenerator(host=host, port=port)

    @staticmethod
    def load_workflow(workflow_path: Path) -> dict[str, Any]:
        return json.loads(workflow_path.read_text(encoding="utf-8"))
