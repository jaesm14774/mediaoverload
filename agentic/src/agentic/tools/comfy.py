from __future__ import annotations

from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.runtime.registry import ToolRegistry


class BuiltinTools:
    def __init__(self, asset_registry: AssetRegistry) -> None:
        self.asset_registry = asset_registry

    def load_manifest(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        manifest = self.asset_registry.get_manifest(workflow_name)
        return {"manifest": manifest.to_dict()}

    def materialize_workflow(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        manifest = self.asset_registry.get_manifest(workflow_name)
        workflow_path = self.asset_registry.materialize_workflow(manifest)
        template = self.asset_registry.load_workflow_template(manifest)
        return {
            "workflow_name": manifest.name,
            "workflow_path": str(workflow_path),
            "template_preview": template.get("title") if template else None,
        }

    def ensure_workflow_ready(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        auto_download = bool(payload.get("auto_download", False))
        return self.asset_registry.ensure_workflow_ready(workflow_name, auto_download)

def register_builtin_tools(tool_registry: ToolRegistry, asset_registry: AssetRegistry) -> None:
    tools = BuiltinTools(asset_registry)
    tool_registry.register("workflow.load_manifest", tools.load_manifest, "Load workflow metadata (from configs/workflow JSON)")
    tool_registry.register("workflow.materialize", tools.materialize_workflow, "Materialize workflow template path")
    tool_registry.register("asset.ensure_workflow_ready", tools.ensure_workflow_ready, "Prepare workflow assets")
