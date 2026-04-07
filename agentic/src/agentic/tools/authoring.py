from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from agentic.assets.registry import AssetRegistry
from agentic.runtime.registry import ToolRegistry


class AgentAuthoringTools:
    def __init__(self, asset_registry: AssetRegistry, root: Path) -> None:
        self.asset_registry = asset_registry
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def plan_asset_acquisition(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        return self.asset_registry.plan_asset_acquisition(workflow_name)

    def acquire_missing_assets(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        return self.asset_registry.acquire_missing_assets(workflow_name)

    def create_workflow_draft(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        variant_name = str(payload.get("variant_name") or f"{workflow_name}_draft")
        manifest = self.asset_registry.get_manifest(workflow_name)
        template = self.asset_registry.load_workflow_template(manifest)
        draft_dir = self.root / "workflow_drafts"
        draft_dir.mkdir(parents=True, exist_ok=True)
        draft_path = draft_dir / f"{variant_name}.json"
        draft_payload = {
            "base_workflow": workflow_name,
            "variant_name": variant_name,
            "summary": str(payload.get("summary") or f"Draft derived from {workflow_name}"),
            "workflow": deepcopy(template),
            "edits": list(payload.get("edits", [])),
        }
        draft_path.write_text(json.dumps(draft_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "workflow_name": workflow_name,
            "variant_name": variant_name,
            "draft_path": str(draft_path),
        }

    def recommend_workflows(self, payload: dict[str, object]) -> dict[str, object]:
        media_type = str(payload["media_type"])
        limit = int(payload.get("limit", 3))
        recommendations = self.asset_registry.recommend_workflows(
            media_type,
            style=str(payload.get("style", "")),
            prompt=str(payload.get("prompt", "")),
            limit=limit,
            require_ready=bool(payload.get("require_ready", False)),
        )
        return {
            "media_type": media_type,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
        }

    def validate_workflow(self, payload: dict[str, object]) -> dict[str, object]:
        workflow_name = str(payload["workflow_name"])
        return self.asset_registry.validate_workflow(workflow_name)

    def patch_workflow_draft(self, payload: dict[str, object]) -> dict[str, object]:
        draft_path = Path(str(payload["draft_path"]))
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        workflow = draft.get("workflow", {})
        patches = list(payload.get("patches", []))
        for patch in patches:
            self._apply_patch(workflow, dict(patch))
        draft["workflow"] = workflow
        draft.setdefault("edits", []).extend(patches)
        draft_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "draft_path": str(draft_path),
            "patch_count": len(patches),
        }

    @staticmethod
    def _apply_patch(workflow: dict[str, Any], patch: dict[str, Any]) -> None:
        node_id = str(patch["node_id"])
        input_key = str(patch["input_key"])
        value = patch.get("value")
        node = workflow.setdefault(node_id, {})
        inputs = node.setdefault("inputs", {})
        inputs[input_key] = value


def register_authoring_tools(tool_registry: ToolRegistry, asset_registry: AssetRegistry, root: Path) -> None:
    tools = AgentAuthoringTools(asset_registry=asset_registry, root=root)
    tool_registry.register("asset.plan_acquisition", tools.plan_asset_acquisition, "Describe missing workflow assets")
    tool_registry.register("asset.acquire_missing", tools.acquire_missing_assets, "Prepare missing workflow asset targets")
    tool_registry.register("workflow.recommend", tools.recommend_workflows, "Recommend workflow manifests for a media goal")
    tool_registry.register("workflow.validate_manifest", tools.validate_workflow, "Validate a workflow manifest and template")
    tool_registry.register("workflow.author.create_draft", tools.create_workflow_draft, "Create a mutable workflow draft from a manifest")
    tool_registry.register("workflow.author.patch_draft", tools.patch_workflow_draft, "Patch an authored workflow draft")
