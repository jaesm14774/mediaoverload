from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AssetRequirement:
    name: str
    kind: str
    target_dir: str
    source: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetRequirement":
        return cls(
            name=data["name"],
            kind=data["kind"],
            target_dir=data["target_dir"],
            source=data.get("source"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "target_dir": self.target_dir,
            "source": self.source,
        }


@dataclass(slots=True)
class WorkflowManifest:
    name: str
    media_types: list[str]
    workflow_path: str
    summary: str
    required_assets: list[AssetRequirement] = field(default_factory=list)
    recommended_defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowManifest":
        assets = [AssetRequirement.from_dict(item) for item in data.get("required_assets", [])]
        return cls(
            name=data["name"],
            media_types=data.get("media_types", []),
            workflow_path=data["workflow_path"],
            summary=data.get("summary", ""),
            required_assets=assets,
            recommended_defaults=data.get("recommended_defaults", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "media_types": self.media_types,
            "workflow_path": self.workflow_path,
            "summary": self.summary,
            "required_assets": [asset.to_dict() for asset in self.required_assets],
            "recommended_defaults": self.recommended_defaults,
        }


class AssetRegistry:
    def __init__(self, manifest_dir: Path, project_root: Path, asset_root: Path | None = None) -> None:
        self.manifest_dir = manifest_dir
        self.project_root = project_root
        self.asset_root = asset_root or project_root
        self._manifests = self._load_manifests()

    def _load_manifests(self) -> dict[str, WorkflowManifest]:
        manifests: dict[str, WorkflowManifest] = {}
        if not self.manifest_dir.exists():
            return manifests
        for path in sorted(self.manifest_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            manifest = WorkflowManifest.from_dict(data)
            manifests[manifest.name] = manifest
        return manifests

    def all_manifests(self) -> list[WorkflowManifest]:
        return list(self._manifests.values())

    def refresh(self) -> None:
        self._manifests = self._load_manifests()

    def get_manifest(self, name: str) -> WorkflowManifest:
        if name not in self._manifests:
            raise KeyError(f"Unknown workflow manifest: {name}")
        return self._manifests[name]

    def pick_workflow(self, media_type: str) -> WorkflowManifest:
        recommendations = self.recommend_workflows(media_type, limit=1)
        if recommendations:
            return self.get_manifest(str(recommendations[0]["workflow_name"]))
        raise LookupError(f"No workflow manifest supports media type: {media_type}")

    def recommend_workflows(
        self,
        media_type: str,
        style: str = "",
        prompt: str = "",
        *,
        limit: int = 3,
        require_ready: bool = False,
    ) -> list[dict[str, Any]]:
        style_terms = self._tokenize(f"{style} {prompt}")
        scored: list[dict[str, Any]] = []
        for manifest in self._manifests.values():
            if media_type not in manifest.media_types:
                continue
            requirement_status = self.ensure_requirements(manifest, auto_download=False)
            ready_count = sum(1 for item in requirement_status if item["status"] == "ready")
            missing_count = len(requirement_status) - ready_count
            if require_ready and missing_count:
                continue
            manifest_terms = self._tokenize(f"{manifest.name} {manifest.summary}")
            style_matches = sorted(style_terms.intersection(manifest_terms))
            score = (ready_count * 10) - (missing_count * 5) + (len(style_matches) * 2)
            rationale_parts = [
                f"ready_assets={ready_count}/{len(requirement_status)}",
                f"missing_assets={missing_count}",
            ]
            if style_matches:
                rationale_parts.append(f"style_matches={', '.join(style_matches)}")
            scored.append(
                {
                    "workflow_name": manifest.name,
                    "media_types": list(manifest.media_types),
                    "summary": manifest.summary,
                    "score": score,
                    "style_matches": style_matches,
                    "missing_assets": missing_count,
                    "ready_assets": ready_count,
                    "required_assets": [asset.to_dict() for asset in manifest.required_assets],
                    "asset_status": requirement_status,
                    "rationale": "; ".join(rationale_parts),
                }
            )
        ranked = sorted(scored, key=lambda item: (-int(item["score"]), int(item["missing_assets"]), str(item["workflow_name"])))
        return ranked[:limit]

    def validate_workflow(self, workflow_name: str) -> dict[str, Any]:
        manifest = self.get_manifest(workflow_name)
        workflow_path = self.materialize_workflow(manifest)
        issues: list[str] = []
        warnings: list[str] = []
        aliases: list[str] = []
        output_nodes: list[str] = []
        unresolved_inputs: list[dict[str, str]] = []

        if not workflow_path.exists():
            return {
                "workflow_name": workflow_name,
                "workflow_path": str(workflow_path),
                "valid": False,
                "issues": [f"Workflow file does not exist: {workflow_path}"],
                "warnings": [],
                "aliases": [],
                "output_nodes": [],
                "unresolved_inputs": [],
            }

        workflow = self.load_workflow_template(manifest)
        if not isinstance(workflow, dict) or not workflow:
            issues.append("Workflow JSON is empty or not a node mapping.")
        else:
            node_ids = set(workflow.keys())
            for node_id, node in workflow.items():
                if not isinstance(node, dict):
                    issues.append(f"Node '{node_id}' is not an object.")
                    continue
                inputs = node.get("inputs", {})
                if not isinstance(inputs, dict):
                    issues.append(f"Node '{node_id}' inputs are not a mapping.")
                    continue
                meta = node.get("_meta", {})
                if isinstance(meta, dict):
                    alias = meta.get("alias")
                    if isinstance(alias, str) and alias:
                        aliases.append(alias)
                class_type = str(node.get("class_type", ""))
                if class_type.startswith("Save") or "Preview" in class_type or "Video" in class_type:
                    output_nodes.append(node_id)
                for input_key, value in inputs.items():
                    if isinstance(value, list) and value:
                        dependency_id = str(value[0])
                        if dependency_id not in node_ids:
                            unresolved_inputs.append({"node_id": str(node_id), "input_key": str(input_key), "missing_node": dependency_id})
            if unresolved_inputs:
                issues.extend(
                    f"Node '{item['node_id']}' input '{item['input_key']}' references missing node '{item['missing_node']}'."
                    for item in unresolved_inputs
                )
            if not output_nodes:
                warnings.append("No obvious output nodes detected; expected Save*/Preview*/Video* class types.")

        return {
            "workflow_name": workflow_name,
            "workflow_path": str(workflow_path),
            "valid": not issues,
            "issues": issues,
            "warnings": warnings,
            "aliases": sorted(set(aliases)),
            "output_nodes": sorted(set(output_nodes)),
            "unresolved_inputs": unresolved_inputs,
        }

    def ensure_workflow_ready(self, workflow_name: str, auto_download: bool) -> dict[str, Any]:
        manifest = self.get_manifest(workflow_name)
        asset_status = self.ensure_requirements(manifest, auto_download)
        workflow_path = self.materialize_workflow(manifest)
        return {
            "workflow_name": manifest.name,
            "workflow_path": str(workflow_path),
            "asset_status": asset_status,
            "auto_download": auto_download,
        }

    def plan_asset_acquisition(self, workflow_name: str) -> dict[str, Any]:
        manifest = self.get_manifest(workflow_name)
        statuses = self.ensure_requirements(manifest, auto_download=False)
        missing = [item for item in statuses if item["status"] != "ready"]
        return {
            "workflow_name": workflow_name,
            "missing_assets": missing,
            "missing_count": len(missing),
        }

    def acquire_missing_assets(self, workflow_name: str) -> dict[str, Any]:
        manifest = self.get_manifest(workflow_name)
        acquisition_plan = self.plan_asset_acquisition(workflow_name)
        prepared_assets: list[dict[str, Any]] = []
        for item in acquisition_plan["missing_assets"]:
            target_path = Path(str(item["target_path"]))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            prepared_assets.append(
                {
                    "asset": item["asset"],
                    "kind": item["kind"],
                    "target_path": str(target_path),
                    "action": "prepare_download_slot",
                    "source": item.get("source"),
                }
            )
        return {
            "workflow_name": manifest.name,
            "prepared_assets": prepared_assets,
            "prepared_count": len(prepared_assets),
        }

    def ensure_requirements(self, manifest: WorkflowManifest, auto_download: bool) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        for requirement in manifest.required_assets:
            candidate_paths = self._candidate_asset_paths(requirement)
            existing_path = next((path for path in candidate_paths if path.exists()), None)
            target_path = existing_path or candidate_paths[0]
            exists = existing_path is not None
            if exists:
                status = "ready"
                action = "reuse"
            elif auto_download:
                status = "scheduled"
                action = "download"
            else:
                status = "missing"
                action = "manual_setup"
            statuses.append(
                {
                    "asset": requirement.name,
                    "kind": requirement.kind,
                    "target_path": str(target_path),
                    "candidate_paths": [str(path) for path in candidate_paths],
                    "status": status,
                    "action": action,
                    "source": requirement.source,
                }
            )
        return statuses

    def _candidate_asset_paths(self, requirement: AssetRequirement) -> list[Path]:
        target_dir = Path(requirement.target_dir)
        candidate_dirs: list[Path] = [self.asset_root / target_dir]

        if target_dir.parts and target_dir.parts[0].lower() == "comfyui":
            trimmed_dir = Path(*target_dir.parts[1:]) if len(target_dir.parts) > 1 else Path()
            candidate_dirs.append(self.asset_root / trimmed_dir)

        if self.asset_root.name.lower() != "comfyui":
            candidate_dirs.append(self.asset_root / "ComfyUI" / target_dir)
            if target_dir.parts and target_dir.parts[0].lower() == "comfyui":
                trimmed_dir = Path(*target_dir.parts[1:]) if len(target_dir.parts) > 1 else Path()
                candidate_dirs.append(self.asset_root / "ComfyUI" / trimmed_dir)

        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for path in candidate_dirs:
            resolved_key = str(path)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            unique_dirs.append(path)

        return [path / requirement.name for path in unique_dirs]

    def materialize_workflow(self, manifest: WorkflowManifest) -> Path:
        workflow_path = self.project_root / manifest.workflow_path
        return workflow_path

    def load_workflow_template(self, manifest: WorkflowManifest) -> dict[str, Any]:
        path = self.materialize_workflow(manifest)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
        return {part for part in normalized.split() if len(part) >= 3}

