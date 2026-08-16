from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic.assets.minimax_h3 import profile_manifest


def _load_workflow_metadata(workflow_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional workflow contracts without making YAML a hard dependency."""

    config_path = workflow_dir / "workflow_config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw = payload.get("workflows", {}) if isinstance(payload, dict) else {}
        return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}
    except Exception:
        return {}


def _metadata_for_workflow(metadata: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return dict(metadata.get(name) or metadata.get(f"{name}.json") or {})


@dataclass(slots=True)
class AssetRequirement:
    name: str
    kind: str
    target_dir: str
    source: str | None = None
    alternate_target_dirs: tuple[str, ...] = ()
    expected_size: int | None = None
    sha256: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetRequirement":
        raw_alt = data.get("alternate_target_dirs") or []
        if isinstance(raw_alt, list):
            alternate = tuple(item.strip() for item in map(str, raw_alt) if item.strip())
        else:
            alternate = ()
        return cls(
            name=data["name"],
            kind=data["kind"],
            target_dir=data["target_dir"],
            source=data.get("source"),
            alternate_target_dirs=alternate,
            expected_size=int(data["expected_size"]) if data.get("expected_size") not in {None, ""} else None,
            sha256=str(data["sha256"]) if data.get("sha256") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "target_dir": self.target_dir,
            "source": self.source,
        }
        if self.alternate_target_dirs:
            payload["alternate_target_dirs"] = list(self.alternate_target_dirs)
        if self.expected_size is not None:
            payload["expected_size"] = self.expected_size
        if self.sha256:
            payload["sha256"] = self.sha256
        return payload


@dataclass(slots=True)
class WorkflowManifest:
    name: str
    media_types: list[str]
    workflow_path: str
    summary: str
    required_assets: list[AssetRequirement] = field(default_factory=list)
    recommended_defaults: dict[str, Any] = field(default_factory=dict)
    asset_extra_roots: list[str] = field(default_factory=list)
    conditioning: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "media_types": self.media_types,
            "workflow_path": self.workflow_path,
            "summary": self.summary,
            "required_assets": [asset.to_dict() for asset in self.required_assets],
            "recommended_defaults": self.recommended_defaults,
            "conditioning": self.conditioning,
        }
        if self.asset_extra_roots:
            payload["asset_extra_roots"] = list(self.asset_extra_roots)
        return payload


_WORKFLOW_RECOMMENDED_DEFAULTS: dict[str, dict[str, Any]] = {
    "anima_anime": {"width": 1024, "height": 1024, "steps": 25, "cfg": 3.5},
    "z_image_i2i_anime": {"denoise": 0.7},
    "minimax_h3_lowvram_i2v": {"width": 608, "height": 352, "length": 124, "frame_rate": 24, "steps": 20},
    "minimax_h3_lowvram_15s_fl2va_i2v": {"width": 608, "height": 352, "length": 362, "frame_rate": 24, "steps": 16},
    "minimax_h3_lowvram_t2v": {"width": 608, "height": 352, "length": 124, "frame_rate": 24, "steps": 20},
    "minimax_h3_native_t2v": {"width": 608, "height": 352, "length": 124, "frame_rate": 24, "steps": 20},
    "minimax_h3_ref2va": {
        "width": 608,
        "height": 352,
        "length": 124,
        "frame_rate": 24,
        "steps": 20,
        "ref_image_size": "match",
    },
}


def _workflow_json_dir(project_root: Path) -> Path:
    """Prefer repo-level configs/workflow when project_root is the agentic package dir."""
    parent_wf = project_root.parent / "configs" / "workflow"
    nested_wf = project_root / "configs" / "workflow"
    if parent_wf.is_dir() and any(parent_wf.glob("*.json")):
        return parent_wf
    if nested_wf.is_dir():
        return nested_wf
    return parent_wf


def _relative_to_project(project_root: Path, absolute_file: Path) -> str:
    root = project_root.resolve()
    wf = absolute_file.resolve()
    try:
        return str(wf.relative_to(root))
    except ValueError:
        return str(Path(os.path.relpath(str(wf), str(root))))


def _infer_media_types(stem: str) -> list[str]:
    lower = stem.lower()
    if lower == "minimax_h3_ref2va":
        return ["native_h3_ref2va", "long_video"]
    shared_image = [
        "image",
        "storyboard",
        "animated_sticker",
        "carousel",
        "sticker_pack",
        "text2img2img",
    ]
    if "upscal" in lower or "tile" in lower:
        return ["image_upscale"]
    if "i2i" in lower or lower == "image_to_image":
        return ["image_refine"]
    if "i2v" in lower:
        return ["image_to_video", "image_to_video_audio", "long_video"]
    return shared_image + ["text2video", "text2img2video", "video_narrate"]


def _synthetic_manifest(
    name: str,
    workflow_rel_path: str,
    metadata: dict[str, Any] | None = None,
) -> WorkflowManifest:
    values = dict(metadata or {})
    return WorkflowManifest(
        name=name,
        media_types=_infer_media_types(name),
        workflow_path=workflow_rel_path,
        summary="",
        required_assets=[],
        recommended_defaults=dict(_WORKFLOW_RECOMMENDED_DEFAULTS.get(name, {})),
        asset_extra_roots=[],
        conditioning=dict(values.get("conditioning") or {}),
    )


def _minimax_h3_manifest(
    name: str,
    workflow_rel_path: str,
    metadata: dict[str, Any] | None = None,
) -> WorkflowManifest:
    values = dict(metadata or {})
    profile_name = {
        "minimax_h3_lowvram_i2v": "balanced-lowvram",
        "minimax_h3_lowvram_t2v": "balanced-lowvram",
        "minimax_h3_lowvram_15s_fl2va_i2v": "balanced-lowvram",
        "minimax_h3_native_t2v": "native-quality",
        "minimax_h3_ref2va": "ref2va-lowvram",
    }.get(name)
    if not profile_name:
        return _synthetic_manifest(name, workflow_rel_path, values)
    payload = profile_manifest(profile_name)
    recommended_defaults = dict(payload["recommended_defaults"])
    recommended_defaults.update(_WORKFLOW_RECOMMENDED_DEFAULTS.get(name, {}))
    return WorkflowManifest(
        name=name,
        media_types=list(payload["media_types"]),
        workflow_path=workflow_rel_path,
        summary=str(payload["summary"]),
        required_assets=[AssetRequirement.from_dict(item) for item in payload["required_assets"]],
        recommended_defaults=recommended_defaults,
        asset_extra_roots=[],
        conditioning=dict(values.get("conditioning") or {}),
    )


class AssetRegistry:
    def __init__(self, project_root: Path, asset_root: Path | None = None) -> None:
        self.project_root = project_root
        self.asset_root = asset_root or project_root.parent
        self._workflow_dir = _workflow_json_dir(project_root)
        self._manifests = self._discover_workflows()

    def _discover_workflows(self) -> dict[str, WorkflowManifest]:
        manifests: dict[str, WorkflowManifest] = {}
        if not self._workflow_dir.is_dir():
            return manifests
        metadata = _load_workflow_metadata(self._workflow_dir)
        for path in sorted(self._workflow_dir.glob("*.json")):
            stem = path.stem
            rel = _relative_to_project(self.project_root, path)
            manifests[stem] = _minimax_h3_manifest(stem, rel, _metadata_for_workflow(metadata, stem))
        return manifests

    def all_manifests(self) -> list[WorkflowManifest]:
        return list(self._manifests.values())

    def refresh(self) -> None:
        self._workflow_dir = _workflow_json_dir(self.project_root)
        self._manifests = self._discover_workflows()

    def get_manifest(self, name: str) -> WorkflowManifest:
        if name not in self._manifests:
            raise KeyError(
                f"Unknown workflow: {name!r}. Expected a JSON file in {self._workflow_dir} (stem matches name)."
            )
        return self._manifests[name]

    def pick_workflow(self, media_type: str) -> WorkflowManifest:
        recommendations = self.recommend_workflows(media_type, limit=1)
        if recommendations:
            return self.get_manifest(str(recommendations[0]["workflow_name"]))
        raise LookupError(f"No workflow JSON supports media type: {media_type} (under {self._workflow_dir})")

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
            candidate_paths = self._candidate_asset_paths(manifest, requirement)
            existing_path = next((path for path in candidate_paths if path.exists()), None)
            target_path = existing_path or candidate_paths[0]
            exists = existing_path is not None
            size_mismatch = bool(
                exists
                and requirement.expected_size is not None
                and existing_path is not None
                and existing_path.stat().st_size != requirement.expected_size
            )
            if exists and size_mismatch:
                status = "corrupt"
                action = "redownload"
            elif exists:
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
                    "expected_size": requirement.expected_size,
                    "sha256": requirement.sha256,
                }
            )
        return statuses

    @staticmethod
    def _dedupe_search_bases(bases: list[Path]) -> list[Path]:
        seen: set[str] = set()
        ordered: list[Path] = []
        for base in bases:
            expanded = base.expanduser()
            key = os.path.normcase(os.path.abspath(str(expanded)))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(expanded)
        return ordered

    @staticmethod
    def _expand_candidate_dirs(base: Path, target_dir: Path) -> list[Path]:
        candidate_dirs: list[Path] = [base / target_dir]
        if target_dir.parts and target_dir.parts[0].lower() == "comfyui":
            trimmed_dir = Path(*target_dir.parts[1:]) if len(target_dir.parts) > 1 else Path()
            candidate_dirs.append(base / trimmed_dir)
        if base.name.lower() != "comfyui":
            candidate_dirs.append(base / "ComfyUI" / target_dir)
            if target_dir.parts and target_dir.parts[0].lower() == "comfyui":
                trimmed_dir = Path(*target_dir.parts[1:]) if len(target_dir.parts) > 1 else Path()
                candidate_dirs.append(base / "ComfyUI" / trimmed_dir)
        return candidate_dirs

    def _candidate_asset_paths(self, manifest: WorkflowManifest, requirement: AssetRequirement) -> list[Path]:
        target_dirs = (Path(requirement.target_dir), *[Path(item) for item in requirement.alternate_target_dirs])
        extra_roots = [Path(item) for item in manifest.asset_extra_roots]
        bases = self._dedupe_search_bases([self.asset_root, *extra_roots])
        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for base in bases:
            for target_dir in target_dirs:
                for dir_path in self._expand_candidate_dirs(base, target_dir):
                    key = str(dir_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_dirs.append(dir_path)
        return [path / requirement.name for path in unique_dirs]

    def materialize_workflow(self, manifest: WorkflowManifest) -> Path:
        return (self.project_root / manifest.workflow_path).resolve()

    def load_workflow_template(self, manifest: WorkflowManifest) -> dict[str, Any]:
        path = self.materialize_workflow(manifest)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = "".join(character.lower() if character.isalnum() else " " for character in text)
        return {part for part in normalized.split() if len(part) >= 3}
