from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import SkillContext, SkillResult


def safe_path_component(value: object, *, field_name: str = "path component") -> str:
    """Validate a caller-provided single path component before joining it."""
    component = str(value)
    if component in {".", ".."} or any(
        character in component for character in ("\\", "/", ":")
    ) or any(ord(character) < 32 for character in component):
        raise ValueError(f"{field_name} must be a single safe path component")
    return component


def slug_path_component(
    value: object,
    *,
    default: str = "item",
    max_length: int = 80,
) -> str:
    """Turn free-form model text into a portable single path component."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("- .")
    slug = slug[:max_length].rstrip("- .")
    return slug or default


def asset_check_result(result: dict[str, object], success_log: str) -> SkillResult:
    """Convert the common asset-readiness response into a skill result."""
    asset_status = result.get("asset_status", [])
    if not isinstance(asset_status, list):
        return SkillResult(status="success", outputs=result, logs=[success_log])

    missing_assets = [
        str(item.get("asset", "")).strip()
        for item in asset_status
        if isinstance(item, dict) and str(item.get("status", "")).lower() != "ready"
    ]
    if not missing_assets:
        return SkillResult(status="success", outputs=result, logs=[success_log])

    workflow_name = str(result.get("workflow_name", "")).strip()
    details = ", ".join(asset for asset in missing_assets if asset) or "unknown assets"
    return SkillResult(
        status="failed",
        outputs=result,
        logs=[f"Workflow assets missing for '{workflow_name}': {details}"],
    )


def build_run_dir(
    output_root: Path,
    prompt: str,
    suffix: str = "",
    *,
    default_slug: str = "run",
    max_slug_length: int = 32,
    suffix_first: bool = False,
) -> Path:
    """Build the canonical timestamped artifact directory used by skills."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(prompt).lower()).strip("-")
    slug = slug[:max_slug_length] or default_slug
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_suffix = safe_path_component(suffix, field_name="run suffix")
    parts = (timestamp, safe_suffix, slug) if suffix_first and safe_suffix else (timestamp, slug, safe_suffix)
    return output_root / "_".join(part for part in parts if part)


def resolve_dependency_value(
    context: SkillContext,
    candidate_keys: tuple[str, ...],
    *,
    required: bool = False,
) -> str | None:
    """Resolve the newest non-empty scalar/list value from node dependencies."""
    for dependency in reversed(context.node.depends_on):
        dependency_output = context.state[dependency]
        if not isinstance(dependency_output, dict):
            continue
        for key in candidate_keys:
            value = dependency_output.get(key)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str) and value:
                return value
    if required:
        raise RuntimeError(f"No dependency output found for node '{context.node.node_id}'")
    return None


def resolve_dependency_text(
    context: SkillContext,
    *,
    input_key: str,
    candidate_keys: tuple[str, ...],
    default: str = "",
) -> str:
    """Resolve an explicit input first, then fall back to dependency text."""
    explicit = context.node.inputs.get(input_key)
    if isinstance(explicit, str) and explicit:
        return explicit
    resolved = resolve_dependency_value(context, candidate_keys)
    return resolved if resolved is not None else default


def resolve_dependency_prompt(context: SkillContext, *, default: str | None = None) -> str:
    return resolve_dependency_text(
        context,
        input_key="prompt",
        candidate_keys=("prompt", "revised_prompt", "creative_brief"),
        default=default if default is not None else context.plan.goal.prompt,
    )


def resolve_dependency_negative_prompt(context: SkillContext) -> str:
    explicit = context.node.inputs.get("negative_prompt")
    if isinstance(explicit, str):
        return explicit
    for dependency in reversed(context.node.depends_on):
        dependency_output = context.state[dependency]
        if isinstance(dependency_output, dict):
            resolved = dependency_output.get("negative_prompt")
            if isinstance(resolved, str):
                return resolved
    return ""


def collect_output_values(
    context: SkillContext,
    keys: tuple[str, ...],
    *,
    first_key_only: bool = False,
) -> list[str]:
    """Collect scalar and list outputs from dependencies without duplicating traversal."""
    collected: list[str] = []
    for dependency in context.node.depends_on:
        dependency_output: Any = context.state[dependency]
        if not isinstance(dependency_output, dict):
            continue
        for key in keys:
            value = dependency_output.get(key)
            if isinstance(value, list):
                if value:
                    collected.extend(str(item) for item in value)
                if first_key_only:
                    break
            elif isinstance(value, str) and value:
                collected.append(value)
                if first_key_only:
                    break
    return collected
