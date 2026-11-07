"""Readable ComfyUI failures and early workflow checks.

ComfyUI rejects a prompt it cannot run with HTTP 400 and a JSON body holding
``error`` plus per-node ``node_errors``. Left raw, that body is a wall of text
(a missing checkpoint even lists every installed model), so it is reduced here
to the few facts a person needs to act on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_DETAIL_LIMIT = 200
_REASONS_SHOWN = 3
_RAW_BODY_LIMIT = 4000


def describe_rejection(status: int, body: str) -> str:
    """Explain a failed ``/prompt`` submission; unstructured replies stay verbatim."""
    prefix = f"ComfyUI /prompt HTTP {status}"
    reasons = _rejection_reasons(body)
    if not reasons:
        return f"{prefix}: {body.strip()[:_RAW_BODY_LIMIT] or 'no response body'}"
    shown = reasons[:_REASONS_SHOWN]
    extra = len(reasons) - len(shown)
    suffix = f"; and {extra} more" if extra else ""
    return f"{prefix}, workflow rejected: " + "; ".join(shown) + suffix


def validate_workflow_graph(workflow: Mapping[str, Any]) -> list[str]:
    """List links that point at nodes missing from an API-format workflow.

    ComfyUI answers such a workflow with an internal KeyError and a stack trace,
    so the broken link is reported here, before anything is submitted.
    """
    node_ids = {str(node_id) for node_id in workflow}
    problems: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping):
            continue
        for input_name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) not in node_ids:
                problems.append(
                    f"{node.get('class_type', 'node')} node {node_id} input "
                    f"'{input_name}' links to missing node {value[0]}"
                )
    return problems


def _rejection_reasons(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []
    entries: list[tuple[str, Mapping[str, Any]]] = []
    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        for node in node_errors.values():
            if isinstance(node, dict):
                class_type = str(node.get("class_type") or "")
                entries.extend(
                    (class_type, error)
                    for error in node.get("errors") or []
                    if isinstance(error, dict)
                )
    top_level = payload.get("error")
    if not entries and isinstance(top_level, dict):
        class_type = str((top_level.get("extra_info") or {}).get("class_type") or "")
        entries.append((class_type, top_level))
    reasons = [_reason(class_type, error) for class_type, error in entries]
    return list(dict.fromkeys(reasons))


def _reason(class_type: str, error: Mapping[str, Any]) -> str:
    extra = error.get("extra_info") or {}
    kind = error.get("type")
    if kind == "value_not_in_list":
        return (
            f"{class_type or 'node'} input '{extra.get('input_name', '?')}': "
            f"'{extra.get('received_value')}' is not installed in ComfyUI"
        )
    if kind == "missing_node_type":
        return f"custom node '{class_type or '?'}' is not installed in ComfyUI"
    text = ": ".join(
        str(part) for part in (error.get("message"), error.get("details")) if part
    )
    text = " ".join(text.split())
    if len(text) > _DETAIL_LIMIT:
        text = text[: _DETAIL_LIMIT - 1] + "…"
    return f"{class_type} {text}".strip()
