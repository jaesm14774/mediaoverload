"""Routing prompt contracts and response validation."""

from __future__ import annotations

import json
import re
from typing import Any
from agentic.runtime.strategies import WORKFLOW_STAGE_KEYS, active_count_policy
from agentic.runtime.prompt_requests import GenerationRoutingRequest
from agentic.runtime.prompting import LONG_VIDEO_SYSTEM_PROMPT
from agentic.runtime.llm_client import PromptClient


def route_generation_strategy(
    client: PromptClient,
    request: GenerationRoutingRequest,
) -> dict[str, Any]:
    normalized_candidates = [
        str(item).strip()
        for item in request.generation_type_candidates
        if str(item).strip()
    ]
    if not normalized_candidates:
        raise ValueError("generation_type_candidates cannot be empty for LLM routing.")
    manager = client.require_manager()
    route_schema = build_generation_strategy_schema(
        generation_type_candidates=normalized_candidates,
        workflow_stage_candidates=request.workflow_stage_candidates,
        count_policies=request.count_policies,
    )
    user_prompt = "\n".join(
        re.sub(r"(?<!\w)Kirby(?!\w)", str(request.character), line, flags=re.IGNORECASE)
        for line in [
            f"Character: {request.character}",
            f"Prompt: {request.prompt}",
            f"Style: {request.style}",
            f"Preferred generation type override: {request.preferred_generation_type or ''}",
            f"Generation type candidates JSON: {json.dumps(normalized_candidates, ensure_ascii=False)}",
            f"Workflow stage candidates JSON: {json.dumps(request.workflow_stage_candidates, ensure_ascii=False)}",
            f"Count policies JSON: {json.dumps(request.count_policies, ensure_ascii=False)}",
            f"Routing hints JSON: {json.dumps(request.routing_hints, ensure_ascii=False)}",
            "Pick the best generation_type, stage workflows, and count plan for the user's request.",
            "Only choose values from the provided candidate lists and policy ranges.",
            "You must populate the workflow stages needed by the chosen generation_type.",
            "Return JSON with keys: generation_type, workflow_plan, count_plan, reason.",
        ]
    )
    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="generation_strategy_route",
            schema=route_schema,
        )
        selected_generation_type = str(payload.get("generation_type") or "").strip()
        if selected_generation_type not in normalized_candidates:
            raise ValueError(
                f"LLM selected unsupported generation_type: {selected_generation_type}"
            )
        workflow_plan = dict(payload.get("workflow_plan") or {})
        count_plan = dict(payload.get("count_plan") or {})
        allowed_stage_candidates = request.workflow_stage_candidates.get(
            selected_generation_type, {}
        )
        normalized_workflow_plan: dict[str, str] = {}
        workflow_corrections: list[str] = []
        for stage_key in WORKFLOW_STAGE_KEYS:
            selected_workflow = str(workflow_plan.get(stage_key) or "").strip()
            allowed_workflows = [
                str(item).strip()
                for item in allowed_stage_candidates.get(stage_key, [])
                if str(item).strip()
            ]
            if allowed_workflows:
                if not selected_workflow:
                    selected_workflow = allowed_workflows[0]
                    workflow_corrections.append(
                        f"{stage_key}=default:{selected_workflow}"
                    )
                elif selected_workflow not in set(allowed_workflows):
                    selected_workflow = allowed_workflows[0]
                    workflow_corrections.append(
                        f"{stage_key}=fallback:{selected_workflow}"
                    )
            elif selected_workflow:
                raise ValueError(
                    f"LLM selected workflow '{selected_workflow}' for unavailable stage '{stage_key}' "
                    f"under generation_type '{selected_generation_type}'."
                )
            normalized_workflow_plan[stage_key] = selected_workflow
        normalized_count_plan: dict[str, int] = {}
        allowed_count_policy = active_count_policy(
            selected_generation_type,
            allowed_stage_candidates,
            request.count_policies.get(selected_generation_type, {}),
        )
        for count_key, policy in allowed_count_policy.items():
            if not isinstance(policy, dict):
                continue
            minimum = int(policy["min"])
            maximum = int(policy["max"])
            # Some OpenRouter free-pool models occasionally omit a field
            # despite the JSON-schema contract. Use the policy minimum as
            # the deterministic safe default instead of failing the whole
            # workflow before generation begins.
            value = int(count_plan.get(count_key, minimum))
            if value < minimum or value > maximum:
                raise ValueError(
                    f"LLM selected out-of-range {count_key}={value} for generation_type '{selected_generation_type}'."
                )
            normalized_count_plan[count_key] = value
        reason = str(payload.get("reason") or "").strip()
        if workflow_corrections:
            correction_note = (
                f"workflow defaults applied ({', '.join(workflow_corrections)})"
            )
            reason = f"{reason}; {correction_note}" if reason else correction_note
        return client.mark_payload(
            {
                "generation_type": selected_generation_type,
                "workflow_plan": normalized_workflow_plan,
                "count_plan": normalized_count_plan,
                "reason": reason,
            }
        )
    except Exception as exc:
        raise client.generation_error("route_generation_strategy", exc) from exc


def build_generation_strategy_schema(
    *,
    generation_type_candidates: list[str],
    workflow_stage_candidates: dict[str, dict[str, list[str]]],
    count_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    branches: list[dict[str, Any]] = []
    for generation_type in generation_type_candidates:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "generation_type": {"const": generation_type},
                    "workflow_plan": build_workflow_plan_schema(
                        workflow_stage_candidates.get(generation_type, {})
                    ),
                    "count_plan": build_count_plan_schema(
                        active_count_policy(
                            generation_type,
                            workflow_stage_candidates.get(generation_type, {}),
                            count_policies.get(generation_type, {}),
                        )
                    ),
                    "reason": {"type": "string"},
                },
                "required": [
                    "generation_type",
                    "workflow_plan",
                    "count_plan",
                    "reason",
                ],
                "additionalProperties": False,
            }
        )
    return {"oneOf": branches}


def build_workflow_plan_schema(
    stage_candidates: dict[str, list[str]],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for stage_key in WORKFLOW_STAGE_KEYS:
        allowed_workflows = [
            str(item).strip()
            for item in stage_candidates.get(stage_key, [])
            if str(item).strip()
        ]
        if allowed_workflows:
            properties[stage_key] = {"type": "string", "enum": allowed_workflows + [""]}
        else:
            properties[stage_key] = {"const": ""}
    return {
        "type": "object",
        "properties": properties,
        "required": list(WORKFLOW_STAGE_KEYS),
        "additionalProperties": False,
    }


def build_count_plan_schema(count_policy: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required_keys = tuple(
        count_key
        for count_key, policy in count_policy.items()
        if isinstance(policy, dict)
    )
    for count_key in required_keys:
        policy = count_policy.get(count_key)
        if isinstance(policy, dict):
            minimum = int(policy.get("min", 1))
            maximum = int(policy.get("max", minimum))
            properties[count_key] = {
                "type": "integer",
                "minimum": minimum,
                "maximum": max(minimum, maximum),
            }
        else:
            properties[count_key] = {"type": "integer", "minimum": 1, "maximum": 32}
    return {
        "type": "object",
        "properties": properties,
        "required": list(required_keys),
        "additionalProperties": False,
    }
