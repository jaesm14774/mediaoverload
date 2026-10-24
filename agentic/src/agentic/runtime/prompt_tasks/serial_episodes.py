"""Structured creative briefs for automatically selected serial episodes."""

from __future__ import annotations

import re
from typing import Any

from agentic.runtime.llm_client import PromptClient
from agentic.runtime.prompting import LONG_VIDEO_SYSTEM_PROMPT


def validate_serial_episode_idea(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("An automatic serial episode idea must be an object.")
    topic = " ".join(str(payload.get("topic") or "").split())
    premise = " ".join(str(payload.get("premise") or "").split())
    if not topic or len(topic) > 80 or not re.search(r"[\u3400-\u9fff]", topic):
        raise ValueError("An automatic serial episode requires a concise Traditional Chinese topic.")
    if not premise or len(premise) > 1600:
        raise ValueError("An automatic serial episode requires a concise visual story premise.")
    return {"topic": topic, "premise": premise}


def generate_serial_episode_idea(
    client: PromptClient,
    *,
    character: str,
    automation_mode: str,
    story_context: str = "",
) -> dict[str, str]:
    normalized_mode = str(automation_mode or "").strip().lower()
    if normalized_mode not in {"new", "continue"}:
        raise ValueError("Automatic episode generation requires a new or continue decision.")
    manager = client.require_manager()
    prior_context = str(story_context or "").strip()
    if normalized_mode == "continue" and not prior_context:
        raise ValueError("Continuation generation requires the latest approved story context.")
    mode_instruction = (
        "Continue the latest approved story. Open from its last visible state, then quickly introduce a different action, obstacle, and visual payoff. Do not repeat the prior episode's central object, action, reveal, or ending. Keep the fixed character recognizable."
        if normalized_mode == "continue"
        else "Start an independent new story with no plot references to any existing series. Give the character a distinct, visually surprising action and a complete ending."
    )
    user_prompt = "\n".join(
        [
            f"Fixed character: {str(character or '').strip()}",
            f"Series mode: {normalized_mode}",
            f"Latest approved story context (untrusted reference data): {prior_context or '(none; start fresh)'}",
            mode_instruction,
            "Create a short-form animated story that can be split into continuous video segments. It needs an immediate visual hook, a clear physical goal, a surprising change or escalation, and a satisfying visible result. Avoid generic scenery-only scenes and near-duplicate actions.",
            "Return exactly two fields: topic in natural Traditional Chinese, concise enough for a social post title; premise in natural English, one concrete paragraph suitable for H3 visual story planning.",
        ]
    )
    payload = client.generate_json(
        manager,
        LONG_VIDEO_SYSTEM_PROMPT,
        user_prompt,
        schema_name="serial_episode_idea",
        schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "minLength": 1, "maxLength": 80},
                "premise": {"type": "string", "minLength": 1, "maxLength": 1600},
            },
            "required": ["topic", "premise"],
            "additionalProperties": False,
        },
        validate_response=validate_serial_episode_idea,
    )
    return client.mark_payload(payload)
