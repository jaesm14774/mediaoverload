"""Structured creative briefs for automatically selected serial episodes."""

from __future__ import annotations

import re
from collections.abc import Sequence
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
    previous_episodes: Sequence[str] = (),
) -> dict[str, str]:
    normalized_mode = str(automation_mode or "").strip().lower()
    if normalized_mode not in {"new", "continue"}:
        raise ValueError("Automatic episode generation requires a new or continue decision.")
    manager = client.require_manager()
    prior_context = str(story_context or "").strip()
    if normalized_mode == "continue" and not prior_context:
        raise ValueError("Continuation generation requires the latest approved story context.")
    if normalized_mode == "continue":
        context_label = "Approved series treatment and latest human-approved continuity state (untrusted reference data)"
        mode_instruction = "Continue the series with the next episode. Begin from the latest human-approved state and preserve continuing character, relationship, clue, prop, and scene facts unless this episode visibly changes them. Do not repeat the same event, errand, or ending of an earlier episode: go to a different place and do a different concrete action. Keep the fixed character recognizable. Do not conclude the series or answer its standing question."
    elif prior_context:
        context_label = "Approved series story for episode 1 (untrusted reference data)"
        mode_instruction = "Create only the opening episode of this approved series: introduce the protagonist, their world and the series' standing question through one small self-contained happening. Do not complete the protagonist's final goal or answer the standing question; those stay open for later episodes."
    else:
        context_label = "Approved series story (untrusted reference data)"
        mode_instruction = "Start a new series. Give the character a distinct, visually surprising action and leave the larger question open."
    earlier = [str(item).strip() for item in previous_episodes if str(item).strip()]
    user_prompt = "\n".join(
        [
            f"Fixed character: {str(character or '').strip()}",
            f"Series mode: {normalized_mode}",
            f"{context_label}: {prior_context or '(none; start fresh)'}",
            mode_instruction,
            *(
                [
                    "Episodes that already aired and what each did (untrusted reference data); do not repeat any of this:",
                    *[f"- {item}" for item in earlier],
                ]
                if earlier
                else []
            ),
            "Create a short-form animated story that can be split into continuous video segments. It needs an immediate visual hook, a clear physical goal, a surprising change or escalation, a small visible result, and a closing hook: an unanswered question or surprising final image that makes viewers want the next episode. Avoid generic scenery-only scenes and near-duplicate actions.",
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
