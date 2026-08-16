from __future__ import annotations

import json
import re
from typing import Any


VIDEO_SEMANTIC_QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "checks": {
            "type": "object",
            "properties": {
                "protagonist_clear": {"type": "boolean"},
                "primary_action_visible": {"type": "boolean"},
                "news_anchor_visible": {"type": "boolean"},
                "progression_visible": {"type": "boolean"},
                "cute_hit": {"type": "boolean"},
                "expression_visible": {"type": "boolean"},
                "single_gag": {"type": "boolean"},
                "first_second_action": {"type": "boolean"},
                "action_completion_visible": {"type": "boolean"},
                "payoff_visible": {"type": "boolean"},
                "unwanted_extra_characters": {"type": "boolean"},
            },
            "required": [
                "protagonist_clear",
                "primary_action_visible",
                "news_anchor_visible",
                "progression_visible",
                "cute_hit",
                "expression_visible",
                "single_gag",
                "first_second_action",
                "action_completion_visible",
                "payoff_visible",
                "unwanted_extra_characters",
            ],
            "additionalProperties": False,
        },
        "observed_story": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "caption_guidance": {"type": "string"},
    },
    "required": ["status", "score", "checks", "observed_story", "issues", "caption_guidance"],
    "additionalProperties": False,
}


def build_video_semantic_qa_prompt(
    *,
    character: str,
    story_spine: dict[str, Any],
    native_shots: list[dict[str, Any]],
    news_context: dict[str, Any],
    rendered_prompt: str,
    duration_seconds: int | float | None = None,
) -> str:
    return "\n".join(
        [
            f"Character: {character}",
            f"News context JSON: {json.dumps(news_context or {}, ensure_ascii=False)}",
            f"Story spine JSON: {json.dumps(story_spine or {}, ensure_ascii=False)}",
            f"Native shot plan JSON: {json.dumps(native_shots or [], ensure_ascii=False)}",
            f"Rendered video prompt: {rendered_prompt}",
            "The attached image is a contact sheet sampled from the final video in time order.",
            "Judge only what is visibly supported by the sampled frames; do not infer missing events from the prompt.",
            "Pass only when one protagonist is clear, the primary action is visible, the concrete news-derived visual anchor is present, and the frames show meaningful progression. 'news-derived visual anchor' means the translated object/action from story.news_trace (for example a glowing orb or shadowy tendrils), not a literal news logo, headline, anchorperson, or readable text.",
            "Cute-hit contract: cute_hit is true only when the sampled frames show an immediate, memorable cute or comedic visual beat with a clear emotional read; a generic standing pose, pretty background, or lore reveal is not a cute hit. expression_visible is true only when the protagonist's face or body reaction is readable at a glance. single_gag is true only when one dominant prop/action/reversal carries the clip; multiple plot devices, combat lore, duplicate characters, or abstract exposition fail it. first_second_action is true only when the visible action begins within the first second, not after an atmospheric hold.",
            f"Duration contract: {int(duration_seconds or 0)} seconds. For clips of 6 seconds or less, action_completion_visible is true only when one physical action clearly starts, changes, and finishes in the sampled frames; a zoom, pan, reaction, or repeated holding pose is not completion. For a 15-second clip, payoff_visible is true only when the final sampled frames show the same objective resolved with concrete physical evidence; a static hold or unexplained camera change is not a payoff.",
            "Set unwanted_extra_characters=true only when extra human/child/duplicate characters are clearly visible and not required by the rendered story; set it to false when no such extra character is visible.",
            "The news title must not be treated as visual evidence. If the video is visually unrelated to the story anchor, fail it.",
            "Ignore the grid used to package these sampled frames as a QA contact sheet. For the rendered content itself, a multi-panel, collage, or split-screen layout is a hard visual issue outside the ref2va reference-video route; do not downgrade it to a subjective style advisory.",
            "Use caption_guidance to state what the publish caption may safely claim from the visible result.",
        ]
    )


def normalize_video_semantic_qa(
    payload: Any,
    *,
    contact_sheet_path: str,
    prompt_mode: str,
    llm_backend: dict[str, Any],
    news_anchor_terms: list[str] | None = None,
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    raw_checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    checks = {
        key: bool(raw_checks.get(key, False))
        for key in (
            "protagonist_clear",
            "primary_action_visible",
            "news_anchor_visible",
            "progression_visible",
            "cute_hit",
            "expression_visible",
            "single_gag",
            "first_second_action",
            "action_completion_visible",
            "payoff_visible",
            "unwanted_extra_characters",
        )
    }
    model_status = str(data.get("status") or "uncertain").strip().lower()
    if model_status not in {"pass", "fail", "uncertain"}:
        model_status = "uncertain"
    try:
        score = max(0, min(100, int(data.get("score") or 0)))
    except (TypeError, ValueError):
        score = 0
    observed_story = str(data.get("observed_story") or "").strip()
    # Vision models sometimes understand and describe the translated object
    # correctly, but set the legacy ``news_anchor_visible`` field false because
    # it sounds like a literal newsroom anchor. If the model's own observed
    # story contains a concrete anchor declared by news_trace, normalize that
    # schema mismatch without weakening protagonist/action/progression checks.
    anchor_reconciled = False
    if not checks["news_anchor_visible"] and news_anchor_terms and observed_story:
        observed_tokens = set(re.findall(r"[a-z0-9]+", observed_story.lower()))
        for raw_term in news_anchor_terms:
            term_tokens = [token for token in re.findall(r"[a-z0-9]+", str(raw_term).lower()) if len(token) > 2]
            if len(term_tokens) >= 2 and sum(token in observed_tokens for token in term_tokens) >= max(2, len(term_tokens) - 1):
                checks["news_anchor_visible"] = True
                anchor_reconciled = True
                break
    if anchor_reconciled and model_status in {"fail", "uncertain"} and not any(
        str(item).strip() for item in (data.get("issues") or [])
    ):
        model_status = "pass"
    required_checks_passed = all(
        checks[key]
        for key in (
            "protagonist_clear",
            "primary_action_visible",
            "news_anchor_visible",
            "progression_visible",
            "cute_hit",
            "expression_visible",
            "single_gag",
            "first_second_action",
            "action_completion_visible",
            "payoff_visible",
        )
    )
    issue_text = " ".join(str(item).strip().lower() for item in (data.get("issues") or []))
    multipanel_issue = any(
        token in issue_text
        for token in ("multi-panel", "multipanel", "split-panel", "split screen", "collage", "grid layout")
    )
    hard_visual_failure = not required_checks_passed or checks["unwanted_extra_characters"] or multipanel_issue
    passed = model_status == "pass" and score >= 70 and not hard_visual_failure
    advisory_only = (
        not passed
        and not hard_visual_failure
        and score >= 70
        and model_status in {"fail", "uncertain"}
    )
    status = "pass" if passed else ("review" if advisory_only else "fail")
    return {
        "contact_sheet_path": contact_sheet_path,
        "passed": passed,
        "advisory_only": advisory_only,
        "status": status,
        "score": score,
        "checks": checks,
        "observed_story": observed_story,
        "issues": [str(item).strip() for item in (data.get("issues") or []) if str(item).strip()],
        "caption_guidance": str(data.get("caption_guidance") or "").strip(),
        "prompt_mode": prompt_mode,
        "llm_backend": llm_backend,
    }
