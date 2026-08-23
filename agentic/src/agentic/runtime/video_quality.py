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
                "required_subjects_clear": {"type": "boolean"},
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
                "interaction_visible": {"type": "boolean"},
                "unexpected_extra_subjects": {"type": "boolean"},
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
    subject_context: dict[str, Any] | None = None,
) -> str:
    context = dict(subject_context or {})
    subjects = [
        str(item.get("name") or "").strip()
        for item in (context.get("subjects") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    interaction_required = bool(dict(context.get("interaction_contract") or {}).get("required", False))
    subject_contract = (
        f"The required subject slots are: {', '.join(subjects)}. Both slots may have the same name; judge them as two declared visual slots."
        if interaction_required and len(subjects) == 2
        else f"The required protagonist is: {character}."
    )
    return "\n".join(
        [
            f"Character: {character}",
            subject_contract,
            f"News context JSON: {json.dumps(news_context or {}, ensure_ascii=False)}",
            f"Story spine JSON: {json.dumps(story_spine or {}, ensure_ascii=False)}",
            f"Native shot plan JSON: {json.dumps(native_shots or [], ensure_ascii=False)}",
            f"Rendered video prompt: {rendered_prompt}",
            "The attached image is a contact sheet sampled from the final video in time order.",
            "Judge only what is visibly supported by the sampled frames; do not infer missing events from the prompt.",
            (
                "Pass only when both declared subject slots are clear, the required interaction is visible, the primary action is visible, the concrete news-derived visual anchor is present, and the frames show meaningful progression."
                if interaction_required
                else "Pass only when one protagonist is clear, the primary action is visible, the concrete news-derived visual anchor is present, and the frames show meaningful progression."
            ) + " 'news-derived visual anchor' means the translated object/action from story.news_trace (for example a glowing orb or shadowy tendrils), not a literal news logo, headline, anchorperson, or readable text.",
            "Cute-hit contract: cute_hit is true only when the sampled frames show an immediate, memorable cute or comedic visual beat with a clear emotional read; a generic standing pose, pretty background, or lore reveal is not a cute hit. expression_visible is true only when the required subject's face or body reaction is readable at a glance. interaction_visible is true only when the declared subject slots visibly affect, respond to, touch, look toward, or jointly manipulate one another or one shared mechanism. single_gag is true only when one dominant prop/action/reversal carries the clip; multiple plot devices, combat lore, unrequested third subjects, or abstract exposition fail it. first_second_action is true only when the visible action begins within the first second, not after an atmospheric hold.",
            "Reference-derived quality signals: a small protagonist against one oversized prop or environment, tactile contact with a real-feeling object, readable squash/stretch or recoil, generous negative space, and a final image that echoes the opening are positive signals when they serve the same gag. Do not require all signals, and do not award them for decorative spectacle without cause and effect.",
            f"Duration contract: {int(duration_seconds or 0)} seconds. For clips of 6 seconds or less, action_completion_visible is true only when one physical action clearly starts, changes, and finishes in the sampled frames; a zoom, pan, reaction, or repeated holding pose is not completion. For a 15-second clip, payoff_visible is true only when the final sampled frames show the same objective resolved with concrete physical evidence; a static hold or unexplained camera change is not a payoff.",
            "Read the contact-sheet panels in chronological order, left-to-right and then top-to-bottom. Each panel is a still sample from a moving video; do not call the rendered video static merely because the QA evidence is packaged as still images. Judge progression from visible changes between adjacent panels.",
            "Set unexpected_extra_subjects=true when any subject outside the declared subject slots is clearly visible and not required by the rendered story. Keep the legacy unwanted_extra_characters field aligned with the same judgement.",
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
    subject_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    raw_checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
    context = dict(subject_context or {})
    interaction_required = bool(dict(context.get("interaction_contract") or {}).get("required", False))
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
    required_subjects_clear = bool(
        raw_checks.get(
            "required_subjects_clear",
            checks["protagonist_clear"] if not interaction_required else False,
        )
    )
    interaction_visible = bool(raw_checks.get("interaction_visible", False))
    unexpected_extra_subjects = bool(
        raw_checks.get("unexpected_extra_subjects", raw_checks.get("unwanted_extra_characters", False))
    )
    checks["required_subjects_clear"] = required_subjects_clear
    checks["interaction_visible"] = interaction_visible
    checks["unexpected_extra_subjects"] = unexpected_extra_subjects
    checks["unwanted_extra_characters"] = unexpected_extra_subjects
    model_status = str(data.get("status") or "uncertain").strip().lower()
    if model_status not in {"pass", "fail", "uncertain"}:
        model_status = "uncertain"
    try:
        score = max(0, min(100, int(data.get("score") or 0)))
    except (TypeError, ValueError):
        score = 0
    observed_story = str(data.get("observed_story") or "").strip()
    # Vision models sometimes describe the translated object correctly while
    # under-reporting the canonical news-anchor check. Reconcile that result
    # only when the observed story contains a declared concrete anchor.
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
    required_checks = [
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
    ]
    if interaction_required:
        required_checks.extend([required_subjects_clear, interaction_visible])
    required_checks_passed = all(required_checks)
    issue_text = " ".join(str(item).strip().lower() for item in (data.get("issues") or []))
    multipanel_issue = any(
        token in issue_text
        for token in ("multi-panel", "multipanel", "split-panel", "split screen", "collage", "grid layout")
    )
    hard_visual_failure = not required_checks_passed or unexpected_extra_subjects or multipanel_issue
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
        "subject_count": 2 if interaction_required else 1,
        "observed_story": observed_story,
        "issues": [str(item).strip() for item in (data.get("issues") or []) if str(item).strip()],
        "caption_guidance": str(data.get("caption_guidance") or "").strip(),
        "prompt_mode": prompt_mode,
        "llm_backend": llm_backend,
    }
