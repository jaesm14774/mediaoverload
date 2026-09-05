from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import GoalRequest


POST_STRATEGY_VERSION = "2026-09-02.v1"

_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "visible_moment",
        "editorial_question": "What single visible moment changes the viewer's understanding?",
        "prompt_direction": "Lead with one concrete visible change, tension, or surprise.",
        "hook_mode": "specific_visible_change",
        "payoff_mode": "visible_consequence",
        "cta_policy": "none unless a specific observation question is natural",
        "hashtag_policy": "zero_or_relevant_semantic_terms",
    },
    {
        "variant_id": "cause_and_effect",
        "editorial_question": "What visible action causes the next visible consequence?",
        "prompt_direction": "Make one supported cause-and-effect relationship easy to understand.",
        "hook_mode": "cause_or_trigger",
        "payoff_mode": "resulting_state",
        "cta_policy": "none unless the causal turn invites a precise reply",
        "hashtag_policy": "zero_or_relevant_semantic_terms",
    },
    {
        "variant_id": "character_choice",
        "editorial_question": "What choice or reaction reveals the protagonist?",
        "prompt_direction": "Center the protagonist's visible choice, reaction, or trade-off; do not add inner thoughts as facts.",
        "hook_mode": "character_decision",
        "payoff_mode": "choice_revealed_by_action",
        "cta_policy": "a specific question is allowed only when the visible choice supports it",
        "hashtag_policy": "zero_or_character_or_action_terms",
    },
    {
        "variant_id": "replay_detail",
        "editorial_question": "Which small visible detail rewards a second look?",
        "prompt_direction": "Point to a concrete detail that is actually visible and gives the viewer a reason to rewatch.",
        "hook_mode": "detail_to_notice",
        "payoff_mode": "detail_reframed",
        "cta_policy": "invite a specific observation only if the detail is unambiguous",
        "hashtag_policy": "zero_or_one_specific_term",
    },
    {
        "variant_id": "contrast",
        "editorial_question": "What visible contrast or state change carries the post?",
        "prompt_direction": "Use a before-and-after, scale, color, or motion contrast only when both sides are visible.",
        "hook_mode": "contrast",
        "payoff_mode": "state_change",
        "cta_policy": "none unless the contrast creates a concrete comparison question",
        "hashtag_policy": "zero_or_relevant_semantic_terms",
    },
    {
        "variant_id": "news_mechanism_bridge",
        "editorial_question": "What sourced real-world mechanism is this visual metaphor translating?",
        "prompt_direction": "Connect one verified news mechanism to the visual metaphor, while clearly separating the artwork from real incident evidence.",
        "hook_mode": "news_mechanism",
        "payoff_mode": "visual_translation",
        "cta_policy": "none unless a precise mechanism question is useful",
        "hashtag_policy": "zero_or_source_topic_terms",
    },
)


def resolve_post_strategy(
    goal: GoalRequest,
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve a varied editorial brief without imposing a caption template.

    The brief is deterministic for a given asset/context set so a run can be
    audited and reproduced. It describes what to notice, not the shape or
    length of the final copy. Human-approved copy remains authoritative.
    """
    constraints = goal.constraints if isinstance(goal.constraints, dict) else {}
    news_context = constraints.get("news_context")
    has_news_context = isinstance(news_context, dict) and bool(
        str(news_context.get("title") or news_context.get("topic") or "").strip()
    )
    explicit_variant = str(constraints.get("post_strategy_id") or "").strip().casefold()
    variant_by_id = {item["variant_id"]: item for item in _VARIANTS}
    selected = variant_by_id.get(explicit_variant)
    if selected and selected["variant_id"] == "news_mechanism_bridge" and not has_news_context:
        selected = None
    selection_mode = "explicit" if selected else "stable_context_hash"

    seed_override = str(constraints.get("post_strategy_seed") or "").strip()
    seed_source = "constraint:post_strategy_seed" if seed_override else "goal_and_media_context"
    seed_parts = [
        seed_override or str(goal.prompt or ""),
        str(goal.media_type or ""),
        str(goal.style or ""),
        "news=" + ("1" if has_news_context else "0"),
    ]
    for path in media_paths or []:
        # Hash only the basename so the trace does not expose absolute paths.
        seed_parts.append(Path(str(path).replace("\\", "/")).name)
    if isinstance(news_context, dict):
        seed_parts.extend(
            str(news_context.get(key) or "").strip()
            for key in ("title", "topic", "source")
        )
    digest = hashlib.sha256("|".join(seed_parts).encode("utf-8")).hexdigest()

    if selected is None:
        eligible = [
            item
            for item in _VARIANTS
            if item["variant_id"] != "news_mechanism_bridge" or has_news_context
        ]
        selected = eligible[int(digest[:8], 16) % len(eligible)]

    return {
        "strategy_version": POST_STRATEGY_VERSION,
        "variant_id": selected["variant_id"],
        "variation_key": digest[:12],
        "selection_mode": selection_mode,
        "selection_seed_source": seed_source,
        "editorial_question": selected["editorial_question"],
        "prompt_direction": selected["prompt_direction"],
        "hook_mode": selected["hook_mode"],
        "payoff_mode": selected["payoff_mode"],
        "cta_policy": selected["cta_policy"],
        "hashtag_policy": selected["hashtag_policy"],
        "discovery_terms": _discovery_terms(news_context),
    }


def _discovery_terms(news_context: Any) -> list[str]:
    """Return explicit source terms for traceability, never inferred keywords."""
    if not isinstance(news_context, dict):
        return []
    terms: list[str] = []
    for key in ("topic", "title", "keywords", "entities"):
        value = news_context.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            term = " ".join(str(item or "").split()).strip("#")
            if not term or term.casefold() in {existing.casefold() for existing in terms}:
                continue
            if len(term) <= 80:
                terms.append(term)
            if len(terms) >= 8:
                return terms
    return terms
