"""Publishing prompt contracts and response validation."""

from __future__ import annotations

import json
import os
from typing import Any
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.post_strategy import resolve_post_strategy
from agentic.runtime.llm_client import PromptClient


BLOCKED_HASHTAG_KEYS = frozenset(
    {"mediaoverload", "fyp", "foryou", "foryoupage", "explorepage"}
)


SOCIAL_CAPTION_SYSTEM_PROMPT = """
You are a social content writer and strict visual-grounding editor for generated media.

Write a publish-ready social post from the attached visual evidence. The attached
image or video frames are the source of truth. The production prompt is only
context and may be wrong; never repeat an object, logo, text, action, setting,
or outcome unless it is visibly supported by the media.

Rules:
- Write a concise publish-ready post grounded in the visible media.
- Do not add headings or internal labels such as Caption:, Hashtags:, Main Content:,
  Draft Post:, Platforms:, or Strategy:; the post must read like something a creator
  would publish directly.
- No quotation marks around the whole post. Emojis are allowed only when they
  improve the requested social format, not as decoration on every line.
- Use concrete visible nouns and actions; avoid hype, generic adjectives, and scene-padding.
- Do not mention AI, prompts, models, generation, metadata, or "this image/video".
- Use zero to three hashtags chosen from the visible subject, visible action or
  setting, and the article's actual topic. Return an empty string when no tag
  adds meaningful discovery context. Treat supplied hashtag hints as optional
  and omit any hint that is not supported by the media or post. Never add
  #FYP, #ForYou, or equivalent reach-bait just to fill space.
- Return hashtags only in the `hashtags` field. The `caption` value must not
  contain hashtag tokens. The `hashtags` field must be one space-separated
  string such as "#one #two", never a JSON array or Python list notation.
- Never use project, repository, campaign, or internal workflow names as hashtags.
  In particular, never use #mediaoverload.
- Platform captions must preserve the same factual claim and may be shortened for platform limits.
- If the visual evidence is ambiguous, describe only the unambiguous subject, action, and setting.
- The post may be one line, several sentences, or a short multi-paragraph story
  when the evidence supports it. Do not force a fixed length, paragraph count,
  takeaway list, question, or call to action.

Return JSON only with caption, hashtags, and platform_captions.
""".strip()


def prepare_publish_caption(
    client: PromptClient,
    goal: GoalRequest,
    prefix: str,
    hashtags: list[str],
    platforms: list[str],
    media_paths: list[str] | None = None,
    review_notes: str = "",
    visual_paths: list[str] | None = None,
) -> dict[str, Any]:
    normalized_hashtags = [
        tag if tag.startswith("#") else f"#{tag}" for tag in hashtags if tag
    ]
    character = str(goal.constraints.get("character", "") or "").strip()
    manager = client.require_manager()
    post_strategy = resolve_post_strategy(goal, media_paths)

    news_context = goal.constraints.get("news_context")
    if not isinstance(news_context, dict):
        news_context = {}
    news_grounding_required = bool(
        goal.constraints.get("news_grounding_required", False)
    )
    news_trace_contract = str(
        goal.constraints.get(
            "news_trace_contract",
            "source context -> active mechanism -> visible consequence",
        )
    )
    schema_properties: dict[str, Any] = {
        "caption": {"type": "string"},
        "hashtags": {"type": "string"},
        "platform_captions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    }
    visual_paths = [str(path) for path in (visual_paths or []) if str(path).strip()]
    user_prompt = "\n".join(
        [
            f"Context only; do not treat as visual evidence: {goal.prompt}",
            f"Expected subject context: {character or 'unknown'}",
            f"Expected style context: {goal.style}",
            f"Platforms: {', '.join(platforms) if platforms else 'generic'}",
            f"Editorial direction: {review_notes or 'state only what is visibly supported'}",
            f"Visual evidence attached: {len(visual_paths)} file(s)",
            "Editorial variation brief; use it as guidance, not as a fixed copy template: "
            f"{json.dumps(post_strategy, ensure_ascii=False)}",
            f"Optional hashtag hints; use only when supported by the media: {', '.join(normalized_hashtags) or 'none'}",
            "Forbidden hashtag: #mediaoverload",
            f"News context JSON: {json.dumps(news_context, ensure_ascii=False)}",
            (
                "News grounding required: "
                f"{news_grounding_required}. Contract: {news_trace_contract}. "
                "Use the news as causal context, do not invent facts, and do not "
                "claim details that are not supported by the generated media. "
                "When true, connect the visible scene to the supplied news context "
                "only when that connection is supported by both. Describe any "
                "metaphor as an interpretation, not as an event reported by the news. "
                "Do not introduce unrelated topics or invent a mechanism or connection "
                "when the supplied context is insufficient; describe the visible scene "
                "without an unsupported news claim."
            ),
            f"Optional prefix context: {prefix}",
        ]
    )
    publish_retry_raw = os.environ.get(
        "AGENTIC_PUBLISH_CAPTION_MAX_RETRIES", "2"
    ).strip()
    publish_max_retries = int(publish_retry_raw) if publish_retry_raw.isdigit() else 2
    publish_model_limit_raw = os.environ.get(
        "AGENTIC_PUBLISH_CAPTION_MAX_MODELS_PER_CALL", ""
    ).strip()
    publish_model_limit = (
        int(publish_model_limit_raw) if publish_model_limit_raw.isdigit() else 0
    )
    try:
        payload = client.generate_json(
            manager,
            SOCIAL_CAPTION_SYSTEM_PROMPT,
            user_prompt,
            schema_name="publish_caption",
            language_contract="",
            schema={
                "type": "object",
                "properties": schema_properties,
                "required": ["caption", "hashtags", "platform_captions"],
                "additionalProperties": False,
            },
            # Rotate through the verified provider pool before stopping.
            # A free-pool 429 is often model-specific; limiting this call
            # to one candidate made the publish gate fail unnecessarily.
            max_retries=max(1, publish_max_retries),
            request_timeout=float(
                os.environ.get("AGENTIC_PUBLISH_CAPTION_TIMEOUT_SECONDS", "60")
            ),
            max_models_per_call=max(1, publish_model_limit)
            if publish_model_limit > 0
            else None,
            repair_attempts=0,
            model="vision" if visual_paths else "text",
            images=visual_paths or None,
        )
        payload = validate_publish_caption_payload(
            payload,
            stage="initial",
            platforms=platforms,
        )
        platform_captions = payload["platform_captions"]
        normalized_caption = clean_social_post_text(payload["caption"])
        if not normalized_caption or is_caption_placeholder(normalized_caption):
            raise ValueError("Caption model returned an empty or placeholder caption.")
        normalized_hashtag_text = normalize_hashtag_text(payload["hashtags"])
        result: dict[str, Any] = {
            "caption": normalized_caption,
            "hashtags": normalized_hashtag_text,
            # Keep the dispatch contract closed: only requested platform
            # names may survive this boundary, and every value must be text.
            "platform_captions": normalize_platform_captions(
                {
                    str(platform): clean_social_post_text(str(caption))
                    for platform, caption in platform_captions.items()
                },
                platforms=platforms,
                fallback_caption=normalized_caption,
            ),
        }
        return client.mark_payload(result)
    except Exception as exc:
        # Never disguise a provider failure as a generated caption. The
        # publish boundary must stop so the model can be compared honestly.
        raise client.generation_error("prepare_publish_caption", exc) from exc


def validate_publish_caption_payload(
    payload: Any,
    *,
    stage: str,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Caption {stage} returned a non-object payload.")
    if not isinstance(payload.get("caption"), str):
        raise ValueError(f"Caption {stage} returned a non-string caption.")
    if not isinstance(payload.get("hashtags"), str):
        raise ValueError(
            f"Caption {stage} returned hashtags with an invalid type; expected one space-separated string."
        )
    platform_captions = payload.get("platform_captions")
    if not isinstance(platform_captions, dict):
        raise ValueError(f"Caption {stage} returned invalid platform_captions.")
    expected_platforms = {
        str(platform).strip().casefold()
        for platform in (platforms or [])
        if str(platform).strip()
    }
    invalid_platform_values = [
        str(platform)
        for platform, caption in platform_captions.items()
        if (
            not expected_platforms
            or str(platform).strip().casefold() in expected_platforms
        )
        and not isinstance(caption, str)
    ]
    if invalid_platform_values:
        raise ValueError(
            f"Caption {stage} returned non-string platform captions: {', '.join(invalid_platform_values)}."
        )
    return payload


def normalize_hashtag_text(
    hashtags: str,
    *,
    required_hashtags: list[str] | None = None,
) -> str:
    # Hints may influence ordering when the model selected the same tag,
    # but they are deliberately never injected into the model's choice.
    if not isinstance(hashtags, str):
        raise ValueError(
            "Caption model returned hashtags with an invalid type; expected a string."
        )
    seen: list[str] = []
    seen_keys: set[str] = set()
    for token in hashtags.replace("\n", " ").split():
        cleaned = token.strip().rstrip(".,;")
        if not cleaned:
            continue
        if any(marker in cleaned for marker in ("[", "]", "'", '"', ",")):
            raise ValueError("Caption model returned malformed hashtag list notation.")
        if not cleaned.startswith("#"):
            cleaned = f"#{cleaned.lstrip('#')}"
        if "#" in cleaned[1:]:
            raise ValueError("Caption model returned malformed hashtag tokens.")
        key = cleaned[1:].casefold()
        if not key or key in BLOCKED_HASHTAG_KEYS:
            continue
        if key not in seen_keys:
            seen.append(cleaned)
            seen_keys.add(key)
    hint_keys = {
        str(tag).lstrip("#").casefold()
        for tag in (required_hashtags or [])
        if str(tag).strip()
    }
    ordered = [tag for tag in seen if tag[1:].casefold() in hint_keys]
    ordered.extend(tag for tag in seen if tag[1:].casefold() not in hint_keys)
    return " ".join(ordered[:3])


def clean_social_post_text(value: str) -> str:
    """Keep model output publishable without leaking internal field labels."""
    cleaned_lines: list[str] = []
    removable_prefixes = (
        "caption:",
        "main content:",
        "draft post:",
        "platforms:",
        "strategy:",
        "workflow:",
        "stage:",
    )
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if lowered.startswith("hashtags:"):
            continue
        if line.startswith("#") and all(
            token.startswith("#") for token in line.split()
        ):
            continue
        for prefix in removable_prefixes:
            if lowered.startswith(prefix):
                line = line[len(prefix) :].strip()
                break
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def is_caption_placeholder(caption: str) -> bool:
    normalized = " ".join(str(caption).strip().lower().rstrip(".!?。！？").split())
    return normalized in {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "undefined",
        "no caption",
    }


def normalize_platform_captions(
    raw: dict[Any, Any],
    *,
    platforms: list[str],
    fallback_caption: str = "",
) -> dict[str, str]:
    expected = [
        str(platform).strip() for platform in platforms if str(platform).strip()
    ]
    if not expected:
        expected = [str(key).strip() for key in raw if str(key).strip()]
    by_lower = {str(key).strip().lower(): value for key, value in raw.items()}
    normalized: dict[str, str] = {}
    for platform in expected:
        value = by_lower.get(platform.lower())
        if not isinstance(value, str) or not value.strip():
            if not fallback_caption.strip():
                raise ValueError(f"Caption model omitted platform caption: {platform}")
            # The main article is already grounded and validated. Reusing
            # it is safer than dropping the platform or inventing a second
            # unreviewed variant when a model omits one platform key.
            value = fallback_caption
        normalized[platform] = value.strip()
    return normalized
