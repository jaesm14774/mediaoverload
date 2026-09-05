from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.post_strategy import resolve_post_strategy


PLATFORM_STRATEGY_VERSION = "2026-09-02.v2"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
YOUTUBE_TITLE_MAX = 100
YOUTUBE_DESCRIPTION_MAX = 5000
FACEBOOK_CAPTION_MAX = 600
FACEBOOK_HASHTAG_MAX = 3
YOUTUBE_TAG_MAX = 12

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "create",
    "clip",
    "content",
    "for",
    "from",
    "generate",
    "in",
    "media",
    "of",
    "on",
    "publish",
    "social",
    "the",
    "this",
    "to",
    "video",
    "with",
}


def build_platform_bundle(
    *,
    goal: GoalRequest,
    caption: str,
    hashtags: str,
    platform_captions: dict[str, str],
    platforms: list[str],
    media_paths: list[str] | None,
    post_strategy: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build platform-native packages without expanding the generic caption contract.

    The LLM remains responsible for factual, visually grounded prose. This
    boundary only packages that prose for each platform and records structural
    checks that can be inspected before dispatch.
    """
    selected_media = [str(path) for path in (media_paths or []) if str(path).strip()]
    safe_hashtags = sanitize_hashtags(hashtags)
    effective_post_strategy = dict(post_strategy or resolve_post_strategy(goal, selected_media))
    effective_platforms = [str(platform) for platform in platforms]
    if not effective_platforms:
        effective_platforms = [str(platform) for platform in platform_captions]

    bundle: dict[str, dict[str, Any]] = {}
    for platform in effective_platforms:
        platform_name = str(platform)
        source_caption = str(platform_captions.get(platform_name) or caption).strip()
        entry = _base_entry(source_caption, safe_hashtags, selected_media)
        entry["post_strategy"] = dict(effective_post_strategy)
        normalized_name = platform_name.casefold()
        if normalized_name == "youtube":
            entry.update(
                _build_youtube_package(
                    goal=goal,
                    caption=source_caption,
                    hashtags=safe_hashtags,
                    media_paths=selected_media,
                    post_strategy=effective_post_strategy,
                )
            )
        elif normalized_name.startswith("facebook"):
            entry.update(
                _build_facebook_package(
                    goal=goal,
                    caption=source_caption,
                    hashtags=safe_hashtags,
                    media_paths=selected_media,
                    post_strategy=effective_post_strategy,
                )
            )
        bundle[platform_name] = entry
    return bundle


def _base_entry(caption: str, hashtags: str, media_paths: list[str]) -> dict[str, Any]:
    return {
        "caption": caption,
        "hashtags": str(hashtags or "").strip(),
        "character_count": len(caption),
        "validation": {
            "has_caption": bool(caption),
            "has_media": bool(media_paths),
            "is_platform_eligible": bool(media_paths),
            "is_platform_publish_ready": bool(caption) and bool(media_paths),
            "is_publish_ready": bool(caption) and bool(media_paths),
            "issues": [] if caption and media_paths else ["caption_and_media_required"],
            "warnings": [],
        },
    }


def _build_youtube_package(
    *,
    goal: GoalRequest,
    caption: str,
    hashtags: str,
    media_paths: list[str],
    post_strategy: dict[str, Any],
) -> dict[str, Any]:
    video_paths = _paths_with_extensions(media_paths, VIDEO_EXTENSIONS)
    answer_line = _answer_first_line(caption)
    title = _youtube_title(answer_line, goal.prompt)
    description = _youtube_description(caption, answer_line, hashtags)
    tags = _youtube_tags(hashtags)
    # Media produced by this runtime is synthetic by default. A caller may
    # not turn off the disclosure through an unverified goal constraint.
    synthetic_media = True
    issues: list[str] = []
    if not video_paths:
        issues.append("requires_video_media")
    if not title:
        issues.append("youtube_title_required")
    if len(title) > YOUTUBE_TITLE_MAX:
        issues.append("youtube_title_exceeds_100_characters")
    if len(description) > YOUTUBE_DESCRIPTION_MAX:
        issues.append("youtube_description_exceeds_5000_characters")
    if not answer_line:
        issues.append("answer_first_line_required")

    risk_flags = _content_risk_flags(caption, hashtags)
    platform_blockers = [
        issue for issue in issues if issue != "requires_video_media"
    ]
    return {
        "format": "video",
        "metadata_source": "derived",
        "additional_params": {
            "youtube_title": title[:YOUTUBE_TITLE_MAX],
            "youtube_description": description[:YOUTUBE_DESCRIPTION_MAX],
            "youtube_tags": tags,
            "youtube_contains_synthetic_media": synthetic_media,
        },
        "content_strategy": {
            "strategy_version": PLATFORM_STRATEGY_VERSION,
            "platform": "youtube",
            "format": "video",
            "search_intent": _search_intent(goal.prompt),
            "answer_first": bool(answer_line),
            "post_strategy_version": str(post_strategy.get("strategy_version") or ""),
            "variant_id": str(post_strategy.get("variant_id") or ""),
            "variation_key": str(post_strategy.get("variation_key") or ""),
            "editorial_question": str(post_strategy.get("editorial_question") or ""),
            "hook": str(post_strategy.get("hook_mode") or ""),
            "payoff": str(post_strategy.get("payoff_mode") or ""),
            "cta_policy": str(post_strategy.get("cta_policy") or ""),
            "hashtag_policy": str(post_strategy.get("hashtag_policy") or ""),
            "discovery_terms": list(post_strategy.get("discovery_terms") or []),
            "originality_basis": str(
                goal.constraints.get("originality_basis") or "creator_produced_or_owned_media"
            ),
            "synthetic_media_disclosed": synthetic_media,
            "copyright_status": str(
                goal.constraints.get("copyright_status") or "review_required"
            ),
            "policy_risk_flags": risk_flags,
            "experiment": {
                "primary_metrics": [
                    "impressions_ctr",
                    "audience_retention",
                    "returning_viewers",
                    "satisfaction_feedback",
                ],
                "avoid_optimizing_for": ["keyword_stuffing", "fixed_hashtag_quota"],
            },
        },
        "validation": {
            "has_caption": bool(caption),
            "has_media": bool(media_paths),
            "is_platform_eligible": bool(video_paths),
            "is_platform_publish_ready": bool(caption)
            and bool(video_paths)
            and not platform_blockers,
            "is_publish_ready": bool(caption) and bool(media_paths),
            "issues": issues,
            "warnings": risk_flags,
            "title_character_count": len(title),
            "description_character_count": len(description),
            "tag_count": len(tags),
        },
    }


def _build_facebook_package(
    *,
    goal: GoalRequest,
    caption: str,
    hashtags: str,
    media_paths: list[str],
    post_strategy: dict[str, Any],
) -> dict[str, Any]:
    video_paths = _paths_with_extensions(media_paths, VIDEO_EXTENSIONS)
    image_paths = _paths_with_extensions(media_paths, IMAGE_EXTENSIONS)
    is_reel = bool(video_paths)
    format_name = "reel" if is_reel else "native_photo" if image_paths else "text_review"
    facebook_caption = _compact_facebook_caption(caption)
    facebook_hashtags = _limit_hashtags(hashtags, FACEBOOK_HASHTAG_MAX)
    issues: list[str] = []
    if not media_paths:
        issues.append("caption_and_media_required")
    if not video_paths and not image_paths:
        issues.append("facebook_media_format_unsupported")
    warnings: list[str] = []
    if is_reel:
        warnings.append("reel_duration_and_canvas_checked_at_publish")
    risk_flags = _content_risk_flags(facebook_caption, facebook_hashtags)
    warnings.extend(risk_flags)
    platform_blockers = [
        issue
        for issue in issues
        if issue not in {"caption_and_media_required"}
    ]
    return {
        "format": format_name,
        "metadata_source": "derived",
        "caption": facebook_caption,
        "hashtags": facebook_hashtags,
        "character_count": len(facebook_caption),
        "additional_params": {"facebook_use_reels": True} if is_reel else {},
        "content_strategy": {
            "strategy_version": PLATFORM_STRATEGY_VERSION,
            "platform": "facebook",
            "format": format_name,
            "post_strategy_version": str(post_strategy.get("strategy_version") or ""),
            "variant_id": str(post_strategy.get("variant_id") or ""),
            "variation_key": str(post_strategy.get("variation_key") or ""),
            "editorial_question": str(post_strategy.get("editorial_question") or ""),
            "hook": str(post_strategy.get("hook_mode") or ""),
            "payoff": str(post_strategy.get("payoff_mode") or ""),
            "cta_policy": str(post_strategy.get("cta_policy") or ""),
            "hashtag_policy": str(post_strategy.get("hashtag_policy") or ""),
            "discovery_terms": list(post_strategy.get("discovery_terms") or []),
            "originality_basis": str(
                goal.constraints.get("originality_basis") or "creator_produced_or_owned_media"
            ),
            "synthetic_media_disclosed": True,
            "copyright_status": str(
                goal.constraints.get("copyright_status") or "review_required"
            ),
            "policy_risk_flags": risk_flags,
            "experiment": {
                "primary_metrics": [
                    "qualified_views",
                    "average_watch_time",
                    "meaningful_comments",
                    "earnings_rate",
                    "non_qualified_views",
                ],
                "avoid_optimizing_for": ["engagement_bait", "excessive_hashtags", "fixed_hashtag_quota"],
            },
        },
        "validation": {
            "has_caption": bool(facebook_caption),
            "has_media": bool(media_paths),
            "is_platform_eligible": bool(video_paths or image_paths),
            "is_platform_publish_ready": bool(facebook_caption)
            and bool(video_paths or image_paths)
            and not platform_blockers,
            "is_publish_ready": bool(facebook_caption) and bool(media_paths),
            "issues": issues,
            "warnings": warnings,
            "caption_character_count": len(facebook_caption),
            "hashtag_count": len(_hashtag_tokens(facebook_hashtags)),
        },
    }


def _youtube_title(answer_line: str, prompt: str) -> str:
    candidate = _single_line(answer_line)
    if not candidate:
        candidate = _single_line(prompt)
    candidate = re.sub(r"^\s*(?:what|why|how)\s+", "", candidate, flags=re.IGNORECASE)
    return candidate.rstrip(".!?。！？ ").strip()[:YOUTUBE_TITLE_MAX].rstrip()


def _youtube_description(caption: str, answer_line: str, hashtags: str) -> str:
    body = str(caption or "").strip()
    answer = str(answer_line or "").strip()
    if answer and body and not body.startswith(answer):
        body = f"{answer}\n\n{body}"
    parts = [part for part in (body, str(hashtags or "").strip()) if part]
    return "\n\n".join(parts).strip()


def _youtube_tags(hashtags: str) -> list[str]:
    # YouTube documents title, thumbnail, and description as the important
    # discovery metadata; tags are retained only when the creator explicitly
    # selected a semantic tag, rather than mining production prompts.
    candidates = _hashtag_tokens(hashtags)
    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip().replace("_", " ")
        key = normalized.casefold()
        if not normalized or key in _STOP_WORDS or key == "mediaoverload":
            continue
        if len(normalized) < 2 or len(normalized) > 40 or key in seen:
            continue
        seen.add(key)
        tags.append(normalized)
        if len(tags) >= YOUTUBE_TAG_MAX:
            break
    return tags


def _answer_first_line(caption: str) -> str:
    candidates = [
        _single_line(part)
        for part in re.split(r"(?:\r?\n)+|(?<=[.!?。！？])\s*", str(caption or ""))
    ]
    fallback = ""
    for candidate in candidates:
        if not candidate:
            continue
        if not fallback:
            fallback = candidate
        if candidate.endswith(("?", "？")):
            continue
        if re.match(
            r"^(?:like|save|share|follow|comment|which|what do you think)\b",
            candidate,
            flags=re.I,
        ):
            continue
        return candidate
    return fallback


def _compact_facebook_caption(caption: str) -> str:
    text = str(caption or "").strip()
    if len(text) <= FACEBOOK_CAPTION_MAX:
        return text
    paragraphs = [
        part.strip()
        for part in re.split(r"\r?\n\s*\r?\n", text)
        if part.strip()
    ]
    selected: list[str] = []
    for paragraph in paragraphs:
        proposed = "\n\n".join([*selected, paragraph])
        if len(proposed) <= FACEBOOK_CAPTION_MAX:
            selected.append(paragraph)
            continue
        break
    compact = "\n\n".join(selected).strip()
    if not compact:
        compact = text[:FACEBOOK_CAPTION_MAX].rsplit(" ", 1)[0].strip()
    return compact[:FACEBOOK_CAPTION_MAX].rstrip()


def _limit_hashtags(hashtags: str, limit: int) -> str:
    return " ".join(f"#{token}" for token in _hashtag_tokens(hashtags)[:limit])


def sanitize_hashtags(value: str) -> str:
    """Keep only public, deduplicated hashtags from a human or model draft."""
    return " ".join(
        f"#{token}"
        for token in _hashtag_tokens(value)
        if token.casefold() != "mediaoverload"
    )


def _hashtag_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"#([^\s#]+)", str(value or "")):
        cleaned = token.strip(
            ".,!?;:()[]{}\\\"'，。！？；：（）【】"
        )
        if cleaned and cleaned.casefold() not in {item.casefold() for item in tokens}:
            tokens.append(cleaned)
    return tokens


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("#", "")).strip()


def _search_intent(prompt: str) -> str:
    return _single_line(prompt)[:160]


def _paths_with_extensions(paths: list[str], extensions: set[str]) -> list[str]:
    return [path for path in paths if Path(path).suffix.casefold() in extensions]


def _content_risk_flags(caption: str, hashtags: str) -> list[str]:
    flags: list[str] = []
    if "#mediaoverload" in str(caption).casefold() or "#mediaoverload" in str(hashtags).casefold():
        flags.append("internal_project_hashtag")
    if len(_hashtag_tokens(hashtags)) > 5:
        flags.append("excessive_hashtags")
    if re.search(r"(?:\n\s*){4,}", str(caption)):
        flags.append("excessive_caption_spacing")
    return flags
