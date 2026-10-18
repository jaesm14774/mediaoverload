"""Pure checks for news selections used by creative runtime components."""

from __future__ import annotations


DEFAULT_EXCLUDE_TITLE_TERMS = (
    "性愛",
    "色情",
    "情色",
    "裸",
    "性侵",
    "強姦",
    "自殺",
    "死亡",
    "猝逝",
    "身亡",
    "命案",
    "凶殺",
    "血腥",
    "屍體",
)


def is_usable_news_selection(title: str, keyword: str) -> bool:
    title_text = str(title or "").strip()
    keyword_text = str(keyword or "").strip()
    if not title_text or not keyword_text:
        return False
    if title_text.casefold() in {"...", "…", "n/a", "na", "null", "unknown", "untitled"}:
        return False
    return any(character.isalnum() for character in title_text)


def is_brand_safe_news_selection(title: str, keyword: str) -> bool:
    text = f"{title} {keyword}".casefold()
    return not any(term.casefold() in text for term in DEFAULT_EXCLUDE_TITLE_TERMS)
