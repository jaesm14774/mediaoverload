"""Pure checks for news selections used by creative runtime components."""

from __future__ import annotations


class UnsuitableNewsError(ValueError):
    """The scene writer found nothing in the article a cute character can gently live through."""


def is_usable_news_selection(title: str, keyword: str) -> bool:
    """A valid headline stays usable when keyword extraction is missing."""
    title_text = str(title or "").strip()
    if not title_text:
        return False
    if title_text.casefold() in {"...", "…", "n/a", "na", "null", "unknown", "untitled"}:
        return False
    return any(character.isalnum() for character in title_text)

