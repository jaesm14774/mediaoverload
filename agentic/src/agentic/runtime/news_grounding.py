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
    "死傷",
    "傷亡",
    "罹難",
    "遇難",
    "遇害",
    "殉職",
    "墜樓",
    "墜機",
    "槍擊",
    "爆炸",
    "縱火",
    "重大車禍",
    "釀死",
    "致死",
    "兒虐",
    "虐童",
    "家暴",
    "性騷擾",
    "戰爭",
    "戰火",
    "恐攻",
    "恐怖攻擊",
    "虐殺",
    "強暴",
    "罷工",
    "工會",
    "勞資糾紛",
    "集體協商",
    "抗議",
    "抗爭",
    "示威",
    "鎮壓",
    "選舉",
    "候選人",
    "政黨",
    "公投",
    "罷免",
    "立委",
    "總統",
    "政治人物",
    "政治獻金",
    "造勢",
    "踩踏",
    "梨泰院",
)


def is_usable_news_selection(title: str, keyword: str) -> bool:
    """A valid headline stays usable when keyword extraction is missing."""
    title_text = str(title or "").strip()
    if not title_text:
        return False
    if title_text.casefold() in {"...", "…", "n/a", "na", "null", "unknown", "untitled"}:
        return False
    return any(character.isalnum() for character in title_text)


def is_brand_safe_news_selection(title: str, keyword: str) -> bool:
    text = f"{title} {keyword}".casefold()
    return not any(term.casefold() in text for term in DEFAULT_EXCLUDE_TITLE_TERMS)
