"""Series carousel: one fixed scene, one page per character of the same group.

The scene is written once with a single ``{character}`` slot. Each page fills
the slot with a different cast member's name and a short look, so the pages
differ in nothing else and read as one set.
"""

from __future__ import annotations

import random
import re
from typing import Any, Iterable

SERIES_SLOT = "{character}"
# Instagram and Threads carousels hold two to ten items.
SERIES_PAGE_RANGE = (2, 10)
# A look is swapped into the slot of a fixed composition; a long one changes the picture.
SERIES_LOOK_MAX_WORDS = 12


def _names_member(text: str, name: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE))


def select_series_members(
    candidates: Iterable[dict[str, Any]],
    *,
    primary: str,
    page_count: int,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Cast the primary character first, then distinct group members by weight."""
    low, high = SERIES_PAGE_RANGE
    if not low <= page_count <= high:
        raise ValueError(f"Series carousel page count must be {low}-{high}, got {page_count}")
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.get("name", "")).strip().casefold()
        if (
            key
            and key not in seen
            and int(candidate.get("status", 1)) == 1
            and float(candidate.get("weight", 1.0)) > 0
        ):
            seen.add(key)
            pool.append(dict(candidate))
    lead = next(
        (c for c in pool if str(c.get("name", "")).casefold() == primary.strip().casefold()),
        None,
    )
    if lead is None:
        raise ValueError(f"Series carousel primary '{primary}' is not an active group member")
    if len(pool) < page_count:
        raise ValueError(
            f"Series carousel needs {page_count} active group members but the group has {len(pool)}"
        )
    chooser = rng or random
    rest = [c for c in pool if c is not lead]
    cast = [lead]
    while len(cast) < page_count:
        pick = chooser.choices(rest, weights=[float(c.get("weight", 1.0)) for c in rest])[0]
        rest.remove(pick)
        cast.append(pick)
    return [
        {
            "name": str(member["name"]).strip(),
            "role_description": str(member.get("role_description") or ""),
            "keywords": str(member.get("keywords") or ""),
        }
        for member in cast
    ]


def series_prompt_sets(
    template: str,
    looks: Iterable[dict[str, Any]],
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill the one slot per member; reject a scene that cannot be shared."""
    scene = str(template or "").strip()
    if scene.count(SERIES_SLOT) != 1:
        raise ValueError(f"Series scene must contain {SERIES_SLOT} exactly once")
    names = [str(member["name"]) for member in members]
    for name in names:
        if _names_member(scene, name):
            raise ValueError(f"Series scene must not name a cast member: {name}")
    look_by_name = {
        str(item.get("name", "")).strip().casefold(): " ".join(str(item.get("look", "")).split())
        for item in looks
    }
    prompt_sets: list[dict[str, Any]] = []
    for name in names:
        look = look_by_name.get(name.casefold(), "")
        if not look:
            raise ValueError(f"Series scene has no look for cast member: {name}")
        if len(look.split()) > SERIES_LOOK_MAX_WORDS:
            raise ValueError(
                f"Series look for {name} is longer than {SERIES_LOOK_MAX_WORDS} words"
            )
        if "{" in look or "}" in look:
            raise ValueError(f"Series look for {name} must not contain a slot")
        if any(_names_member(look, other) for other in names if other != name):
            raise ValueError(f"Series look for {name} must not name another cast member")
        prompt_sets.append(
            {
                "label": name,
                "character": name,
                "prompt": scene.replace(SERIES_SLOT, f"{name}, {look}"),
                "self_contained": True,
            }
        )
    return prompt_sets
