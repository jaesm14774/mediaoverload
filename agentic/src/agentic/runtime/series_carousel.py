"""Series carousel: one shared world, one page per character of the same group.

The scene fixes the medium, place, light, and weather once. Every page then adds
its own moment (what that member does, drawn from the happening) and its own
spot in the frame, and leads with that member's roster description word for
word. Pages share a world, never an action or a composition.
"""

from __future__ import annotations

from collections import Counter
import random
import re
from typing import Any, Iterable

from agentic.runtime.prompting import visual_identity_phrase

# Where a page puts its character in the frame: a three-by-three grid.
PLACEMENTS = (
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
)
# Instagram and Threads carousels hold two to ten items.
SERIES_PAGE_RANGE = (2, 10)


def _named_members(text: str, members: list[dict[str, Any]]) -> set[str]:
    """Cast names written in the text; the longest name wins ("Parasol Waddle Dee" over "Waddle Dee")."""
    names = sorted((str(member["name"]) for member in members), key=len, reverse=True)
    pattern = "|".join(re.escape(name) for name in names)
    found = re.findall(rf"(?<!\w)(?:{pattern})(?!\w)", text, flags=re.IGNORECASE)
    return {name.casefold() for name in found}


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


def _validate_placements(pages: list[dict[str, Any]]) -> None:
    unknown = [page["placement"] for page in pages if page["placement"] not in PLACEMENTS]
    if unknown:
        raise ValueError(f"Series placement must be one of {', '.join(PLACEMENTS)}; got {unknown[0]!r}")
    # At most one page dead centre; any other spot holds at most half the set, rounded up.
    for placement, count in Counter(page["placement"] for page in pages).items():
        limit = 1 if placement == "center" else -(-len(pages) // 2)
        if count > limit:
            raise ValueError(
                f"Series pages may not stack in one spot: {count} pages at {placement}, limit {limit}"
            )


def series_prompt_sets(
    scene: str, pages: list[dict[str, Any]], members: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build one prompt per member from the shared scene and that member's own moment.

    Rejects a scene that names a member, a cast that is not covered page for
    page, a moment that is not about its own member alone, and placements that
    stack the set into one composition.
    """
    scene = " ".join(str(scene or "").split())
    if not scene:
        raise ValueError("Series scene is empty")
    if "{" in scene or any("{" in str(page.get("moment", "")) for page in pages):
        raise ValueError("Series scene and moments must not contain a template slot")
    if _named_members(scene, members):
        raise ValueError("Series scene must not name a cast member; each page names its own")
    if [str(page.get("name", "")).casefold() for page in pages] != [
        str(member["name"]).casefold() for member in members
    ]:
        raise ValueError("Series pages must cover the cast in order, one page per member")
    _validate_placements(pages)
    prompt_sets: list[dict[str, Any]] = []
    for member, page in zip(members, pages, strict=True):
        name = str(member["name"])
        description = " ".join(str(member.get("role_description") or "").split())
        if not description:
            raise ValueError(f"Series cast member has no roster description: {name}")
        moment = " ".join(str(page.get("moment") or "").split())
        if _named_members(moment, members) != {name.casefold()}:
            raise ValueError(f"Series moment must be about {name} alone: {moment!r}")
        identity = visual_identity_phrase(f"{name}: {description}").rstrip(". ")
        placement = page["placement"]
        prompt_sets.append(
            {
                "label": name,
                "character": name,
                "moment": moment,
                "placement": placement,
                "prompt": (
                    f"{identity}; {scene} {moment} "
                    f"{name} is placed at the {placement.replace('-', ' ')} of the frame."
                ),
                "self_contained": True,
            }
        )
    return prompt_sets
