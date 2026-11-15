"""How a required pair of characters should meet inside one picture or clip.

Every prompt writer that receives a two-character cast shares this one rule,
so a pair is written as two natures answering each other instead of two names
placed in one frame. Single-subject requests never see it.
"""

from __future__ import annotations

from typing import Any

PAIR_INTERACTION_CRAFT = (
    "Pair interaction craft (this brief explicitly requests co-leads): the two cast members are co-leads of one "
    "shared beat, and the viewer is moved by "
    "how each one acts according to its own nature. First decide each one's nature: its personality in its own "
    "story or, for an animal, how that species really behaves, shown through one feature from its listed "
    "description such as a body part, an ability, or a carried item. Never swap natures and never give both the "
    "same reaction. Build the beat from the contrast between them (size, temperament, ability, pace): one acts "
    "out of its nature, the other answers out of its own, and the answer changes what the first does next, so "
    "each action causes the other's reaction. When both are the same character, they share one nature and the "
    "contrast is the moment: one initiates and the other answers, each drawn at a different point of the same "
    "feeling. Make the peak one small, physical, specific moment that only these "
    "two would have, such as a touch, a shared effort, a held look, a rescue, or a forgiven clumsiness, placed "
    "where both faces read. Match the feeling to the occasion: warmth or wonder for light news, quiet "
    "company for serious news, never a gag at someone's harm. Never pose the two side by side, both facing the "
    "viewer, or doing the same thing in parallel; at every beat each is visibly aware of the other and "
    "responding to it. The occasion comes from the source; the two natures decide how they meet it."
)


def required_pair_names(subject_context: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return the two cast names when the context requires a visible interaction."""

    context = subject_context if isinstance(subject_context, dict) else {}
    contract = context.get("interaction_contract")
    if not (isinstance(contract, dict) and contract.get("required")):
        return None
    names = [
        str(subject.get("name") or "").strip()
        for subject in context.get("subjects") or []
        if isinstance(subject, dict) and str(subject.get("name") or "").strip()
    ]
    return (names[0], names[1]) if len(names) == 2 else None


def pair_interaction_craft(subject_context: dict[str, Any] | None) -> str:
    """The shared craft rule for a required pair, or an empty string otherwise."""

    return PAIR_INTERACTION_CRAFT if required_pair_names(subject_context) else ""
