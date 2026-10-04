from __future__ import annotations

import random
from typing import Any


def select_weighted_route(
    weights: dict[str, Any],
    candidates: list[str],
    *,
    rng: random.Random | None = None,
) -> str:
    """Select a candidate randomly in proportion to its configured weight."""
    normalized_candidates = [str(candidate).strip() for candidate in candidates if str(candidate).strip()]
    if not normalized_candidates:
        return "text2longvideo"

    normalized_weights: dict[str, float] = {}
    for candidate in normalized_candidates:
        try:
            weight = float(weights.get(candidate, 0))
        except (TypeError, ValueError):
            weight = 0.0
        if weight > 0:
            normalized_weights[candidate] = weight
    if not normalized_weights:
        normalized_weights = {candidate: 1.0 for candidate in normalized_candidates}

    return _weighted_choice(normalized_weights, rng or random)


def _weighted_choice(weights: dict[str, float], chooser: Any) -> str:
    total = sum(weights.values())
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if threshold <= cumulative:
            return name
    return next(reversed(weights))
