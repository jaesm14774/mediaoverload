from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def select_weighted_route(
    weights: dict[str, Any],
    candidates: list[str],
    *,
    rng: random.Random | None = None,
    state_path: Path | None = None,
    diversity_config: dict[str, Any] | None = None,
) -> str:
    """Select a route while keeping weighted randomness diverse across runs.

    When ``state_path`` is provided, integer-like weights are materialized as a
    persisted shuffle bag.  Each route therefore appears the configured
    number of times per bag before the bag is refilled, instead of allowing a
    short scheduler window to repeat the same route by chance.
    """
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

    chooser = rng or random
    config = dict(diversity_config or {})
    if state_path is None or config.get("enabled", True) is False:
        return _weighted_choice(normalized_weights, chooser)

    try:
        state = _load_state(state_path)
        signature = _bag_signature(normalized_weights)
        bag = [item for item in state.get("bag", []) if item in normalized_weights]
        if state.get("signature") != signature or not bag:
            bag = _build_bag(normalized_weights)
            chooser.shuffle(bag)
        selected = bag.pop()
        history = [item for item in state.get("history", []) if item in normalized_weights]
        history.append(selected)
        window = max(1, int(config.get("history_window", 24) or 24))
        _write_state(
            state_path,
            {
                "version": 1,
                "signature": signature,
                "bag": bag,
                "history": history[-window:],
            },
        )
        return selected
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # Routing must remain available if the optional diversity state is not
        # writable in a container or is interrupted during replacement.
        return _weighted_choice(normalized_weights, chooser)


def _weighted_choice(weights: dict[str, float], chooser: Any) -> str:
    total = sum(weights.values())
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if threshold <= cumulative:
            return name
    return next(reversed(weights))


def _bag_signature(weights: dict[str, float]) -> list[str]:
    return [f"{name}={weight:g}" for name, weight in sorted(weights.items())]


def _build_bag(weights: dict[str, float]) -> list[str]:
    bag: list[str] = []
    for name, weight in weights.items():
        repetitions = max(1, int(round(weight)))
        bag.extend([name] * repetitions)
    return bag


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)
