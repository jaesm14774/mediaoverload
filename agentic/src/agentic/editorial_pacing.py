from __future__ import annotations


EDITORIAL_PACING_SPEEDS = {"fast": 1.5, "normal": 1.0, "slow": 0.8}


def motion_pacing_labels(scores: list[float]) -> list[str]:
    """Map relative motion energy to editorial beats without changing uniform clips."""
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    spread = high - low
    if high < 0.02 or spread < max(0.015, high * 0.2):
        return ["normal"] * len(scores)

    labels: list[str] = []
    for score in scores:
        relative_motion = (score - low) / spread
        if relative_motion <= 0.25:
            labels.append("slow")
        elif relative_motion >= 0.75:
            labels.append("fast")
        else:
            labels.append("normal")
    return labels
