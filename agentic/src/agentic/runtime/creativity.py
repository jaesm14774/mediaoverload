from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic.runtime.contracts import GoalRequest


class IdeaDirector:
    """Produces lightweight idea variations for the planner."""

    def generate_variations(self, goal: GoalRequest) -> list[dict[str, Any]]:
        base_style = goal.style
        palette = [
            base_style,
            f"{base_style} high-contrast montage",
            f"{base_style} slow cinematic sweep",
        ]
        return [
            {
                "style": style_variant,
                "duration": goal.duration_seconds,
                "media_type": goal.media_type,
                "notes": "auto-generated variation",
            }
            for style_variant in palette
        ]


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 2
    retry_statuses: tuple[str, ...] = ("retry_needed", "transient_error")

    def should_retry(self, status: str, attempt: int) -> bool:
        return status in self.retry_statuses and attempt < self.max_attempts
