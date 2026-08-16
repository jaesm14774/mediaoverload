from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEvent:
    node_id: str
    skill_name: str
    status: str
    attempt: int = 1
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class RunMemory:
    """In-memory buffer that tracks node activity for the current run."""

    def __init__(self, max_events: int = 200) -> None:
        self.max_events = max_events
        self._events: deque[MemoryEvent] = deque(maxlen=max_events)

    def record(self, event: MemoryEvent) -> None:
        self._events.append(event)

    def as_serializable(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": event.node_id,
                "skill_name": event.skill_name,
                "status": event.status,
                "attempt": event.attempt,
                "outputs": event.outputs,
                "metrics": event.metrics,
                "logs": event.logs,
            }
            for event in self._events
        ]

    def last(self) -> MemoryEvent | None:
        return self._events[-1] if self._events else None
