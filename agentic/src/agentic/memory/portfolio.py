from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PortfolioRecord:
    goal_signature: str
    workflow_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class PortfolioMemory:
    """
    Lightweight portfolio memory that persists successful (or failed) runs.
    For now it is file-based JSONL so it remains simple and inspectable.
    """

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("", encoding="utf-8")

    def append(self, record: PortfolioRecord) -> None:
        entry = {
            "goal_signature": record.goal_signature,
            "workflow_name": record.workflow_name,
            "status": record.status,
            "metrics": record.metrics,
            "notes": record.notes,
        }
        with self.storage_path.open("a", encoding="utf-8") as fp:
            fp.write(f"{json.dumps(entry)}\n")

    def load_recent(self, limit: int = 20) -> list[PortfolioRecord]:
        if not self.storage_path.exists():
            return []
        records: list[PortfolioRecord] = []
        with self.storage_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(
                    PortfolioRecord(
                        goal_signature=data["goal_signature"],
                        workflow_name=data["workflow_name"],
                        status=data["status"],
                        metrics=data.get("metrics", {}),
                        notes=data.get("notes", []),
                    )
                )
        return records[-limit:]
