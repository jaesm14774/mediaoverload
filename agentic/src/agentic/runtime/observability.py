from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert runtime values into stable JSON without leaking object repr noise."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            pass
    return str(value)


def _safe_slug(value: str, *, fallback: str = "record") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.")
    return slug[:100] or fallback


class RunRecorder:
    """Persist one run's decisions, LLM calls, node outputs, and final manifest."""

    def __init__(self, runs_root: Path, run_id: str) -> None:
        self.run_id = _safe_slug(str(run_id), fallback="run")
        self.run_dir = Path(runs_root) / self.run_id
        self.llm_dir = self.run_dir / "llm"
        self.node_dir = self.run_dir / "nodes"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self.node_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self._lock = threading.Lock()
        self._sequence = 0
        self.record_event("run.created")

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def record_event(self, event: str, **payload: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": str(event),
            "payload": _json_safe(payload),
        }
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def record_llm_call(
        self,
        *,
        schema_name: str,
        attempt: int,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        response: Any = None,
        parsed_payload: Any = None,
        error: str = "",
        model: str = "text",
        model_id: str = "",
        images: list[str] | None = None,
        response_format_used: bool = True,
    ) -> Path:
        call_path = self.start_llm_call(
            schema_name=schema_name,
            attempt=attempt,
            messages=messages,
            schema=schema,
            model=model,
            model_id=model_id,
            images=images,
            response_format_used=response_format_used,
        )
        self.complete_llm_call(
            call_path,
            response=response,
            parsed_payload=parsed_payload,
            error=error,
        )
        return call_path

    def start_llm_call(
        self,
        *,
        schema_name: str,
        attempt: int,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        model: str = "text",
        model_id: str = "",
        images: list[str] | None = None,
        response_format_used: bool = True,
    ) -> Path:
        sequence = self._next_sequence()
        call_id = f"{sequence:04d}"
        call_path = self.llm_dir / f"{call_id}_{_safe_slug(schema_name)}.json"
        record = {
            "run_id": self.run_id,
            "call_id": call_id,
            "schema_name": schema_name,
            "attempt": int(attempt),
            "model_role": model,
            "model_id": str(model_id or ""),
            "status": "pending",
            "response_format_used": bool(response_format_used),
            "images": images or [],
            "messages": messages,
            "schema": schema,
            "raw_response": None,
            "parsed_payload": None,
            "error": "",
        }
        call_path.write_text(
            json.dumps(_json_safe(record), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.record_event(
            "llm.call",
            call_id=call_id,
            schema_name=schema_name,
            attempt=int(attempt),
            status="pending",
            path=str(call_path),
        )
        return call_path

    def complete_llm_call(
        self,
        call_path: Path,
        *,
        response: Any = None,
        parsed_payload: Any = None,
        error: str = "",
        model_id: str | None = None,
    ) -> None:
        record = json.loads(call_path.read_text(encoding="utf-8"))
        record.update(
            {
                "status": "success" if not error else "failed",
                "raw_response": _json_safe(response),
                "parsed_payload": _json_safe(parsed_payload),
                "error": str(error or ""),
            }
        )
        if model_id is not None:
            record["model_id"] = str(model_id or "")
        call_path.write_text(
            json.dumps(_json_safe(record), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.record_event(
            "llm.call.completed",
            call_id=record.get("call_id", ""),
            schema_name=record.get("schema_name", ""),
            attempt=record.get("attempt", 0),
            status=record.get("status", ""),
            path=str(call_path),
            error=str(error or ""),
        )

    def record_node(
        self,
        *,
        node_id: str,
        skill_name: str,
        status: str,
        attempt: int,
        outputs: dict[str, Any],
        metrics: dict[str, Any],
        logs: list[str],
    ) -> Path:
        node_path = self.node_dir / f"{_safe_slug(node_id)}_attempt_{int(attempt)}.json"
        record = {
            "run_id": self.run_id,
            "node_id": node_id,
            "skill_name": skill_name,
            "status": status,
            "attempt": int(attempt),
            "outputs": outputs,
            "metrics": metrics,
            "logs": logs,
        }
        node_path.write_text(
            json.dumps(_json_safe(record), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.record_event(
            "node.completed",
            node_id=node_id,
            skill_name=skill_name,
            status=status,
            attempt=int(attempt),
            path=str(node_path),
        )
        return node_path

    def record_workflow_result(self, workflow_name: str, result: dict[str, Any]) -> Path:
        result_path = self.run_dir / f"workflow_{_safe_slug(workflow_name)}.json"
        result_path.write_text(
            json.dumps(_json_safe(result), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.record_event(
            "workflow.completed",
            workflow_name=workflow_name,
            status=result.get("status", ""),
            path=str(result_path),
        )
        return result_path

    def finalize(self, payload: dict[str, Any]) -> Path:
        manifest = {
            "run_id": self.run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **_json_safe(payload),
            "observability": {
                "run_dir": str(self.run_dir),
                "events_path": str(self.events_path),
                "llm_dir": str(self.llm_dir),
                "node_dir": str(self.node_dir),
            },
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.record_event("run.finalized", manifest_path=str(self.manifest_path), status=payload.get("status", ""))
        return self.manifest_path
