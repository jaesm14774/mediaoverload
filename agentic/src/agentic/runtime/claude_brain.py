"""A chat model whose answers come from an operator (Claude Code) through files.

Every LLM call of a run is written to ``<dir>/NNNN.request.{md,json}`` and the
call blocks until ``<dir>/NNNN.response.txt`` appears. The pipeline around the
model is untouched, so a run served this way differs from an OpenRouter run
only in who wrote the answers.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agentic.runtime.model_backends import MAX_PROVIDER_RETRIES, ProviderRequestError

BRAIN_DIR_ENV = "AGENTIC_CLAUDE_BRAIN_DIR"
BRAIN_WAIT_ENV = "AGENTIC_CLAUDE_BRAIN_WAIT_SECONDS"
BRAIN_MODEL_NAME = "claude-code-brain"
DEFAULT_MAX_ATTEMPTS = 3
RESPONSE_SETTLE_SECONDS = 0.3
POLL_SECONDS = 0.25
FINISHED_FILE = "finished.json"

_REQUEST_NAME = re.compile(r"^(\d{4})\.request\.json$")


def request_path(directory: Path, seq: int) -> Path:
    return directory / f"{seq:04d}.request.json"


def response_path(directory: Path, seq: int) -> Path:
    return directory / f"{seq:04d}.response.txt"


def pending_sequences(directory: Path) -> list[int]:
    """Requests that still wait for an answer, oldest first."""
    if not directory.is_dir():
        return []
    sequences = sorted(
        int(match.group(1))
        for entry in directory.iterdir()
        if (match := _REQUEST_NAME.match(entry.name))
    )
    return [seq for seq in sequences if not response_path(directory, seq).exists()]


def render_request_markdown(request: dict[str, Any]) -> str:
    parts = [
        f"# LLM request {request['seq']:04d} ({request['modality']})",
        f"- created_at: {request['created_at']}",
        f"- attempt: {request['attempt']} of {request['max_attempts']}"
        " (a retry re-sends the identical prompt; the pipeline gives no reason, like a provider retry)",
    ]
    if request["images"]:
        parts.append("- images (read them before answering):")
        parts.extend(f"  - {image}" for image in request["images"])
    for message in request["messages"]:
        parts.append(f"\n## {message['role']}\n\n{message['content']}")
    if request["response_schema"]:
        parts.append(
            "\n## response schema (informational; answer text must be the raw value)\n\n"
            + json.dumps(request["response_schema"], ensure_ascii=False, indent=2)
        )
    return "\n".join(parts) + "\n"


def _write_request(directory: Path, request: dict[str, Any]) -> None:
    seq = request["seq"]
    (directory / f"{seq:04d}.request.md").write_text(
        render_request_markdown(request), encoding="utf-8"
    )
    path = request_path(directory, seq)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _claim_sequence(directory: Path) -> int:
    """Reserve the next sequence number; safe across threads and processes."""
    existing = [
        int(match.group(1))
        for entry in directory.iterdir()
        if (match := _REQUEST_NAME.match(entry.name))
    ]
    seq = max(existing, default=0) + 1
    while True:
        try:
            descriptor = os.open(
                directory / f"{seq:04d}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            seq += 1
            continue
        os.close(descriptor)
        return seq


def _mark_finished(directory: Path) -> None:
    if not directory.is_dir():
        return
    (directory / FINISHED_FILE).write_text(
        json.dumps({"finished_at": datetime.now().isoformat(timespec="seconds")}),
        encoding="utf-8",
    )


class ClaudeBrainModel:
    """Implements the ``ChatModel`` protocol by asking the operator."""

    def __init__(self, directory: str | Path, modality: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / FINISHED_FILE).unlink(missing_ok=True)
        atexit.register(_mark_finished, self.directory)
        self.modality = modality
        self.last_success_model = ""
        self.last_attempt_model = BRAIN_MODEL_NAME
        self.last_attempts: list[dict[str, Any]] = []
        raw_wait = os.environ.get(BRAIN_WAIT_ENV, "3600").strip()
        self.wait_seconds = float(raw_wait) if raw_wait else 3600.0

    def chat_completion(
        self,
        messages: list[dict],
        images: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        validator: Callable[[str], Any] | None = kwargs.get("_response_validator")
        response_format = kwargs.get("response_format") or {}
        requested = kwargs.get("max_retries")
        max_attempts = min(
            MAX_PROVIDER_RETRIES,
            max(1, int(requested)) if requested is not None else DEFAULT_MAX_ATTEMPTS,
        )
        seq = _claim_sequence(self.directory)
        request = {
            "seq": seq,
            "modality": self.modality,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": messages,
            "images": [str(image) for image in images or []],
            "response_schema": (response_format.get("json_schema") or {}).get("schema"),
            "attempt": 1,
            "max_attempts": max_attempts,
        }
        _write_request(self.directory, request)
        (self.directory / f"{seq:04d}.claim").unlink(missing_ok=True)
        self.last_attempts = []
        while True:
            started_at = time.monotonic()
            text = self._wait_for_response(seq)
            failure: Exception | None = None
            if validator is not None:
                try:
                    validator(text)
                except Exception as exc:
                    failure = exc
            self.last_attempts.append(
                {
                    "provider": "claude_brain",
                    "model": BRAIN_MODEL_NAME,
                    "duration_seconds": round(time.monotonic() - started_at, 3),
                    "status": "failed" if failure else "success",
                    **({"error_type": type(failure).__name__} if failure else {}),
                }
            )
            if failure is None:
                self.last_success_model = BRAIN_MODEL_NAME
                return text
            attempt = request["attempt"]
            response_path(self.directory, seq).rename(
                self.directory / f"{seq:04d}.response.rejected-{attempt}.txt"
            )
            (self.directory / f"{seq:04d}.rejection-{attempt}.txt").write_text(
                f"{type(failure).__name__}: {failure}\n", encoding="utf-8"
            )
            if attempt >= max_attempts:
                raise ProviderRequestError(
                    f"{BRAIN_MODEL_NAME} answer {seq:04d} failed validation "
                    f"{max_attempts} times: {type(failure).__name__}: {failure}"
                ) from failure
            request["attempt"] = attempt + 1
            _write_request(self.directory, request)

    def _wait_for_response(self, seq: int) -> str:
        path = response_path(self.directory, seq)
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            if path.exists() and time.time() - path.stat().st_mtime >= RESPONSE_SETTLE_SECONDS:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
            time.sleep(POLL_SECONDS)
        raise ProviderRequestError(
            f"{BRAIN_MODEL_NAME}: no answer for request {seq:04d} within {self.wait_seconds:.0f}s"
        )
