"""User scenarios for running the pipeline with Claude Code as the LLM."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.runtime.claude_brain import (
    BRAIN_DIR_ENV,
    FINISHED_FILE,
    pending_sequences,
    response_path,
)
from agentic.runtime.llm_client import PromptClient

SCHEMA = {
    "type": "object",
    "properties": {"caption": {"type": "string"}},
    "required": ["caption"],
}


def _operator(directory: Path, answers: list[str], served: list[Path]) -> threading.Thread:
    """The person (Claude Code) on the other side of the files."""

    def serve() -> None:
        remaining = list(answers)
        deadline = time.monotonic() + 20
        while remaining and time.monotonic() < deadline:
            pending = pending_sequences(directory)
            if pending:
                seq = pending[0]
                served.append(directory / f"{seq:04d}.request.md")
                response_path(directory, seq).write_text(remaining.pop(0), encoding="utf-8")
            time.sleep(0.05)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


class ClaudeBrainScenarios(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name) / "brain"
        self.addCleanup(self._tmp.cleanup)
        env = patch.dict(os.environ, {BRAIN_DIR_ENV: str(self.directory)})
        env.start()
        self.addCleanup(env.stop)

    def _ask(self) -> dict:
        client = PromptClient(mode="llm")
        return client.generate_json(
            client.require_manager(),
            "You write captions.",
            "Caption this news: typhoon day off.",
            "caption_payload",
            SCHEMA,
        )

    def test_user_gets_the_operators_answer_and_the_full_prompt_is_kept(self) -> None:
        """User: Given a run served by Claude, When a step needs the LLM,
        Then the step receives Claude's answer and the whole prompt stays on disk."""
        served: list[Path] = []
        operator = _operator(self.directory, ['{"caption": "Kirby naps in the rain"}'], served)

        result = self._ask()
        operator.join(timeout=5)

        self.assertEqual(result["caption"], "Kirby naps in the rain")
        prompt_on_disk = served[0].read_text(encoding="utf-8")
        self.assertIn("You write captions.", prompt_on_disk)
        self.assertIn("typhoon day off", prompt_on_disk)

    def test_user_sees_an_invalid_answer_rejected_and_asked_again(self) -> None:
        """User: Given Claude answers in prose instead of JSON, When the pipeline validates it,
        Then Claude is asked again, as a provider retry would, and the valid second answer is used."""
        served: list[Path] = []
        operator = _operator(
            self.directory,
            ['Sure! Here is the caption you asked for.', '{"caption": "fixed"}'],
            served,
        )

        result = self._ask()
        operator.join(timeout=5)

        self.assertEqual(result["caption"], "fixed")
        self.assertEqual(len(served), 2)
        self.assertIn("attempt: 2 of", served[1].read_text(encoding="utf-8"))
        self.assertNotIn("Sure! Here is", served[1].read_text(encoding="utf-8"))

    def test_operator_can_tell_the_run_finished(self) -> None:
        """User: Given a served run, When the process ends, Then the operator is told it is over."""
        served: list[Path] = []
        operator = _operator(self.directory, ['{"caption": "x"}'], served)
        self._ask()
        operator.join(timeout=5)

        from agentic.runtime.claude_brain import _mark_finished

        _mark_finished(self.directory)
        self.assertTrue(json.loads((self.directory / FINISHED_FILE).read_text("utf-8")))
        self.assertEqual(pending_sequences(self.directory), [])


if __name__ == "__main__":
    unittest.main()
