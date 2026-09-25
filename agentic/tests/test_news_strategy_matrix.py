from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_news_strategy_matrix import (
    _checked_matrix_path,
    _make_prompt_trace,
    _safe_terminal_text,
)


class NewsStrategyMatrixEvidenceTests(unittest.TestCase):
    def test_model_summary_path_outside_run_media_is_ignored(self) -> None:
        """User Given node output contains a model-influenced summary path outside the run, When prompt evidence is collected, Then the external file is not read or copied into the trace."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "strategy" / "run_01"
            trace_dir = run_dir / "trace"
            nodes_dir = trace_dir / "nodes"
            nodes_dir.mkdir(parents=True)
            outside_summary = root / "outside_summary.json"
            outside_summary.write_text(json.dumps({"payload": {"secret": "must stay out"}}), encoding="utf-8")
            (nodes_dir / "native-story_attempt_1.json").write_text(
                json.dumps({"node_id": "native-story", "outputs": {"summary_path": str(outside_summary)}}),
                encoding="utf-8",
            )

            result = _make_prompt_trace(trace_dir, run_dir)

            self.assertEqual(result["comfy_payloads"], [])

    def test_matrix_path_rejects_escape(self) -> None:
        """User Given an evidence path resolves outside its matrix root, When the runner validates it, Then the path is rejected before file access."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "matrix"
            root.mkdir()
            outside = Path(temporary_directory) / "outside.json"

            with self.assertRaises(RuntimeError):
                _checked_matrix_path(root, outside)

    def test_progress_text_escapes_terminal_and_bidi_controls(self) -> None:
        """User Given a news title contains line-break and bidi controls, When progress is printed, Then control characters are rendered as escaped text."""
        self.assertEqual(
            _safe_terminal_text("title\n\u202e status"),
            "title\\u000a\\u202e status",
        )


if __name__ == "__main__":
    unittest.main()
