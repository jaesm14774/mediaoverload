from __future__ import annotations

import unittest

from agentic.app.main import _build_prompt_summary


class AppMainTests(unittest.TestCase):
    def test_build_prompt_summary_returns_prompt_metadata_only(self) -> None:
        summary = _build_prompt_summary(
            {
                "node_prompt_modes": {"idea-brief": "llm"},
                "prompt_lineage": [{"node_id": "review-refine-prompt", "revised_prompt": "better prompt"}],
                "node_outputs": {"idea-brief": {"prompt": "raw"}},
            }
        )

        self.assertEqual(summary["node_prompt_modes"]["idea-brief"], "llm")
        self.assertEqual(summary["prompt_lineage"][0]["node_id"], "review-refine-prompt")
        self.assertNotIn("node_outputs", summary)

    def test_build_prompt_summary_surfaces_fallback_nodes_and_backend_info(self) -> None:
        summary = _build_prompt_summary(
            {
                "prompt_lineage": [
                    {
                        "node_id": "idea-brief",
                        "fallback_reason": "manager_unavailable",
                        "llm_backend": {"text_provider": "gemini", "text_model": "gemini-flash-lite-latest"},
                    }
                ]
            }
        )

        self.assertEqual(summary["fallback_nodes"]["idea-brief"], "manager_unavailable")
        self.assertEqual(summary["llm_backends"]["idea-brief"]["text_provider"], "gemini")


if __name__ == "__main__":
    unittest.main()
