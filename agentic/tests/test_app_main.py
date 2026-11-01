from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from agentic.app.main import _build_prompt_summary


class AppMainTests(unittest.TestCase):
    def test_user_given_the_cli_entrypoint_when_review_commands_are_loaded_then_generation_backends_stay_unloaded(self) -> None:
        """User Given a CLI process needs to load review commands When its entrypoint module is imported Then LLM, Comfy workflow, and media service backends remain unloaded."""
        project_root = Path(__file__).resolve().parents[2]
        source_root = project_root / "agentic" / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        script = (
            "import json, sys; import agentic.app.main; "
            "heavy = ('agentic.runtime.llm_engine', 'agentic.runtime.llm_client', "
            "'agentic.runtime.model_backends', 'agentic.tools.media_services', "
            "'agentic.skills.longvideo', 'agentic.tools.comfy_h3', "
            "'agentic.app.character_config', 'agentic.tools.publishing_adapter', "
            "'agentic.tools.social_native', 'requests', "
            "'agentic.runtime.serial_video', 'pymysql', 'cryptography'); "
            "print(json.dumps([name for name in heavy if name in sys.modules]))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=env,
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(json.loads(completed.stdout), [])

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
