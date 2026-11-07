"""Readable ComfyUI failures and early workflow checks.

The rejection bodies below were captured from a real local ComfyUI ``/prompt``
endpoint (validation only; nothing was queued), then trimmed. They pin the
contract the diagnostics read: ``error`` and ``node_errors`` with typed entries.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from agentic.tools.comfy_backend import AgenticComfyCommunicator
from agentic.tools.comfy_diagnostics import validate_workflow_graph

INSTALLED_MODELS = [
    "sdxl\\novaAnimeXL_ilV180.safetensors",
    "z_image\\zImageTurboAnime_aioBF16.safetensors",
]

MISSING_MODEL_REJECTION = {
    "error": {
        "type": "prompt_outputs_failed_validation",
        "message": "Prompt outputs failed validation",
        "details": "",
        "extra_info": {},
    },
    "node_errors": {
        "1": {
            "errors": [
                {
                    "type": "value_not_in_list",
                    "message": "Value not in list",
                    "details": "ckpt_name: 'does_not_exist_model.safetensors' not in "
                    + repr(INSTALLED_MODELS),
                    "extra_info": {
                        "input_name": "ckpt_name",
                        "input_config": [INSTALLED_MODELS, {"tooltip": "checkpoint"}],
                        "received_value": "does_not_exist_model.safetensors",
                    },
                }
            ],
            "dependent_outputs": ["3"],
            "class_type": "CheckpointLoaderSimple",
        }
    },
}

MISSING_NODE_REJECTION = {
    "error": {
        "type": "missing_node_type",
        "message": "Node 'NoSuchCustomNode' not found. The custom node may not be installed.",
        "details": "Node ID '#2'",
        "extra_info": {
            "node_id": "2",
            "class_type": "NoSuchCustomNode",
            "node_title": "NoSuchCustomNode",
        },
    },
    "node_errors": {},
}


def _rejecting_server(status: int, body: bytes) -> tuple[ThreadingHTTPServer, Thread]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _submit_to_rejecting_server(status: int, body: bytes) -> str:
    server, thread = _rejecting_server(status, body)
    try:
        communicator = AgenticComfyCommunicator("127.0.0.1", server.server_port, timeout=2)
        try:
            communicator.queue_prompt({"1": {"class_type": "X", "inputs": {}}}, "prompt-id")
        except RuntimeError as exc:
            return str(exc)
        raise AssertionError("queue_prompt accepted a rejected workflow")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class ComfyRejectionTests(unittest.TestCase):
    def test_user_submits_a_workflow_naming_an_uninstalled_model_when_comfy_rejects_it_then_the_failure_names_the_model_without_dumping_installed_models(self) -> None:
        """User Given a workflow that names a model ComfyUI does not have When ComfyUI rejects the prompt Then the failure names that model and does not list the installed ones."""
        message = _submit_to_rejecting_server(
            400, json.dumps(MISSING_MODEL_REJECTION).encode()
        )

        self.assertIn("does_not_exist_model.safetensors", message)
        self.assertIn("CheckpointLoaderSimple", message)
        self.assertNotIn("novaAnimeXL", message)

    def test_user_submits_a_workflow_needing_a_missing_custom_node_when_comfy_rejects_it_then_the_failure_names_the_node_type(self) -> None:
        """User Given a workflow using a custom node ComfyUI lacks When ComfyUI rejects the prompt Then the failure names the node type."""
        message = _submit_to_rejecting_server(
            400, json.dumps(MISSING_NODE_REJECTION).encode()
        )

        self.assertIn("NoSuchCustomNode", message)

    def test_user_submits_a_workflow_when_comfy_answers_with_an_unstructured_error_then_status_and_body_are_still_reported(self) -> None:
        """User Given a ComfyUI that answers with a non-JSON error When a prompt is submitted Then the HTTP status and the reply text reach the user."""
        message = _submit_to_rejecting_server(502, b"upstream gateway exploded")

        self.assertIn("502", message)
        self.assertIn("upstream gateway exploded", message)


class WorkflowGraphTests(unittest.TestCase):
    def test_user_given_a_workflow_with_a_link_to_a_missing_node_when_checked_then_the_broken_link_is_named(self) -> None:
        """User Given a workflow whose input points at a node that is not in it When it is checked Then the problem names the node, input and missing target."""
        workflow = {
            "3": {"class_type": "SaveImage", "inputs": {"images": ["99", 0], "filename_prefix": "x"}},
        }

        problems = validate_workflow_graph(workflow)

        self.assertEqual(len(problems), 1)
        for expected in ("SaveImage", "images", "99"):
            self.assertIn(expected, problems[0])

    def test_user_given_a_consistent_workflow_when_checked_then_no_problem_is_reported(self) -> None:
        """User Given a workflow whose links all resolve When it is checked Then nothing is reported, whether node ids are text or numbers."""
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": [1, 0], "filename_prefix": "x"}},
        }

        self.assertEqual(validate_workflow_graph(workflow), [])

    def test_user_runs_any_shipped_workflow_when_it_is_checked_then_every_link_resolves(self) -> None:
        """User Given each workflow shipped in configs/workflow When it is checked Then no link points at a missing node."""
        workflow_dir = Path(__file__).resolve().parents[2] / "configs" / "workflow"
        paths = sorted(workflow_dir.glob("*.json"))
        self.assertTrue(paths, "no shipped workflows found")

        for path in paths:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(validate_workflow_graph(workflow), [])


if __name__ == "__main__":
    unittest.main()
