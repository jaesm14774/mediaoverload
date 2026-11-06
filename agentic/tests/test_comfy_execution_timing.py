from __future__ import annotations

import json
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

import websocket

from agentic.tools.comfy_backend import AgenticComfyCommunicator, ComfyPromptTiming


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _CompletedWebsocket:
    connected = True
    sock = None

    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.messages = [
            (
                2.5,
                {
                    "type": "execution_start",
                    "data": {"prompt_id": "completed-prompt"},
                },
            ),
            (
                0.0,
                {
                    "type": "execution_cached",
                    "data": {
                        "prompt_id": "completed-prompt",
                        "nodes": ["loader", "encoder", "conditioning"],
                    },
                },
            ),
            (
                0.0,
                {
                    "type": "executing",
                    "data": {"prompt_id": "completed-prompt", "node": "text"},
                },
            ),
            (
                3.0,
                {
                    "type": "executing",
                    "data": {"prompt_id": "completed-prompt", "node": "sampler"},
                },
            ),
            (
                4.0,
                {
                    "type": "executing",
                    "data": {"prompt_id": "completed-prompt", "node": None},
                },
            ),
        ]

    def settimeout(self, _timeout: float) -> None:
        return

    def recv(self) -> str:
        elapsed, message = self.messages.pop(0)
        self.clock.advance(elapsed)
        return json.dumps(message)


class _HistoryCompletedWebsocket(_CompletedWebsocket):
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.messages = [
            (
                2.5,
                {
                    "type": "execution_start",
                    "data": {"prompt_id": "completed-prompt"},
                },
            )
        ]

    def recv(self) -> str:
        if self.messages:
            return super().recv()
        self.clock.advance(2.5)
        raise websocket.WebSocketTimeoutException("terminal event missing")


class ComfySubmitTimeoutTests(unittest.TestCase):
    def test_user_given_comfy_is_slow_to_acknowledge_a_prompt_when_the_submit_times_out_then_the_orphaned_job_is_interrupted(self) -> None:
        """User Given ComfyUI accepts a prompt but is too busy to acknowledge it in time When the submit times out Then the failure names the submit and the job the client can no longer wait for is interrupted instead of burning the GPU."""
        accepted_ids: list[str] = []
        interrupts: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.path == "/prompt":
                    accepted_ids.append(json.loads(body)["prompt_id"])
                    time.sleep(1.0)  # busy server: accepted, acknowledged too late
                    return
                if self.path == "/interrupt":
                    interrupts.append("interrupt")
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_error(404)

            def do_GET(self) -> None:
                if self.path == "/queue" and accepted_ids:
                    body = json.dumps(
                        {"queue_running": [[0, accepted_ids[0]]], "queue_pending": []}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def log_message(self, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=server.server_port)
        communicator.ws = _CompletedWebsocket(_FakeClock())  # type: ignore[assignment]
        communicator.submit_timeout = 0.2
        try:
            with tempfile.TemporaryDirectory() as directory:
                success, messages = communicator.process_workflow(
                    workflow={"text": {"class_type": "PrimitiveString", "inputs": {"value": "x"}}},
                    updates=[],
                    output_path=directory,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertFalse(success)
        self.assertIn("/prompt", messages[0])
        self.assertEqual(len(interrupts), 1)


class ComfyExecutionTimingTests(unittest.TestCase):
    def test_user_given_comfy_reports_node_events_when_a_render_completes_then_timings_identify_workflow_node_intervals(self) -> None:
        """User Given ComfyUI reports executing events between workflow nodes When updated and unmodified workflows complete Then timings identify each observed node interval and preserve cached-node evidence."""
        submitted_requests: list[dict[str, Any]] = []
        media = {"filename": "render.png", "subfolder": "", "type": "output"}
        media_bytes = b"rendered image bytes"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/prompt":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                submitted_requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps({"prompt_id": "completed-prompt"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/history/completed-prompt":
                    body = json.dumps(
                        {
                            "completed-prompt": {
                                "outputs": {"save-image": {"images": [media]}}
                            }
                        }
                    ).encode()
                    content_type = "application/json"
                elif urlsplit(self.path).path == "/view":
                    body = media_bytes
                    content_type = "image/png"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clock = _FakeClock()
        communicator = AgenticComfyCommunicator(
            host="127.0.0.1", port=server.server_port, clock=clock
        )
        communicator.ws = _CompletedWebsocket(clock)  # type: ignore[assignment]
        template = {
            "text": {
                "class_type": "PrimitiveString",
                "inputs": {"value": "original"},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {"seed": 1},
            },
        }
        timings: list[ComfyPromptTiming] = []
        try:
            with tempfile.TemporaryDirectory() as directory:
                success, files = communicator.process_workflow(
                    workflow=template,
                    updates=[
                        {
                            "type": "direct_update",
                            "node_id": "text",
                            "inputs": {"value": "render request"},
                        }
                    ],
                    output_path=directory,
                    timing_sink=timings,
                )
                self.assertTrue(success)
                self.assertEqual(len(files), 1)
                self.assertEqual(Path(files[0]).read_bytes(), media_bytes)

                communicator.ws = _CompletedWebsocket(clock)  # type: ignore[assignment]
                success, files = communicator.process_workflow(
                    workflow=template,
                    updates=[],
                    output_path=directory,
                    timing_sink=timings,
                )
                self.assertTrue(success)
                self.assertEqual(len(files), 1)
                self.assertEqual(Path(files[0]).read_bytes(), media_bytes)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(template["text"]["inputs"]["value"], "original")
        self.assertEqual(
            submitted_requests[0]["prompt"]["text"]["inputs"]["value"],
            "render request",
        )
        self.assertEqual(submitted_requests[1]["prompt"], template)
        self.assertEqual(len(timings), 2)
        phases = timings[0]
        self.assertTrue(
            {
                "workflow_prepare_seconds",
                "queue_submit_seconds",
                "execution_wait_seconds",
                "queue_wait_seconds",
                "server_execution_seconds",
                "execution_start_observed",
                "result_retrieval_seconds",
                "total_seconds",
            }.issubset(phases)
        )
        self.assertEqual(phases["queue_wait_seconds"], 2.5)
        self.assertEqual(phases["server_execution_seconds"], 7.0)
        self.assertTrue(phases["execution_start_observed"])
        self.assertEqual(
            phases["cached_node_ids"], ["loader", "encoder", "conditioning"]
        )
        self.assertEqual(phases["cached_node_count"], 3)
        self.assertTrue(phases["cached_nodes_observed"])
        self.assertEqual(phases["execution_wait_seconds"], 9.5)
        self.assertEqual(
            phases["node_execution_intervals"],
            [
                {
                    "node_id": "text",
                    "node_type": "PrimitiveString",
                    "seconds": 3.0,
                },
                {
                    "node_id": "sampler",
                    "node_type": "KSampler",
                    "seconds": 4.0,
                },
            ],
        )
        self.assertTrue(phases["node_execution_intervals_complete"])
        self.assertTrue(
            all(
                value >= 0
                for value in phases.values()
                if isinstance(value, (int, float))
            )
        )

    def test_user_given_terminal_websocket_event_is_lost_when_history_confirms_success_then_server_time_is_still_recorded(self) -> None:
        """User Given Comfy reports execution_start but omits its terminal websocket event When history confirms success Then the observed server processing interval is still recorded."""
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/history/completed-prompt":
                    self.send_error(404)
                    return
                body = json.dumps(
                    {
                        "completed-prompt": {
                            "status": {"completed": True, "status_str": "success"}
                        }
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clock = _FakeClock()
        communicator = AgenticComfyCommunicator(
            host="127.0.0.1", port=server.server_port, clock=clock
        )
        communicator.ws = _HistoryCompletedWebsocket(clock)  # type: ignore[assignment]
        timings: ComfyPromptTiming = {
            "queue_wait_seconds": 0.0,
            "server_execution_seconds": 0.0,
            "execution_start_observed": False,
            "cached_node_ids": [],
            "cached_node_count": 0,
            "cached_nodes_observed": False,
        }
        try:
            communicator.wait_for_completion(
                "completed-prompt", timing_sink=timings
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(timings["queue_wait_seconds"], 2.5)
        self.assertEqual(timings["server_execution_seconds"], 2.5)
        self.assertEqual(timings["cached_node_count"], 0)
        self.assertFalse(timings["cached_nodes_observed"])
        self.assertTrue(timings["execution_start_observed"])


if __name__ == "__main__":
    unittest.main()
