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
    def test_user_given_comfy_is_slow_to_acknowledge_a_prompt_when_the_submit_times_out_then_the_failure_names_the_submit_and_the_orphaned_job_is_interrupted_by_its_own_id(self) -> None:
        """User Given ComfyUI accepts a prompt but is too busy to acknowledge it in time When the submit times out Then the failure names the submit and the job the client can no longer wait for is interrupted by its own prompt id instead of burning the GPU."""
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
                    interrupts.append(json.loads(body).get("prompt_id"))
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
        self.assertEqual(interrupts, accepted_ids)


class _ScriptedWebsocket:
    """Replays (seconds_elapsed, event) steps; "timeout" simulates a silent socket."""

    connected = True
    sock = None

    def __init__(self, clock: _FakeClock, prompt_id: str, steps: list[tuple[float, str]]) -> None:
        self.clock = clock
        self.prompt_id = prompt_id
        self.steps = list(steps)

    def settimeout(self, _timeout: float) -> None:
        return

    def recv(self) -> str:
        elapsed, event = self.steps.pop(0)
        self.clock.advance(elapsed)
        if event == "timeout":
            raise websocket.WebSocketTimeoutException("silent")
        if event == "chatter":  # another run's progress frame
            return json.dumps(
                {"type": "executing", "data": {"prompt_id": "someone-else", "node": "sampler"}}
            )
        node = None if event == "finished" else "sampler"
        message_type = "execution_start" if event == "start" else "executing"
        return json.dumps(
            {"type": message_type, "data": {"prompt_id": self.prompt_id, "node": node}}
        )


class _FakeComfyServer:
    """Real local HTTP server answering the /queue, /history and DELETE /queue contract."""

    def __init__(self, *, running: list[str] = (), pending: list[str] = ()) -> None:
        self.running = list(running)
        self.pending = list(pending)
        self.deleted: list[str] = []
        # Mirrors ComfyUI: an interrupt that names a prompt id only acts while that id is running.
        self.interrupted: list[str] = []
        self.unscoped_interrupts = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/queue":
                    self._json(
                        {
                            "queue_running": [[0, prompt_id] for prompt_id in owner.running],
                            "queue_pending": [[1, prompt_id] for prompt_id in owner.pending],
                        }
                    )
                elif self.path.startswith("/history/"):
                    self._json({})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                # ComfyUI's contract: POST /queue {"delete": [ids]} (it has no DELETE route).
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path == "/queue":
                    owner.deleted.extend(json.loads(body).get("delete", []))
                elif self.path == "/interrupt":
                    target = json.loads(body).get("prompt_id")
                    if target is None:
                        owner.unscoped_interrupts += 1
                    elif target in owner.running:
                        owner.interrupted.append(target)
                else:
                    self.send_error(404)
                    return
                self._json({})

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_FakeComfyServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)

    @property
    def port(self) -> int:
        return self._server.server_port


class ComfyCancelTests(unittest.TestCase):
    def test_user_given_a_run_is_abandoned_while_its_prompt_is_queued_when_it_cancels_then_only_that_prompt_leaves_the_queue(self) -> None:
        """User Given my prompt is waiting behind another run's prompt When my run is abandoned Then my prompt is removed from the queue and the other run keeps rendering untouched."""
        with _FakeComfyServer(running=["other-run"], pending=["my-prompt", "another-run"]) as comfy:
            AgenticComfyCommunicator(host="127.0.0.1", port=comfy.port).cancel_prompt("my-prompt")

        self.assertEqual(comfy.deleted, ["my-prompt"])
        self.assertEqual(comfy.interrupted, [])
        self.assertEqual(comfy.unscoped_interrupts, 0)

    def test_user_given_my_prompt_is_rendering_when_my_run_is_abandoned_then_only_my_prompt_is_interrupted(self) -> None:
        """User Given my prompt is the one rendering and another run waits behind it When my run is abandoned Then my render is stopped by its own id and the waiting run is untouched."""
        with _FakeComfyServer(running=["my-prompt"], pending=["another-run"]) as comfy:
            AgenticComfyCommunicator(host="127.0.0.1", port=comfy.port).cancel_prompt("my-prompt")

        self.assertEqual(comfy.interrupted, ["my-prompt"])
        self.assertEqual(comfy.unscoped_interrupts, 0)
        self.assertEqual(comfy.deleted, [])

    def test_user_given_another_run_is_rendering_when_my_run_is_abandoned_then_that_render_is_never_interrupted(self) -> None:
        """User Given another run is rendering and my prompt already left the queue When my run is abandoned Then nothing is interrupted, so the other run's render survives."""
        with _FakeComfyServer(running=["other-run"]) as comfy:
            AgenticComfyCommunicator(host="127.0.0.1", port=comfy.port).cancel_prompt("my-prompt")

        self.assertEqual(comfy.interrupted, [])
        self.assertEqual(comfy.unscoped_interrupts, 0)


class ComfyQueueWaitTests(unittest.TestCase):
    def _communicator(self, comfy: _FakeComfyServer, clock: _FakeClock) -> AgenticComfyCommunicator:
        return AgenticComfyCommunicator(
            host="127.0.0.1", port=comfy.port, timeout=3600, clock=clock
        )

    def test_user_given_my_prompt_waits_in_a_long_queue_when_it_finally_renders_then_the_wait_never_counts_against_the_render_timeout(self) -> None:
        """User Given other runs keep my prompt queued for longer than the render timeout When my prompt starts and finishes rendering Then the run succeeds instead of timing out while it was only waiting."""
        clock = _FakeClock()
        with _FakeComfyServer(running=["other-run"], pending=["my-prompt"]) as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock,
                "my-prompt",
                [(4000, "timeout"), (4000, "timeout"), (0, "start"), (60, "finished")],
            )
            communicator.wait_for_completion("my-prompt")

    def test_user_given_my_prompt_renders_for_longer_than_the_timeout_when_it_keeps_running_then_the_run_fails_with_a_timeout(self) -> None:
        """User Given my prompt has started rendering When it runs past the render timeout Then the run fails with a timeout."""
        clock = _FakeClock()
        with _FakeComfyServer(running=["my-prompt"]) as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock,
                "my-prompt",
                [(0, "start"), (2000, "timeout"), (2000, "timeout")],
            )
            with self.assertRaises(TimeoutError):
                communicator.wait_for_completion("my-prompt")

    def test_user_given_the_start_message_is_lost_when_the_queue_shows_my_prompt_rendering_then_the_render_timeout_still_applies(self) -> None:
        """User Given ComfyUI never announces that my prompt started When the queue shows it rendering past the timeout Then the run still fails instead of waiting forever."""
        clock = _FakeClock()
        with _FakeComfyServer(running=["my-prompt"]) as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock,
                "my-prompt",
                [(10, "timeout"), (2000, "timeout"), (2000, "timeout")],
            )
            with self.assertRaises(TimeoutError):
                communicator.wait_for_completion("my-prompt")

    def test_user_given_comfyui_restarts_and_forgets_my_prompt_when_i_wait_in_the_queue_then_the_run_fails_instead_of_waiting_forever(self) -> None:
        """User Given ComfyUI lost my queued prompt When it is in neither the queue nor the history Then the run fails with a clear reason."""
        clock = _FakeClock()
        with _FakeComfyServer() as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock, "my-prompt", [(5, "timeout")]
            )
            with self.assertRaisesRegex(RuntimeError, "no longer"):
                communicator.wait_for_completion("my-prompt")

    def test_user_given_other_runs_keep_the_socket_busy_when_comfyui_has_forgotten_my_prompt_then_the_run_still_fails_instead_of_waiting_forever(self) -> None:
        """User Given other runs' progress frames arrive constantly When ComfyUI no longer knows my prompt Then the run fails anyway, because a busy socket must not hide a lost prompt."""
        clock = _FakeClock()
        with _FakeComfyServer() as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock, "my-prompt", [(2, "chatter")] * 4
            )
            with self.assertRaisesRegex(RuntimeError, "no longer"):
                communicator.wait_for_completion("my-prompt")

    def test_user_given_other_runs_keep_the_socket_busy_when_my_prompt_renders_past_the_timeout_then_the_run_fails_with_a_timeout(self) -> None:
        """User Given my start message was lost and other runs keep the socket busy When the queue shows my prompt rendering past the timeout Then the run fails with a timeout."""
        clock = _FakeClock()
        with _FakeComfyServer(running=["my-prompt"]) as comfy:
            communicator = self._communicator(comfy, clock)
            communicator.ws = _ScriptedWebsocket(  # type: ignore[assignment]
                clock, "my-prompt", [(6, "chatter"), (2000, "chatter"), (2000, "chatter")]
            )
            with self.assertRaises(TimeoutError):
                communicator.wait_for_completion("my-prompt")


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
