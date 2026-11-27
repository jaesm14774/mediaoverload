"""ComfyUI restart contract before H3; a fake HTTP server stands in for ComfyUI + Manager.

The fake reproduces what a real ComfyUI 0.45 with ComfyUI-Manager 4.2.2 did on 2026-11-27:
the reboot POST is answered by a dropped connection, /system_stats then refuses
connections for ~30 s, and the new process answers again. Real-GPU behaviour runs separately.
"""
from __future__ import annotations

import json
import socket
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib import request

from agentic.tools.comfy_backend import restart_comfyui

DOWN = "drop"
OLD = "ok"


class _FakeComfy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, reboot_status: int | None, polls_after_reboot: list[str]) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.reboot_status = reboot_status
        # What each /system_stats poll after the reboot POST sees, in order:
        # OLD = the old process still answers, DOWN = connection dropped.
        # Once the script is used up the new process answers.
        self.script = list(polls_after_reboot)
        self.rebooted = False
        self.reboots = 0
        self.new_process_polls = 0


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def _drop(self) -> None:
        self.close_connection = True
        self.connection.shutdown(socket.SHUT_RDWR)

    def _answer(self, generation: int) -> None:
        body = json.dumps({"system": {"comfyui_version": "0.45.3"}, "generation": generation}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/system_stats":
            self.send_error(404)
            return
        if not self.server.rebooted:
            self._answer(1)
            return
        if self.server.script:
            if self.server.script.pop(0) == DOWN:
                self._drop()
            else:
                self._answer(1)
            return
        self.server.new_process_polls += 1
        self._answer(2)

    def do_POST(self) -> None:
        if self.path != "/v2/manager/reboot":
            self.send_error(404)
            return
        # Manager rejects simple form posts, so a JSON content type is part of the contract.
        if self.headers.get("Content-Type") != "application/json":
            self.send_error(400)
            return
        if self.server.reboot_status is not None:
            self.send_error(self.server.reboot_status)
            return
        self.server.reboots += 1
        self.server.rebooted = True
        self._drop()


def _quick(port: int, timeout_seconds: float = 5) -> None:
    restart_comfyui("127.0.0.1", port, timeout_seconds=timeout_seconds, poll_seconds=0.01)


class ComfyRestartTests(unittest.TestCase):
    def _serve(self, reboot_status: int | None = None, polls_after_reboot: list[str] | None = None) -> _FakeComfy:
        server = _FakeComfy(reboot_status=reboot_status, polls_after_reboot=polls_after_reboot or [])
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_user_requests_an_h3_render_given_comfyui_with_manager_when_it_is_prepared_then_comfyui_restarted_and_answers_again(self) -> None:
        """User requests an H3 render Given ComfyUI with Manager When the render is prepared Then ComfyUI has restarted and answers again before anything is submitted."""
        server = self._serve(polls_after_reboot=[DOWN] * 5)

        _quick(server.server_port)

        self.assertEqual(server.reboots, 1)
        with request.urlopen(f"http://127.0.0.1:{server.server_port}/system_stats", timeout=2) as response:
            self.assertEqual(json.loads(response.read())["generation"], 2)

    def test_user_requests_an_h3_render_given_the_old_process_lingers_when_it_is_prepared_then_the_render_waits_for_the_new_process(self) -> None:
        """User requests an H3 render Given the old ComfyUI keeps answering for a moment after the reboot request When the render is prepared Then it waits for the new process, not the old one."""
        server = self._serve(polls_after_reboot=[OLD] * 4 + [DOWN] * 5)

        _quick(server.server_port)

        self.assertEqual(server.script, [])
        self.assertGreaterEqual(server.new_process_polls, 1)

    def test_user_requests_an_h3_render_given_a_transient_drop_while_the_old_process_shuts_down_when_it_is_prepared_then_the_render_still_waits_for_the_new_process(self) -> None:
        """User requests an H3 render Given one dropped connection while the old ComfyUI is still shutting down When the render is prepared Then it does not take that blip for the restart and still waits for the new process."""
        server = self._serve(polls_after_reboot=[OLD, DOWN, OLD, OLD, DOWN, DOWN, DOWN, DOWN, DOWN])

        _quick(server.server_port)

        self.assertEqual(server.script, [])
        self.assertGreaterEqual(server.new_process_polls, 1)

    def test_user_requests_an_h3_render_given_manager_refuses_reboot_when_it_is_prepared_then_the_refusal_is_reported(self) -> None:
        """User requests an H3 render Given Manager refuses to reboot ComfyUI When the render is prepared Then the refusal is reported instead of rendering on a stale process."""
        server = self._serve(reboot_status=403)

        with self.assertRaisesRegex(RuntimeError, "403"):
            _quick(server.server_port)

        self.assertEqual(server.reboots, 0)

    def test_user_requests_an_h3_render_given_comfyui_is_not_running_when_it_is_prepared_then_the_error_says_it_is_unreachable(self) -> None:
        """User requests an H3 render Given nothing listens on the ComfyUI port When the render is prepared Then the error says ComfyUI is unreachable instead of waiting for a restart."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]

        with self.assertRaisesRegex(RuntimeError, "not reachable"):
            _quick(closed_port)

    def test_user_requests_an_h3_render_given_comfyui_ignores_the_reboot_when_it_is_prepared_then_it_fails_instead_of_rendering_on_the_old_process(self) -> None:
        """User requests an H3 render Given ComfyUI never goes down after the reboot request When the render is prepared Then it fails instead of rendering on the old process."""
        server = self._serve(polls_after_reboot=[OLD] * 10**6)

        with self.assertRaisesRegex(RuntimeError, "did not restart"):
            _quick(server.server_port, timeout_seconds=0.3)

    def test_user_requests_an_h3_render_given_comfyui_never_comes_back_when_it_is_prepared_then_it_fails_with_an_actionable_error(self) -> None:
        """User requests an H3 render Given ComfyUI stays down after the restart When the render is prepared Then it fails with an error that says ComfyUI did not come back."""
        server = self._serve(polls_after_reboot=[DOWN] * 10**6)

        with self.assertRaisesRegex(RuntimeError, "did not come back"):
            _quick(server.server_port, timeout_seconds=0.3)


if __name__ == "__main__":
    unittest.main()
