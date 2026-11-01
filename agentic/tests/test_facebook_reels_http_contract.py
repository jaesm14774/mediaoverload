from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlparse

from agentic.tools.social_native import FacebookPlatform, MediaPost


class _ReelApiState:
    def __init__(self, *, upload_success: bool = True, finish_status: int = 200) -> None:
        self.upload_success = upload_success
        self.finish_status = finish_status
        self.calls: list[str] = []
        self.upload_bytes = 0
        self.upload_headers: dict[str, str] = {}
        self.finish_params: dict[str, list[str]] = {}


class _ReelApiServer(ThreadingHTTPServer):
    state: _ReelApiState


def _handler_for(state: _ReelApiState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, body: dict[str, object], status: int = 200) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path == "/upload/video-1":
                state.calls.append("upload")
                state.upload_headers = {key.lower(): value for key, value in self.headers.items()}
                state.upload_bytes = len(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self._respond({"success": state.upload_success})
                return
            if params.get("upload_phase") == ["start"]:
                state.calls.append("start")
                upload_url = f"http://127.0.0.1:{self.server.server_port}/upload/video-1"
                self._respond({"video_id": "video-1", "upload_url": upload_url})
                return
            if params.get("upload_phase") == ["finish"]:
                state.calls.append("finish")
                state.finish_params = params
                if state.finish_status >= 400:
                    self._respond({"error": {"message": "transient finish response"}}, state.finish_status)
                else:
                    self._respond({"success": True})
                return
            self._respond({"error": {"message": "unexpected request"}}, 404)

        def do_GET(self) -> None:
            state.calls.append("status")
            self._respond(
                {
                    "id": "video-1",
                    "status": {
                        "video_status": "ready",
                        "uploading_phase": {"status": "complete"},
                        "processing_phase": {"status": "complete"},
                        "publishing_phase": {"status": "not_started"},
                    },
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    return Handler


@contextmanager
def _serve(state: _ReelApiState) -> Iterator[str]:
    server = _ReelApiServer(("127.0.0.1", 0), _handler_for(state))
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FacebookReelsHttpContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory(prefix="facebook-reels-contract-")
        cls.root = Path(cls._temporary.name)
        cls.video_path = cls.root / "reel.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=0x17233b:s=720x1280:r=24:d=4.2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4.2",
                "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.video_path),
            ],
            check=True,
            capture_output=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _platform(self, api_base: str) -> FacebookPlatform:
        keys = ("FB_PAGE_ID", "FB_PAGE_ACCESS_TOKEN")
        previous = {key: os.environ.get(key) for key in keys}
        os.environ.update({"FB_PAGE_ID": "local-page", "FB_PAGE_ACCESS_TOKEN": "local-contract-token"})
        try:
            platform = FacebookPlatform(str(self.root / "credentials"))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        platform.GRAPH_API_BASE = api_base
        return platform

    def _post(self, platform: FacebookPlatform) -> bool:
        return platform.upload_post(
            MediaPost(
                media_paths=[str(self.video_path)],
                caption="Draft caption",
                additional_params={
                    "facebook_use_reels": True,
                    "facebook_video_state": "DRAFT",
                },
            )
        )

    def test_user_given_a_valid_vertical_reel_when_draft_upload_finishes_then_file_and_receipt_are_verified(self) -> None:
        """User Given a valid vertical Reel and a local Graph API When a draft upload finishes Then the binary transfer and verified receipt follow the external API contract."""
        state = _ReelApiState()
        with _serve(state) as api_base:
            platform = self._platform(api_base)

            result = self._post(platform)

        self.assertTrue(result)
        self.assertEqual(state.calls, ["start", "upload", "finish", "status"])
        self.assertGreater(state.upload_bytes, 0)
        self.assertEqual(state.upload_headers["authorization"], "OAuth local-contract-token")
        self.assertEqual(state.upload_headers["offset"], "0")
        self.assertEqual(state.upload_headers["file_size"], str(state.upload_bytes))
        self.assertEqual(state.finish_params["upload_phase"], ["finish"])
        self.assertEqual(state.finish_params["video_state"], ["DRAFT"])
        self.assertEqual(state.finish_params["description"], ["Draft caption"])
        self.assertEqual(platform.last_publish_receipt["external_id"], "video-1")  # type: ignore[index]
        self.assertEqual(platform.last_publish_receipt["status"], "draft_ready")  # type: ignore[index]
        self.assertTrue(platform.last_publish_receipt["verified"])  # type: ignore[index]

    def test_user_given_a_finish_response_error_when_the_external_reel_is_ready_then_the_upload_is_reconciled(self) -> None:
        """User Given a transient finish response error When status confirms the draft Reel is ready Then the successful external artifact is reconciled."""
        state = _ReelApiState(finish_status=500)
        with _serve(state) as api_base:
            platform = self._platform(api_base)

            result = self._post(platform)

        self.assertTrue(result)
        self.assertEqual(state.calls, ["start", "upload", "finish", "status"])
        self.assertEqual(platform.last_publish_receipt["status"], "draft_ready")  # type: ignore[index]
        self.assertTrue(platform.last_publish_receipt["details"]["reconciled_after_finish_error"])  # type: ignore[index]

    def test_user_given_a_failed_binary_transfer_when_uploading_a_reel_then_finish_is_not_sent(self) -> None:
        """User Given a rejected binary upload When a Reel draft is sent Then publishing does not enter the finish phase."""
        state = _ReelApiState(upload_success=False)
        with _serve(state) as api_base:
            platform = self._platform(api_base)

            with self.assertRaisesRegex(RuntimeError, "did not confirm success"):
                self._post(platform)

        self.assertEqual(state.calls, ["start", "upload"])
        self.assertEqual(platform.last_publish_receipt["status"], "upload_failed")  # type: ignore[index]
        self.assertFalse(platform.last_publish_receipt["verified"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
