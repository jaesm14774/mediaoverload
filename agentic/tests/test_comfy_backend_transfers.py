from __future__ import annotations

import hashlib
import json
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.parse import parse_qs, urlsplit

from agentic.tools.comfy_backend import AgenticComfyCommunicator


class _TransferServer(ThreadingHTTPServer):
    daemon_threads = True
    upload_content_length = 0
    upload_content_type = ""
    upload_body = b""
    upload_request_path = ""
    download_body = b""
    download_query: dict[str, list[str]] = {}


class ComfyBackendTransferTests(unittest.TestCase):
    def _start_server(self, handler: type[BaseHTTPRequestHandler]) -> tuple[_TransferServer, Thread]:
        server = _TransferServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    @staticmethod
    def _stop_server(server: _TransferServer, thread: Thread) -> None:
        server.shutdown()
        thread.join()
        server.server_close()

    def test_user_uploads_a_conditioning_image_given_a_large_file_when_comfy_receives_it_then_the_complete_multipart_image_is_available(self) -> None:
        """User uploads a conditioning image Given a large local file When Comfy receives it Then the complete multipart image is available."""
        image_content = bytes(range(256)) * 8192

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers["Content-Length"])
                remaining = content_length
                body = bytearray()
                while remaining:
                    chunk = self.rfile.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    body.extend(chunk)
                    remaining -= len(chunk)
                self.server.upload_content_length = content_length
                self.server.upload_content_type = self.headers.get("Content-Type", "")
                self.server.upload_body = bytes(body)
                self.server.upload_request_path = self.path
                response_body = json.dumps(
                    {"name": "conditioning.png", "subfolder": "references"}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "conditioning.png"
            image_path.write_bytes(image_content)
            server, thread = self._start_server(Handler)
            try:
                communicator = AgenticComfyCommunicator(
                    "127.0.0.1", server.server_port, timeout=2
                )
                uploaded_file = communicator.upload_image(
                    str(image_path), subfolder="references", include_sha256=True
                )
            finally:
                self._stop_server(server, thread)

        self.assertEqual(uploaded_file.comfy_path, "references/conditioning.png")
        self.assertEqual(uploaded_file.sha256, hashlib.sha256(image_content).hexdigest())
        self.assertEqual(server.upload_request_path, "/upload/image")
        self.assertIn("multipart/form-data; boundary=", server.upload_content_type)
        self.assertEqual(server.upload_content_length, len(server.upload_body))
        self.assertIn(image_content, server.upload_body)

    def test_user_uploads_a_conditioning_image_without_digest_evidence_when_comfy_receives_it_then_upload_receipt_omits_sha256(self) -> None:
        """User Given an ordinary conditioning upload When Comfy receives it Then no unused SHA-256 pass is requested."""

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                remaining = int(self.headers["Content-Length"])
                while remaining:
                    chunk = self.rfile.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                response_body = json.dumps({"name": "conditioning.png"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "conditioning.png"
            image_path.write_bytes(b"local conditioning image")
            server, thread = self._start_server(Handler)
            try:
                communicator = AgenticComfyCommunicator(
                    "127.0.0.1", server.server_port, timeout=2
                )
                uploaded_file = communicator.upload_image(str(image_path))
            finally:
                self._stop_server(server, thread)

        self.assertEqual(uploaded_file.comfy_path, "conditioning.png")
        self.assertIsNone(uploaded_file.sha256)

    def test_user_downloads_comfy_output_given_server_media_when_saved_then_bytes_and_request_parameters_are_preserved(self) -> None:
        """User downloads a Comfy output Given server media and a destination When the file is saved Then bytes and request parameters are preserved."""
        download_body = bytes(range(251)) * 8192

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.server.download_query = parse_qs(urlsplit(self.path).query)
                self.server.download_body = download_body
                self.send_response(200)
                self.send_header("Content-Length", str(len(download_body)))
                self.end_headers()
                self.wfile.write(download_body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "render.mp4"
            server, thread = self._start_server(Handler)
            try:
                communicator = AgenticComfyCommunicator(
                    "127.0.0.1", server.server_port, timeout=2
                )
                communicator.download_media_file(
                    "render.mp4",
                    "scene folder",
                    "output",
                    str(output_path),
                )
            finally:
                self._stop_server(server, thread)

            self.assertEqual(output_path.read_bytes(), download_body)
            self.assertEqual(list(Path(directory).iterdir()), [output_path])

        self.assertEqual(server.download_query["filename"], ["render.mp4"])
        self.assertEqual(server.download_query["subfolder"], ["scene folder"])
        self.assertEqual(server.download_query["type"], ["output"])
