"""Public save-results contracts through a local ComfyUI-shaped HTTP endpoint."""

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from PIL import Image
import pytest

from agentic.tools.comfy_backend import AgenticComfyCommunicator


@pytest.fixture
def comfy_endpoint():
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "blue").save(image, format="PNG")
    media = image.getvalue()
    metadata = {"filename": "render.png", "subfolder": "", "type": "output"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/history/prompt-1":
                body = json.dumps({"prompt-1": {"outputs": {"save-image": {"images": [metadata]}}}}).encode()
                content_type = "application/json"
            elif self.path.startswith("/view?"):
                body, content_type = media, "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield AgenticComfyCommunicator(host="127.0.0.1", port=server.server_port), metadata, media
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("folder_type", ["output", "temp"])
@pytest.mark.parametrize("filename", ["../outside.png", "..\\outside.png", "nested/../../outside.png", "nested\\..\\..\\outside.png", "/outside.png", "\\outside.png", "C:\\outside.png", "C:outside.png", "\\\\server\\share\\outside.png"])
def test_user_given_unsafe_provider_filenames_when_saving_results_then_no_file_is_written_outside_the_run(filename, folder_type, tmp_path, comfy_endpoint):
    """User Given a provider filename with POSIX or Windows traversal or an absolute path When results are saved through the HTTP adapter Then the run reports failure and writes no media."""
    communicator, metadata, _ = comfy_endpoint
    metadata.update(filename=filename, type=folder_type)
    output = tmp_path / "run"
    output.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"existing media")
    success, errors = communicator.save_results("prompt-1", str(output), None)
    assert success is False
    assert errors
    assert not list(output.rglob("*.png"))
    assert outside.read_bytes() == b"existing media"


@pytest.mark.parametrize("folder_type", ["output", "temp"])
def test_user_given_a_valid_provider_filename_when_saving_results_then_original_media_bytes_are_preserved(folder_type, tmp_path, comfy_endpoint):
    """User Given a valid generated media filename When results are saved through the HTTP adapter Then the media stays inside the run directory and preserves the provider bytes."""
    communicator, metadata, media = comfy_endpoint
    metadata["type"] = folder_type
    output = tmp_path / "run"
    success, saved = communicator.save_results("prompt-1", str(output), None)
    assert success is True
    assert saved == [str(output / "render.png")]
    assert (output / "render.png").read_bytes() == media


@pytest.mark.parametrize("folder_type", ["output", "temp"])
def test_user_given_a_symlink_outside_the_run_when_provider_media_are_saved_then_existing_files_are_preserved(folder_type, tmp_path, comfy_endpoint):
    """User Given a relative media path that resolves through a symlink outside the run When results are saved Then the adapter rejects it and preserves the existing destination."""
    communicator, metadata, _ = comfy_endpoint
    output = tmp_path / "run"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    preserved = outside / "render.png"
    preserved.write_bytes(b"existing destination")
    try:
        (output / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a local symlink is not permitted")
    metadata.update(filename="link/render.png", type=folder_type)
    success, errors = communicator.save_results("prompt-1", str(output), None)
    assert success is False
    assert errors
    assert preserved.read_bytes() == b"existing destination"
