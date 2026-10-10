"""ChatModel contracts through the real HTTP adapter and a local TLS endpoint.

This verifies transport interoperability, not a cloud provider's availability
or the quality of its model output.
"""

import base64
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from queue import Empty, Queue
import ssl
from threading import Thread
from types import SimpleNamespace

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from PIL import Image
import pytest

from agentic.runtime.model_backends import (
    ModelConfig,
    OpenAICompatibleModel,
    ProviderRequestError,
)
from agentic.runtime.prompt_requests import JsonChatRequest
from agentic.runtime.structured_chat import chat_json
from provider_fakes import ScriptedChatModel


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = tmp_path / "server.pem", tmp_path / "server-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(certificate_path))
    monkeypatch.setenv("MEDIAOVERLOAD_CONTRACT_API_KEY", "local-contract-key")
    monkeypatch.setenv("AGENTIC_ALLOWED_IMAGE_ROOTS", str(tmp_path))
    responses, requests = Queue(), []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                {"path": self.path, "headers": dict(self.headers), "payload": payload}
            )
            try:
                status, body, headers = responses.get_nowait()
            except Empty:
                status, body, headers = (
                    500,
                    {"error": "no contract response configured"},
                    {},
                )
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            for header, value in headers.items():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield SimpleNamespace(
            url=f"https://127.0.0.1:{server.server_port}/v1",
            responses=responses,
            requests=requests,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def adapter(endpoint, *, supports_response_format=True):
    return OpenAICompatibleModel(
        ModelConfig("contract-model"),
        provider_name="contract-provider",
        base_url=endpoint.url,
        api_key_env=("MEDIAOVERLOAD_CONTRACT_API_KEY",),
        supports_response_format=supports_response_format,
    )


def completion(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.mark.parametrize("implementation", ["http", "scripted"])
def test_user_given_a_chat_provider_when_requesting_structured_content_then_real_and_scripted_providers_obey_the_same_public_contract(
    endpoint,
    implementation,
):
    """User Given a real HTTP adapter or scripted ChatModel When the same structured request is submitted Then both return response text accepted by the real JSON transport."""
    content = '{"text":"種子落在手心。"}'
    endpoint.responses.put((200, completion(content), {}))
    model = (
        adapter(endpoint) if implementation == "http" else ScriptedChatModel([content])
    )
    result = chat_json(
        JsonChatRequest(
            manager=SimpleNamespace(text_model=model),
            system_prompt="Write a line.",
            user_prompt="A seed lands safely.",
            schema_name="provider_contract_v1",
            schema={"type": "object"},
            language_contract="Use Traditional Chinese.",
            max_retries=1,
            repair_attempts=0,
        )
    )
    assert result["text"] == "種子落在手心。"
    if implementation == "http":
        request = endpoint.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["headers"]["Authorization"] == "Bearer local-contract-key"
        assert request["payload"]["model"] == "contract-model"
        assert all(
            "Use Traditional Chinese." in message["content"]
            for message in request["payload"]["messages"]
        )


def test_user_given_a_real_image_when_calling_a_vision_provider_then_the_request_contains_the_original_image(
    endpoint,
    tmp_path,
):
    """User Given an allowed real image When calling a vision provider Then its bytes and text reach the HTTP endpoint and the response remains text."""
    image = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "green").save(image)
    endpoint.responses.put((200, completion("A green square."), {}))
    result = adapter(endpoint).chat_completion(
        [{"role": "user", "content": "Describe it."}],
        images=[str(image)],
        max_retries=1,
    )
    assert result == "A green square."
    parts = endpoint.requests[0]["payload"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Describe it."}
    assert (
        base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])
        == image.read_bytes()
    )


def test_user_given_an_incompatible_schema_provider_when_requesting_json_then_the_wire_request_respects_its_capability(
    endpoint,
):
    """User Given a provider without response-format support When JSON is requested Then its HTTP request omits that option while content still reaches the caller."""
    endpoint.responses.put((200, completion('{"ok":true}'), {}))
    result = adapter(endpoint, supports_response_format=False).chat_completion(
        [{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
        max_retries=1,
    )
    assert json.loads(result)["ok"]
    assert "response_format" not in endpoint.requests[0]["payload"]


def test_user_given_a_rate_limited_provider_when_retry_is_allowed_then_a_later_http_response_can_complete_the_request(
    endpoint,
):
    """User Given a provider response with a zero-delay retry instruction When retry is allowed Then the eventual successful HTTP response is returned."""
    endpoint.responses.put((429, {"error": "rate limited"}, {"Retry-After": "0"}))
    endpoint.responses.put((200, completion("Recovered."), {}))
    assert (
        adapter(endpoint).chat_completion(
            [{"role": "user", "content": "Continue."}],
            max_retries=2,
            initial_retry_delay=0,
        )
        == "Recovered."
    )


def test_user_given_a_failed_http_provider_when_no_retry_remains_then_failure_is_explicit(
    endpoint,
):
    """User Given a failed HTTP provider When no retry remains Then a provider error reaches the caller rather than empty generated content."""
    endpoint.responses.put((500, {"error": "unavailable"}, {}))
    with pytest.raises(ProviderRequestError):
        adapter(endpoint).chat_completion(
            [{"role": "user", "content": "Continue."}], max_retries=1
        )
