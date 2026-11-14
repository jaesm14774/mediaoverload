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
    FallbackChatModel,
    ModelConfig,
    OpenAICompatibleModel,
    ProviderRequestError,
)
from agentic.runtime.observability import RunRecorder
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


def adapter(endpoint, *, model_name="contract-model", supports_response_format=True):
    return OpenAICompatibleModel(
        ModelConfig(model_name),
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


def test_user_given_a_real_image_and_primary_provider_failure_when_vision_falls_back_then_each_provider_receives_the_original_image(
    endpoint,
    tmp_path,
):
    """User Given a real image and a primary provider failure When vision falls back to another provider Then both HTTP requests carry the original image bytes and the fallback result is returned."""
    image = tmp_path / "fallback-reference.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    endpoint.responses.put((503, {"error": "temporarily unavailable"}, {}))
    endpoint.responses.put((200, completion("A blue square."), {}))
    model = FallbackChatModel(
        adapter(endpoint, model_name="primary-model"),
        adapter(endpoint, model_name="fallback-model"),
    )

    result = model.chat_completion(
        [{"role": "user", "content": "Describe this image."}],
        images=[str(image)],
        max_retries=1,
    )

    assert result == "A blue square."
    assert [request["payload"]["model"] for request in endpoint.requests] == [
        "primary-model",
        "fallback-model",
    ]
    encoded_images = [
        request["payload"]["messages"][0]["content"][1]["image_url"]["url"]
        for request in endpoint.requests
    ]
    assert all(
        base64.b64decode(value.split(",", 1)[1]) == image.read_bytes()
        for value in encoded_images
    )
    assert encoded_images[0] == encoded_images[1]


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
    model = adapter(endpoint)
    assert (
        model.chat_completion(
            [{"role": "user", "content": "Continue."}],
            max_retries=2,
            initial_retry_delay=0,
        )
        == "Recovered."
    )
    assert model.last_attempts[0]["request_count"] == 2
    assert model.last_attempts[0]["status"] == "success"


def test_user_given_a_failed_http_provider_when_no_retry_remains_then_failure_is_explicit(
    endpoint,
):
    """User Given a failed HTTP provider When no retry remains Then a provider error reaches the caller rather than empty generated content."""
    endpoint.responses.put((500, {"error": "unavailable"}, {}))
    with pytest.raises(ProviderRequestError):
        adapter(endpoint).chat_completion(
            [{"role": "user", "content": "Continue."}], max_retries=1
        )


def test_user_given_a_provider_fallback_when_a_structured_call_is_recorded_then_elapsed_time_and_each_attempt_are_saved(
    endpoint,
    tmp_path,
):
    """User Given the first provider fails and a later provider responds When a structured LLM call is recorded Then elapsed time and each provider attempt are saved without duplicating prompt or response data."""
    endpoint.responses.put((500, {"error": "unavailable"}, {}))
    endpoint.responses.put((200, completion('{"ready":true}'), {}))
    recorder = RunRecorder(tmp_path, "provider-attempt-timing")
    manager = SimpleNamespace(
        text_model=FallbackChatModel(
            adapter(endpoint, model_name="primary-model"),
            adapter(endpoint, model_name="fallback-model"),
        )
    )

    result = chat_json(
        JsonChatRequest(
            manager=manager,
            system_prompt="Return a small object.",
            user_prompt="Return ready true.",
            schema_name="provider_attempt_timing_v1",
            schema={"type": "object"},
            recorder=recorder,
            max_retries=1,
            repair_attempts=0,
        )
    )

    call = json.loads(
        next(recorder.run_dir.joinpath("llm").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert result["ready"] is True
    assert call["chat_completion_elapsed_seconds"] >= 0
    attempts = call["provider_attempts"]
    assert [attempt["model"] for attempt in attempts] == [
        "primary-model",
        "fallback-model",
    ]
    assert [attempt["status"] for attempt in attempts] == ["failed", "success"]
    assert all(attempt["duration_seconds"] >= 0 for attempt in attempts)
    assert all(attempt["request_count"] == 1 for attempt in attempts)
    assert all("messages" not in attempt and "response" not in attempt for attempt in attempts)
    assert len(endpoint.requests) == 2
    completed_event = next(
        event
        for event in map(json.loads, recorder.events_path.read_text(encoding="utf-8").splitlines())
        if event["event"] == "llm.call.completed"
    )
    assert completed_event["payload"]["chat_completion_elapsed_seconds"] >= 0
    assert completed_event["payload"]["provider_attempts_count"] == 2


def truncated_reasoning_completion():
    return {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": 'We need to decide suitability. Excerpt: "今天鋒面通過、各地',
                },
            }
        ]
    }


def test_user_given_a_reasoning_only_completion_cut_off_by_the_output_limit_when_it_is_read_then_the_error_says_the_output_was_truncated(
    endpoint,
):
    """User Given a provider that spends its whole output budget reasoning When the completion has no answer text Then the error says the output hit its length limit instead of passing reasoning off as the answer."""
    endpoint.responses.put((200, truncated_reasoning_completion(), {}))
    with pytest.raises(ProviderRequestError) as failure:
        adapter(endpoint).chat_completion(
            [{"role": "user", "content": "Continue."}], max_retries=1
        )
    assert "length" in str(failure.value)


def test_user_given_a_provider_failure_when_the_attempt_is_recorded_then_the_reason_is_saved_with_it(
    endpoint,
    tmp_path,
):
    """User Given the first provider returns no answer and a later provider responds When the structured call is recorded Then the failed attempt carries its reason so the next outage can be diagnosed from the run log."""
    endpoint.responses.put((200, truncated_reasoning_completion(), {}))
    endpoint.responses.put((200, completion('{"ready":true}'), {}))
    recorder = RunRecorder(tmp_path, "provider-attempt-reason")
    manager = SimpleNamespace(
        text_model=FallbackChatModel(
            adapter(endpoint, model_name="primary-model"),
            adapter(endpoint, model_name="fallback-model"),
        )
    )

    chat_json(
        JsonChatRequest(
            manager=manager,
            system_prompt="Return a small object.",
            user_prompt="Return ready true.",
            schema_name="provider_attempt_reason_v1",
            schema={"type": "object"},
            recorder=recorder,
            max_retries=1,
            repair_attempts=0,
        )
    )

    call = json.loads(
        next(recorder.run_dir.joinpath("llm").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    failed, succeeded = call["provider_attempts"]
    assert "length" in failed["error"]
    assert "error" not in succeeded


def test_user_given_the_cloudflare_provider_when_it_is_built_then_it_asks_for_an_output_budget_beyond_the_provider_default(
    monkeypatch,
):
    """User Given the free Cloudflare reasoning model whose own default output limit is 256 tokens When the provider is built Then it requests a budget large enough for reasoning plus a JSON answer."""
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    model = OpenAICompatibleModel.from_provider(
        "cloudflare", ModelConfig("@cf/openai/gpt-oss-120b")
    )
    assert model.config.max_tokens is not None and model.config.max_tokens >= 2048


def test_user_given_a_provider_with_no_choices_when_a_fallback_follows_then_the_chain_moves_on_instead_of_aborting(
    endpoint,
):
    """User Given the first provider returns an empty choices list and a later provider responds When the chain runs Then the later provider's answer is returned."""
    endpoint.responses.put((200, {"choices": []}, {}))
    endpoint.responses.put((200, completion("Recovered."), {}))
    chain = FallbackChatModel(
        adapter(endpoint, model_name="primary-model"),
        adapter(endpoint, model_name="fallback-model"),
    )
    assert chain.chat_completion([{"role": "user", "content": "Continue."}], max_retries=1) == "Recovered."


def test_user_given_an_answer_rejected_by_validation_when_the_provider_retries_then_the_model_is_told_what_was_wrong(
    endpoint,
):
    """User Given a model whose first answer fails the caller's validation When the provider retries Then the retry shows the model its rejected answer and the reason, and the corrected answer is returned."""
    endpoint.responses.put((200, completion('{"count": 3}'), {}))
    endpoint.responses.put((200, completion('{"count": 4}'), {}))

    def must_be_even(text):
        if json.loads(text)["count"] % 2:
            raise ValueError("count must be even")

    result = adapter(endpoint).chat_completion(
        [{"role": "user", "content": "Pick a count."}],
        max_retries=2,
        initial_retry_delay=0,
        _response_validator=must_be_even,
    )

    assert json.loads(result)["count"] == 4
    retry_messages = endpoint.requests[1]["payload"]["messages"]
    retry_text = " ".join(str(message["content"]) for message in retry_messages)
    assert "count must be even" in retry_text and '{"count": 3}' in retry_text
    assert retry_messages[0]["content"] == "Pick a count."


def test_user_given_repeated_validation_rejections_when_the_provider_keeps_retrying_then_only_the_latest_rejection_is_shown(
    endpoint,
):
    """User Given three answers rejected for different reasons When the provider retries Then each retry carries only the previous rejection, so the conversation does not grow without bound."""
    for count in (1, 3, 4):
        endpoint.responses.put((200, completion(json.dumps({"count": count})), {}))

    def must_be_even(text):
        count = json.loads(text)["count"]
        if count % 2:
            raise ValueError(f"count {count} must be even")

    adapter(endpoint).chat_completion(
        [{"role": "user", "content": "Pick a count."}],
        max_retries=3,
        initial_retry_delay=0,
        _response_validator=must_be_even,
    )

    last_request = " ".join(
        str(message["content"]) for message in endpoint.requests[2]["payload"]["messages"]
    )
    assert "count 3 must be even" in last_request
    assert "count 1 must be even" not in last_request
