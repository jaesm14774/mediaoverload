from __future__ import annotations

import os
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentic.runtime.llm_manager_adapter import _add_auxiliary_fallbacks
from agentic.runtime.model_backends import (
    FallbackChatModel,
    ModelConfig,
    OpenAICompatibleModel,
    OpenRouterModel,
    OpenRouterRotatingModel,
    ProviderRequestError,
    build_model,
    provider_credentials_present,
    provider_default_model,
    provider_spec,
)


class OpenAICompatibleModelTests(unittest.TestCase):
    def _response(self, content: str = "ok", status_code: int = 200, headers: dict | None = None) -> Mock:
        response = Mock(status_code=status_code, headers=headers or {})
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    def test_provider_registry_contains_three_free_auxiliaries(self) -> None:
        self.assertEqual(provider_default_model("gemini", "vision"), "gemini-3.5-flash")
        self.assertEqual(provider_default_model("groq", "vision"), "qwen/qwen3.6-27b")
        self.assertEqual(provider_default_model("mistral", "text"), "mistral-small-latest")
        for provider in ("gemini", "groq", "mistral"):
            self.assertIn("base_url", provider_spec(provider))
            self.assertTrue(provider_spec(provider)["api_key_env"])

    def test_build_model_uses_common_openai_compatible_adapter(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            model = build_model("groq", ModelConfig(model_name="qwen/qwen3.6-27b"))

        self.assertIsInstance(model, OpenAICompatibleModel)
        self.assertEqual(model.provider_name, "groq")
        self.assertEqual(model.base_url, "https://api.groq.com/openai/v1/chat/completions")

    def test_openrouter_uses_common_adapter_and_preserves_headers(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            model = OpenRouterModel(ModelConfig(model_name="model-a"))

        self.assertIsInstance(model, OpenAICompatibleModel)
        self.assertEqual(model.base_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(model.headers["X-Title"], "MediaOverload")

    def test_chat_completion_omits_incompatible_mistral_schema_parameter(self) -> None:
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
            model = build_model("mistral", ModelConfig(model_name="model-a", temperature=0.2, max_tokens=64))

        with patch("agentic.runtime.model_backends.requests.post", return_value=self._response("hello")) as post:
            result = model.chat_completion(
                [{"role": "user", "content": "Say hello"}],
                response_format={"type": "json_object"},
            )

        self.assertEqual(result, "hello")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "model-a")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertNotIn("response_format", payload)
        self.assertEqual(post.call_args.kwargs["timeout"], (10.0, 30.0))

    def test_groq_omits_json_schema_response_format_for_llama(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            model = build_model("groq", ModelConfig(model_name="llama-3.3-70b-versatile"))

        with patch("agentic.runtime.model_backends.requests.post", return_value=self._response('{"ok":true}')) as post:
            model.chat_completion(
                [{"role": "user", "content": "Return JSON."}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "tiny", "schema": {"type": "object"}},
                },
            )

        self.assertNotIn("response_format", post.call_args.kwargs["json"])

    def test_chat_completion_encodes_images_as_openai_image_url_parts(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            model = build_model("gemini", ModelConfig(model_name="gemini-3.5-flash"))

        with tempfile.NamedTemporaryFile(suffix=".png", dir=Path.cwd(), delete=False) as image_file:
            image_file.write(b"png")
            image_path = image_file.name
        try:
            with patch("agentic.runtime.model_backends.requests.post", return_value=self._response("visual")) as post:
                result = model.chat_completion(
                    [{"role": "user", "content": "Describe it"}],
                    images=[image_path],
                )
        finally:
            Path(image_path).unlink(missing_ok=True)

        self.assertEqual(result, "visual")
        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe it"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_response_validator_retries_malformed_model_output(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=False):
            model = build_model("groq", ModelConfig(model_name="model-a"))

        with patch(
            "agentic.runtime.model_backends.requests.post",
            side_effect=[self._response("<pad>"), self._response('{"ok":true}')],
        ) as post:
            result = model.chat_completion(
                [{"role": "user", "content": "Return JSON."}],
                max_retries=2,
                initial_retry_delay=0,
                _response_validator=json.loads,
            )

        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(post.call_count, 2)

    def test_retry_after_is_honored_for_429(self) -> None:
        rate_limited = self._response(status_code=429, headers={"Retry-After": "0"})
        from requests import HTTPError

        rate_limited.raise_for_status.side_effect = HTTPError(response=rate_limited)
        success = self._response("recovered")
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
            model = build_model("mistral", ModelConfig(model_name="mistral-small-latest"))

        with patch("agentic.runtime.model_backends.requests.post", side_effect=[rate_limited, success]) as post, patch(
            "agentic.runtime.model_backends.time.sleep"
        ) as sleep:
            result = model.chat_completion(
                [{"role": "user", "content": "test"}],
                max_retries=2,
                initial_retry_delay=10,
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_fallback_chain_tries_auxiliaries_in_order(self) -> None:
        primary = Mock()
        primary.chat_completion.side_effect = ProviderRequestError("primary down")
        auxiliary_a = Mock()
        auxiliary_a.chat_completion.side_effect = ProviderRequestError("first auxiliary down")
        auxiliary_b = Mock()
        auxiliary_b.chat_completion.return_value = "ok"

        chain = FallbackChatModel(primary, auxiliary_a, auxiliary_b)
        self.assertEqual(chain.chat_completion([]), "ok")
        self.assertEqual(chain.last_success_model, "")
        self.assertEqual(chain.last_attempt_model, "")
        self.assertEqual(primary.chat_completion.call_count, 1)
        self.assertEqual(auxiliary_a.chat_completion.call_count, 1)
        self.assertEqual(auxiliary_b.chat_completion.call_count, 1)

    def test_openrouter_pool_exhaustion_reaches_auxiliary_fallback(self) -> None:
        with patch.dict(os.environ, {"open_router_token": "test-key"}, clear=False):
            primary = OpenRouterRotatingModel(
                ModelConfig(model_name="m1"),
                ["m1", "m2"],
                random_each_call=False,
            )
        with patch.object(
            primary._inner,
            "chat_completion_single_model",
            side_effect=ProviderRequestError("openrouter down"),
        ) as completion:
            auxiliary = Mock()
            auxiliary.chat_completion.return_value = "fallback-ok"
            chain = FallbackChatModel(primary, auxiliary)

            self.assertEqual(chain.chat_completion([]), "fallback-ok")

        self.assertEqual(completion.call_count, 2)
        self.assertEqual(auxiliary.chat_completion.call_count, 1)

    def test_fallback_chain_reserves_deadline_for_auxiliary_provider(self) -> None:
        primary = Mock()
        primary.chat_completion.side_effect = ProviderRequestError("primary timed out")
        auxiliary = Mock()
        auxiliary.chat_completion.return_value = "fallback-ok"

        chain = FallbackChatModel(primary, auxiliary)
        deadline = time.monotonic() + 60

        self.assertEqual(chain.chat_completion([], _deadline=deadline), "fallback-ok")
        primary_deadline = primary.chat_completion.call_args.kwargs["_deadline"]
        auxiliary_deadline = auxiliary.chat_completion.call_args.kwargs["_deadline"]
        self.assertLess(primary_deadline, deadline)
        self.assertLessEqual(auxiliary_deadline, deadline)
        self.assertGreater(auxiliary_deadline, primary_deadline)

    def test_missing_auxiliary_key_is_skipped_without_wrapping_primary(self) -> None:
        primary = Mock()
        backend = {
            "allow_text_fallback": True,
            "text_fallback_providers": ["groq", "mistral"],
            "text_fallback_models": ["llama-3.3-70b-versatile", "mistral-small-latest"],
        }
        with patch.dict(os.environ, {}, clear=False):
            with patch("agentic.runtime.llm_manager_adapter.provider_credentials_present", return_value=False):
                result = _add_auxiliary_fallbacks(primary, backend, "text")

        self.assertIs(result, primary)
        self.assertEqual(backend["text_fallback_skipped"], ["groq:missing_api_key", "mistral:missing_api_key"])


if __name__ == "__main__":
    unittest.main()
