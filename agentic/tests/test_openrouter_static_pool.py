from __future__ import annotations

import json
import os
import unittest
from unittest.mock import Mock, patch

from agentic.runtime.llm_manager_adapter import _discover_pool
from agentic.runtime.model_backends import (
    ModelConfig,
    OpenRouterRotatingModel,
    OpenRouterModel,
    static_openrouter_models,
    static_openrouter_model_modes,
)


class OpenRouterStaticPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._api_key_patch = patch.dict(
            os.environ,
            {
                "open_router_token": "test-openrouter-key",
                "OPENROUTER_API_KEY": "test-openrouter-key",
                "OPENROUTER_API_TOKEN": "test-openrouter-key",
            },
        )
        self._api_key_patch.start()
        self.addCleanup(self._api_key_patch.stop)

    def test_scheduler_uses_static_pool_without_catalog_request(self) -> None:
        backend = {
            "openrouter_discover_models": False,
            "openrouter_free_pool_size": 0,
            "openrouter_text_models": [],
        }

        with patch(
            "agentic.runtime.llm_manager_adapter.OpenRouterModelCatalog.candidates",
            side_effect=AssertionError("catalog must not be called by scheduler"),
        ):
            pool = _discover_pool(backend, "text")

        self.assertEqual(pool, static_openrouter_models("text"))
        self.assertEqual(backend["openrouter_text_pool_source"], "static_config")

    def test_static_config_contains_all_tested_text_and_vision_routes(self) -> None:
        text = static_openrouter_models("text")
        vision = static_openrouter_models("vision")

        self.assertIn("nvidia/nemotron-3-ultra-550b-a55b:free", text)
        self.assertIn("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", text)
        self.assertNotIn("nvidia/nemotron-nano-9b-v2:free", text)
        self.assertCountEqual(
            vision,
            [
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
                "google/gemma-4-31b-it:free",
            ],
        )

    def test_rotating_adapter_applies_prompt_only_mode(self) -> None:
        model = OpenRouterRotatingModel(
            ModelConfig(model_name="m1"),
            ["m1"],
            model_modes={"m1": "prompt_only"},
        )

        with patch.object(model._inner, "chat_completion_single_model", return_value='{"ok":true}') as call:
            result = model.chat_completion(
                messages=[{"role": "user", "content": "Return JSON."}],
                response_format={"type": "json_object"},
            )

        self.assertEqual(result, '{"ok":true}')
        self.assertNotIn("response_format", call.call_args.kwargs)

    def test_rotating_adapter_applies_reasoning_off_mode(self) -> None:
        model = OpenRouterRotatingModel(
            ModelConfig(model_name="m1"),
            ["m1"],
            model_modes={"m1": "reasoning_off"},
        )

        with patch.object(model._inner, "chat_completion_single_model", return_value='{"ok":true}') as call:
            model.chat_completion(messages=[{"role": "user", "content": "Return JSON."}])

        self.assertEqual(call.call_args.kwargs["reasoning"], {"enabled": False})

    def test_rotating_adapter_can_bound_candidates_per_call(self) -> None:
        model = OpenRouterRotatingModel(
            ModelConfig(model_name="m1"),
            ["m1", "m2", "m3"],
            random_each_call=False,
        )

        with patch.object(model._inner, "chat_completion_single_model", return_value='{"ok":true}') as call:
            model.chat_completion(
                messages=[{"role": "user", "content": "Return JSON."}],
                max_models_per_call=1,
                request_timeout=20,
            )

        self.assertEqual(call.call_args.args[0], "m1")
        self.assertEqual(call.call_args.kwargs["request_timeout"], 20)

    def test_rotating_adapter_moves_to_next_model_on_invalid_validated_output(self) -> None:
        model = OpenRouterRotatingModel(
            ModelConfig(model_name="m1"),
            ["m1", "m2"],
            random_each_call=False,
        )
        calls: list[str] = []

        def fake_completion(model_name: str, *args: object, **kwargs: object) -> str:
            del args
            calls.append(model_name)
            response = "<pad>" if model_name == "m1" else '{"ok":true}'
            validator = kwargs.get("_response_validator")
            if callable(validator):
                try:
                    validator(response)
                except Exception as exc:
                    from agentic.runtime.model_backends import ProviderRequestError

                    raise ProviderRequestError(str(exc)) from exc
            return response

        with patch.object(model._inner, "chat_completion_single_model", side_effect=fake_completion):
            result = model.chat_completion(
                messages=[{"role": "user", "content": "Return JSON."}],
                _response_validator=json.loads,
            )

        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(calls, ["m1", "m2"])

    def test_explicit_pool_keeps_models_in_verified_static_pool(self) -> None:
        backend = {
            "openrouter_discover_models": False,
            "openrouter_vision_pool_mode": True,
            "openrouter_vision_models": [
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
            ],
        }

        pool = _discover_pool(backend, "vision")

        self.assertEqual(
            pool,
            [
                "google/gemma-4-26b-a4b-it:free",
                "nvidia/nemotron-nano-12b-v2-vl:free",
            ],
        )
        self.assertEqual(backend["openrouter_vision_pool_source"], "env_static_list_filtered")

    def test_single_model_retries_429_using_retry_after(self) -> None:
        rate_limited = Mock(status_code=429, headers={"Retry-After": "0"})
        rate_limited.raise_for_status.side_effect = __import__("requests").HTTPError(response=rate_limited)
        success = Mock(status_code=200, headers={})
        success.raise_for_status.return_value = None
        success.json.return_value = {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
        model = OpenRouterModel(ModelConfig(model_name="m1"))

        with patch("agentic.runtime.model_backends.requests.post", side_effect=[rate_limited, success]) as post, patch(
            "agentic.runtime.model_backends.time.sleep"
        ) as sleep:
            result = model.chat_completion_single_model(
                "m1",
                [{"role": "user", "content": "Return JSON."}],
                max_retries=2,
                initial_retry_delay=30,
            )

        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
