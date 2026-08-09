from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic.runtime.llm_manager_adapter import _discover_pool
from agentic.runtime.model_backends import (
    ModelConfig,
    OpenRouterRotatingModel,
    static_openrouter_models,
    static_openrouter_model_modes,
)


class OpenRouterStaticPoolTests(unittest.TestCase):
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
            ],
        )
        self.assertEqual(
            static_openrouter_model_modes("vision")["nvidia/nemotron-nano-12b-v2-vl:free"],
            "prompt_only",
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


if __name__ == "__main__":
    unittest.main()
