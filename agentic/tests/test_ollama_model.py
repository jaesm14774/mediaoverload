from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentic.runtime.model_backends import ModelConfig, OllamaModel


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"message": {"content": '{"ok": true}'}}


class OllamaModelTests(unittest.TestCase):
    def test_chat_payload_uses_native_controls_and_base64_images(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENTIC_OLLAMA_THINK": "false",
                "OLLAMA_API_BASE_URL": "http://host.docker.internal:11434/",
                "AGENTIC_OLLAMA_NUM_CTX": "8192",
                "AGENTIC_OLLAMA_NUM_PREDICT": "800",
                "AGENTIC_OLLAMA_TOP_P": "0.8",
                "AGENTIC_OLLAMA_PRESENCE_PENALTY": "1.5",
            },
            clear=False,
        ), patch(
            "agentic.runtime.model_backends.OpenAICompatibleModel._encode_image_to_base64",
            return_value="data:image/png;base64,ZmFrZQ==",
        ), patch(
            "agentic.runtime.model_backends.requests.post",
            return_value=_FakeResponse(),
        ) as post:
            result = OllamaModel(ModelConfig("qwen3.8-27b-ud-q2xl-local")).chat_completion(
                messages=[{"role": "user", "content": "Return JSON."}],
                images=["candidate.png"],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
                request_timeout=12,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(result, '{"ok": true}')
        self.assertEqual(post.call_args.args[0], "http://host.docker.internal:11434/api/chat")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], {"type": "object"})
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_predict"], 800)
        self.assertEqual(payload["messages"][-1]["images"], ["ZmFrZQ=="])
        self.assertEqual(post.call_args.kwargs["timeout"], 12.0)


if __name__ == "__main__":
    unittest.main()
