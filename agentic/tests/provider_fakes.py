"""Scripted responses at the public ChatModel provider boundary.

The real HTTP adapter contract is exercised by test_provider_http_contract; these
providers use the same messages/images/options -> response-text interface.
"""

from __future__ import annotations

import json
from typing import Any


class ScriptedChatModel:
    def __init__(self, responses: list[Any], *, repeat_last: bool = False) -> None:
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.requests: list[dict[str, Any]] = []

    def chat_completion(
        self, messages: list[dict], images: list[str] | None = None, **kwargs
    ) -> str:
        self.requests.append(
            {"messages": messages, "images": images, "options": kwargs}
        )
        if not self.responses:
            raise RuntimeError("Scripted provider has no response for this request")
        response = self.responses[0]
        if len(self.responses) > 1 or not self.repeat_last:
            self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return (
            response
            if isinstance(response, str)
            else json.dumps(response, ensure_ascii=False)
        )


class ScriptedLLMManager:
    def __init__(self, responses: list[Any], *, repeat_last: bool = False) -> None:
        self.text_model = ScriptedChatModel(responses, repeat_last=repeat_last)
        self.vision_model = self.text_model
