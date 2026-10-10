"""Structured response contracts at the provider boundary; no live LLM required."""

from types import SimpleNamespace
import json

import pytest

from agentic.runtime.prompt_requests import JsonChatRequest
from agentic.runtime.structured_chat import chat_json
from agentic.runtime.observability import RunRecorder
from provider_fakes import ScriptedLLMManager


class ResponseModel:
    """Deterministic ChatModel; provider HTTP tests verify this same interface."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def chat_completion(self, messages, **kwargs):
        self.messages = messages
        return next(self.responses)


def test_user_given_a_new_schema_when_requesting_json_then_language_is_explicit():
    """User Given a new schema unrelated to any existing strategy When requesting Traditional Chinese Then the caller's language contract reaches the provider unchanged."""
    model = ResponseModel('{"text":"種子落在手心。"}')
    result = chat_json(
        JsonChatRequest(
            SimpleNamespace(text_model=model),
            "Write a line.",
            "A seed lands safely.",
            "new_strategy_v1",
            {"type": "object"},
            language_contract="Use Traditional Chinese.",
        )
    )
    assert result["text"] == "種子落在手心。"
    assert all(
        "Use Traditional Chinese." in message["content"] for message in model.messages
    )
    assert all("English only" not in message["content"] for message in model.messages)


def test_user_given_invalid_domain_output_when_repairing_then_the_declared_validator_controls_acceptance():
    """User Given valid JSON with invalid domain content When repairing the response Then the caller's validator accepts only the repaired result."""
    model = ResponseModel('{"count":-1}', '{"count":2}')

    def validate(payload):
        if payload["count"] < 0:
            raise ValueError("count must be non-negative")
        return payload

    result = chat_json(
        JsonChatRequest(
            SimpleNamespace(text_model=model),
            "Plan a batch.",
            "Choose a count.",
            "new_batch_v1",
            {"type": "object"},
            validate_response=validate,
            repair_attempts=1,
        )
    )
    assert result["count"] >= 0


def test_user_given_an_array_for_an_object_contract_when_reading_json_then_it_is_rejected():
    """User Given an object response contract When the provider returns an array Then the incompatible response fails without being passed to the strategy."""
    with pytest.raises(ValueError, match="object"):
        chat_json(
            JsonChatRequest(
                SimpleNamespace(text_model=ResponseModel("[]")),
                "Plan.",
                "Return a plan.",
                "new_object_v1",
                {"type": "object"},
                repair_attempts=0,
            )
        )


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"text":"ready"}\n```',
        '```\n{"text":"ready"}\n```',
        '<think>analysis</think>{"text":"ready"}',
        '<think>{"text":"ready"}',
        'Result: {"text":"ready","nested":[{"value":"a } bracket and \\"quote\\""}]} End.',
    ],
)
def test_user_given_provider_formatting_when_json_is_read_then_the_complete_payload_is_recovered(
    response,
):
    """User Given provider formatting around a valid payload When a task reads JSON Then the complete value survives including nested structures and quoted delimiters."""
    result = chat_json(
        JsonChatRequest(
            ScriptedLLMManager([response]),
            "Write.",
            "Return text.",
            "formatting_v1",
            {"type": "object"},
            repair_attempts=0,
        )
    )
    assert result["text"] == "ready"


@pytest.mark.parametrize("response", ["", "{", "{]", "prefix [ } trailing"])
def test_user_given_incomplete_json_when_no_repair_is_allowed_then_no_partial_result_is_accepted(
    response,
):
    """User Given an empty or structurally broken response When the task forbids repair Then parsing fails instead of returning partial content."""
    with pytest.raises(json.JSONDecodeError):
        chat_json(
            JsonChatRequest(
                ScriptedLLMManager([response]),
                "Write.",
                "Return text.",
                "broken_v1",
                {"type": "object"},
                repair_attempts=0,
            )
        )


def test_user_given_an_object_for_an_array_contract_when_json_is_read_then_the_task_rejects_it():
    """User Given an array contract When the provider returns an object Then generation fails with the expected type error."""
    with pytest.raises(ValueError, match="array"):
        chat_json(
            JsonChatRequest(
                ScriptedLLMManager([{}]),
                "List.",
                "Return items.",
                "array_v1",
                {"type": "array"},
                repair_attempts=0,
            )
        )


def test_user_given_failed_provider_repair_when_generation_stops_then_original_failure_and_evidence_are_retained(
    tmp_path,
):
    """User Given malformed JSON followed by a provider outage When repair fails Then diagnostics retain the original error and record the failed responses."""
    recorder = RunRecorder(tmp_path, "failure-evidence")
    with pytest.raises(json.JSONDecodeError) as failure:
        chat_json(
            JsonChatRequest(
                ScriptedLLMManager(["not JSON", TimeoutError("provider timed out")]),
                "Write.",
                "Return text.",
                "failure_v1",
                {"type": "object"},
                recorder=recorder,
                repair_attempts=1,
            )
        )
    assert "provider timed out" in " ".join(failure.value.__notes__)
    evidence = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in recorder.run_dir.joinpath("llm").glob("*.json")
    ]
    assert evidence and all(call["status"] == "failed" for call in evidence)
    assert any("provider timed out" in call["error"] for call in evidence)
