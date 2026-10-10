"""Provider-neutral JSON calls, repair, deadlines and recording.

Language and domain validation belong to the caller, never to schema-name
conventions in this transport.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from agentic.runtime.observability import RunRecorder
from agentic.runtime.prompt_requests import JsonChatRequest


def chat_json(request: JsonChatRequest) -> Any:
    manager = request.manager
    system_prompt = request.system_prompt
    user_prompt = request.user_prompt
    schema_name = request.schema_name
    schema = request.schema
    model = request.model
    images = request.images
    recorder = request.recorder
    max_retries = request.max_retries
    request_timeout = request.request_timeout
    max_models_per_call = request.max_models_per_call
    repair_attempts = request.repair_attempts
    use_response_format = bool(request.use_response_format)
    chat_model = manager.vision_model if model == "vision" else manager.text_model
    recorder = recorder or getattr(manager, "_mediaoverload_run_recorder", None)
    expected_json = "JSON array" if schema.get("type") == "array" else "JSON object"
    opening = "[" if expected_json == "JSON array" else "{"
    closing = "]" if expected_json == "JSON array" else "}"
    language_contract = request.language_contract
    system_contract = (
        f"{language_contract}\n\n" if language_contract else ""
    ) + request.output_contract
    user_contract = f"\n\n{language_contract}" if language_contract else ""
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"{system_contract}"
                f"OUTPUT CONTRACT: Return exactly one valid {expected_json} "
                + (
                    "that matches the supplied schema. "
                    if use_response_format
                    else "using the requested fields as a guide; optional fields may be omitted. "
                )
                + "Do not return markdown, code fences, XML tags, explanations, analysis, or a second object. "
                f"Put the JSON value directly in the final answer, starting with {opening} and ending with {closing}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                f"{user_contract}"
                f"\nFINAL FORMAT: output JSON only. Before sending, silently validate that it parses as one complete "
                f"{expected_json}"
                + (
                    " and that all required fields are present."
                    if use_response_format
                    else ". Optional creative metadata may be omitted when it is not useful."
                )
            ),
        },
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }
    first_error: Exception

    def parse_response(response: str) -> Any:
        parsed = parse_json(response)
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(parsed, dict):
            raise ValueError("JSON response must be an object")
        if expected_type == "array" and not isinstance(parsed, list):
            raise ValueError("JSON response must be an array")
        if request.validate_response is not None:
            parsed = request.validate_response(parsed)
        return parsed

    response: Any = None
    chat_options: dict[str, Any] = {}
    if max_retries is not None:
        chat_options["max_retries"] = max(1, int(max_retries))
    if request_timeout is None:
        timeout_raw = os.environ.get(
            "AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS", "30"
        ).strip()
        try:
            request_timeout = max(1.0, float(timeout_raw))
        except ValueError:
            request_timeout = 30.0
    if request_timeout is not None:
        chat_options["request_timeout"] = max(1.0, float(request_timeout))
    if max_models_per_call is not None:
        chat_options["max_models_per_call"] = max(1, int(max_models_per_call))
    total_timeout_raw = os.environ.get(
        "AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS", "180"
    ).strip()
    try:
        total_timeout = max(1.0, float(total_timeout_raw))
    except ValueError:
        total_timeout = 180.0
    chat_options["_deadline"] = time.monotonic() + total_timeout
    model_id_before_call = model_id(chat_model)
    call_path = (
        recorder.start_llm_call(
            schema_name=schema_name,
            attempt=1,
            messages=messages,
            schema=schema,
            model=model,
            model_id=model_id_before_call,
            images=images,
            response_format_used=use_response_format,
        )
        if isinstance(recorder, RunRecorder)
        else None
    )
    try:
        completion_options = {
            "_response_validator": parse_response,
            **chat_options,
        }
        if use_response_format:
            completion_options["response_format"] = response_format
        response = chat_model.chat_completion(
            messages=messages, images=images, **completion_options
        )
        parsed = parse_response(response)
        if isinstance(recorder, RunRecorder) and call_path is not None:
            recorder.complete_llm_call(
                call_path,
                response=response,
                parsed_payload=parsed,
                model_id=model_id(chat_model),
            )
        return parsed
    except Exception as exc:
        first_error = exc
        if isinstance(recorder, RunRecorder) and call_path is not None:
            recorder.complete_llm_call(
                call_path,
                response=response,
                error=f"{type(exc).__name__}: {exc}",
                model_id=model_id(chat_model),
            )

    repair_errors: list[Exception] = []
    for repair_round in range(max(0, int(repair_attempts))):
        repair_messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"{system_contract}"
                    "JSON REPAIR MODE: Your previous answer did not satisfy the response contract. "
                    f"Validation error: {first_error}. "
                    f"Return only one complete {expected_json}"
                    + (
                        " matching the schema"
                        if use_response_format
                        else " using the requested fields as a guide"
                    )
                    + ". No markdown, no code fences, "
                    "no reasoning, no comments, no prose, and no trailing text. "
                    f"This is repair pass {repair_round + 1} of {repair_attempts}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\n"
                    f"{user_contract}"
                    "\nThis is a strict parser repair attempt. Silently correct formatting and missing required "
                    + (
                        "fields, then output the complete "
                        if use_response_format
                        else "JSON syntax, then output the complete "
                    )
                    + f"{expected_json} only."
                ),
            },
        ]
        repaired_response: Any = None
        repair_call_path = (
            recorder.start_llm_call(
                schema_name=schema_name,
                attempt=repair_round + 2,
                messages=repair_messages,
                schema=schema,
                model=model,
                model_id=model_id(chat_model),
                images=images,
                response_format_used=False,
            )
            if isinstance(recorder, RunRecorder)
            else None
        )
        try:
            repaired_response = chat_model.chat_completion(
                messages=repair_messages,
                images=images,
                _response_validator=parse_response,
                **chat_options,
            )
            parsed = parse_response(repaired_response)
            if isinstance(recorder, RunRecorder) and repair_call_path is not None:
                recorder.complete_llm_call(
                    repair_call_path,
                    response=repaired_response,
                    parsed_payload=parsed,
                    model_id=model_id(chat_model),
                )
            return parsed
        except Exception as repair_error:
            repair_errors.append(repair_error)
            if isinstance(recorder, RunRecorder) and repair_call_path is not None:
                recorder.complete_llm_call(
                    repair_call_path,
                    response=repaired_response,
                    error=f"{type(repair_error).__name__}: {repair_error}",
                    model_id=model_id(chat_model),
                )

    for repair_error in repair_errors:
        first_error.add_note(
            f"JSON repair attempt also failed: {type(repair_error).__name__}: {repair_error}"
        )
    raise first_error from (repair_errors[-1] if repair_errors else None)


def model_id(model: Any) -> str:
    """Return the concrete model that handled the latest request."""
    current = model
    visited: set[int] = set()
    for _ in range(5):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        for attribute in ("last_success_model", "last_attempt_model"):
            value = str(getattr(current, attribute, "") or "").strip()
            if value:
                return value
        config = getattr(current, "config", None)
        configured = str(getattr(config, "model_name", "") or "").strip()
        if configured:
            return configured
        current = getattr(current, "_primary", None)
    return ""


def parse_json(response: str) -> Any:
    cleaned = response.strip()
    if not cleaned:
        raise json.JSONDecodeError("Expecting value", cleaned, 0)
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[-1]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        fragment = extract_json_fragment(cleaned)
        if fragment:
            return json.loads(fragment)
        raise


def extract_json_fragment(text: str) -> str | None:
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        stack: list[str] = []
        in_string = False
        escaped = False
        for end in range(index, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
                continue
            if current in "{[":
                stack.append(current)
                continue
            if current == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif current == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif current in "}]":
                break
            if not stack:
                return text[index : end + 1]
    return None
