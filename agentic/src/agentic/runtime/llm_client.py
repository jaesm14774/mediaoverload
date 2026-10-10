"""Shared LLM lifecycle, provider selection and response provenance."""

from __future__ import annotations

import json
from typing import Any, Callable
from agentic.runtime.llm_manager_adapter import build_llm_manager
from agentic.runtime.llm_config import resolve_backend_info
from agentic.runtime.structured_chat import chat_json, model_id
from agentic.runtime.observability import RunRecorder
from agentic.runtime.prompt_requests import JsonChatRequest
from agentic.runtime.prompting import ENGLISH_GENERATION_RESPONSE_CONTRACT


class PromptGenerationError(RuntimeError):
    """Raised when a prompt-producing step cannot complete with an LLM."""


class PromptClient:
    def __init__(
        self,
        mode: str = "auto",
        manager: Any | None = None,
        recorder: RunRecorder | None = None,
        *,
        manager_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.mode = mode
        self._manager = manager
        self.recorder = recorder
        self._backend_info: dict[str, Any] | None = None
        self._manager_error: str | None = None
        self._manager_factory = manager_factory or build_llm_manager

    def mark_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["prompt_mode"] = "llm"
        enriched["llm_backend"] = self.backend_info()
        model_id = self.current_model_id("text")
        if model_id:
            enriched["llm_model"] = model_id
        return enriched

    def backend_info(self) -> dict[str, Any]:
        if self._backend_info is None:
            self._backend_info = resolve_backend_info(self.mode)
        enriched = dict(self._backend_info)
        if self._manager_error:
            enriched["manager_error"] = self._manager_error
        if self._manager is not None:
            tm = self._manager.text_model
            primary = getattr(tm, "_primary", tm)
            last = getattr(primary, "last_success_model", "") or getattr(
                tm, "last_success_model", ""
            )
            if last:
                enriched["openrouter_last_text_model"] = last
        return enriched

    def template_fallback(
        self,
        payload: dict[str, Any],
        exc: Exception | None = None,
        *,
        fallback_reason: str = "manager_unavailable",
    ) -> dict[str, Any]:
        fallback = dict(payload)
        fallback["prompt_mode"] = "template"
        fallback["fallback_reason"] = fallback_reason
        fallback["llm_backend"] = self.backend_info()
        if exc is not None:
            fallback["manager_error"] = f"{type(exc).__name__}: {exc}"
        elif self._manager_error:
            fallback["manager_error"] = self._manager_error
        return fallback

    def manager_or_none(self) -> Any | None:
        if self._manager is not None:
            self.attach_recorder(self._manager)
            return self._manager
        if self.mode == "template":
            self._manager_error = (
                "LLM prompt generation is required but AGENTIC_LLM_MODE=template."
            )
            return None
        try:
            backend = self.backend_info()
            self._manager = self._manager_factory(backend)
            # The manager adds resolved/skipped fallback candidates during
            # construction; retain those fields for run diagnostics.
            self._backend_info = dict(backend)
            self.attach_recorder(self._manager)
            self._manager_error = None
            return self._manager
        except Exception as exc:
            self._manager_error = f"{type(exc).__name__}: {exc}"
            return None

    def require_manager(self) -> Any:
        manager = self.manager_or_none()
        if manager is not None:
            return manager
        raise self.generation_error("manager_initialization")

    def attach_recorder(self, manager: Any) -> None:
        if self.recorder is None:
            return
        try:
            setattr(manager, "_mediaoverload_run_recorder", self.recorder)
        except Exception:
            # Some third-party manager wrappers use slots. Prompt generation
            # must remain functional even when observability cannot attach.
            return

    def generation_error(
        self, operation: str, exc: Exception | None = None
    ) -> PromptGenerationError:
        backend = self.backend_info()
        details = self._manager_error
        if exc is not None:
            details = f"{type(exc).__name__}: {exc}"
        message = f"LLM prompt generation failed during {operation}."
        if details:
            message = f"{message} {details}"
        message = f"{message} Backend={json.dumps(backend, ensure_ascii=False, sort_keys=True)}"
        return PromptGenerationError(message)

    def generate_json(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["recorder"] = self.recorder
        kwargs.setdefault("language_contract", ENGLISH_GENERATION_RESPONSE_CONTRACT)
        return chat_json(JsonChatRequest(*args, **kwargs))

    def current_model_id(self, modality: str) -> str:
        manager = self._manager
        if manager is None:
            return ""
        model = manager.vision_model if modality == "vision" else manager.text_model
        return model_id(model)
