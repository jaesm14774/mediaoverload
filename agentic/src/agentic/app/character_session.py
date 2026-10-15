"""User-directed full reruns shared by manual and scheduled entry points."""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from agentic.app.character_requests import CharacterWorkflowRequest
from agentic.app.character_workflow import run_character_workflow


def run_character_session(
    request: CharacterWorkflowRequest,
    *,
    workflow_runner: Callable[[CharacterWorkflowRequest], dict[str, Any]] = run_character_workflow,
) -> dict[str, Any]:
    request = replace(request, review=replace(request.review, allow_regenerate=True))
    while True:
        result = workflow_runner(request)
        if str(result.get("status") or "").strip().lower() != "regeneration_requested":
            return result
        logging.getLogger(os.getenv("AGENTIC_RUN_LOGGER_NAME") or __name__).info(
            "workflow.regenerate | previous_run_id=%s | config=%s",
            result.get("run_id", ""), request.config_path,
        )
        request = replace(request, generation=replace(request.generation, seed=None))
