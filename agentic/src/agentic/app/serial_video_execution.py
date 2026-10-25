"""Run serial CLI episodes through one shared completion contract."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def run_serial_episodes(
    *,
    episode_count: int,
    arguments_for_episode: Callable[[int], Sequence[str]],
    serial_runner: Callable[[Sequence[str]], Any],
) -> dict[str, Any]:
    episodes_started = 0
    automation_decisions: list[dict[str, Any]] = []

    for episode_index in range(episode_count):
        run_result = serial_runner(arguments_for_episode(episode_index))
        if not isinstance(run_result, dict):
            return {
                "status": "failed",
                "generation_type": "serial_video",
                "episodes_started": episodes_started,
                "reason": "The serial runner ended without a completion status.",
            }

        workflow_stage = str(run_result.get("workflow_stage") or "").strip()
        if workflow_stage == "serial_onboarding_blocked":
            return {
                "status": "blocked",
                "generation_type": "serial_video",
                "episodes_started": episodes_started,
                "onboarding_stage": str(run_result.get("onboarding_stage") or "unknown"),
                "series_id": str(run_result.get("series_id") or ""),
                "reason": str(run_result.get("reason") or "Discord review did not complete."),
            }

        status = str(run_result.get("status") or "").strip().lower()
        if status in {"blocked", "failed"}:
            return {
                "status": status,
                "generation_type": "serial_video",
                "episodes_started": episodes_started,
                "automation_decisions": automation_decisions,
                "reason": str(run_result.get("reason") or "Serial video execution did not complete."),
            }
        if status != "success":
            return {
                "status": "failed",
                "generation_type": "serial_video",
                "episodes_started": episodes_started,
                "reason": "The serial runner returned an unknown completion status.",
            }

        mode = str(run_result.get("automation_mode") or "").strip()
        if mode:
            automation_decisions.append(
                {
                    "mode": mode,
                    "series_id": str(run_result.get("series_id") or ""),
                    "episode_number": run_result.get("episode_number"),
                }
            )

        if workflow_stage == "story_onboarding":
            return {
                "status": status,
                "generation_type": "serial_video",
                "episodes_started": episodes_started,
                "automation_decisions": automation_decisions,
                "onboarding_stage": str(run_result.get("onboarding_stage") or "story_draft"),
                "onboarding_series_id": str(run_result.get("series_id") or ""),
                "story_plan_id": str(run_result.get("story_plan_id") or ""),
            }

        episodes_started += 1

    return {
        "status": "success",
        "generation_type": "serial_video",
        "episodes_started": episodes_started,
        "automation_decisions": automation_decisions,
    }
