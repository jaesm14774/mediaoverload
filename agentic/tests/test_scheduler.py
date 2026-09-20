from __future__ import annotations

import random
from pathlib import Path

from scheduler.scheduler import (
    REPO_ROOT,
    _run_scheduled_job_safe,
    load_scheduler_config,
    run_scheduled_job,
)


def test_user_given_scheduler_environment_contains_a_character_when_loading_config_then_it_resolves_the_character_contract(monkeypatch):
    """User Given scheduler environment is configured When config loads Then the character contract is resolved."""

    monkeypatch.setenv("SCHEDULER_CHARACTER", "kirby")
    monkeypatch.setenv("SCHEDULER_MODE", "interval")
    monkeypatch.setenv("SCHEDULER_INTERVAL_HOURS", "6")

    config = load_scheduler_config()

    assert config.character == "kirby"
    assert config.interval_hours == 6
    assert config.config_path == REPO_ROOT / "configs" / "characters" / "kirby.yaml"


def test_user_given_scheduler_config_when_running_a_job_then_the_workflow_receives_the_effective_request(monkeypatch):
    """User Given a configured scheduler When a job runs Then the injected workflow receives the effective request."""

    for key, value in {
        "SCHEDULER_CHARACTER": "kirby",
        "SCHEDULER_PROMPT": "Kirby neon short",
        "SCHEDULER_NEWS_DRIVEN": "true",
        "SCHEDULER_NEWS_HISTORY_PATH": "/tmp/kirby-news.json",
        "SCHEDULER_DRY_RUN_PUBLISH": "true",
        "SCHEDULER_ENABLE_REVIEW_LOOP": "false",
    }.items():
        monkeypatch.setenv(key, value)
    config = load_scheduler_config()
    captured = []

    def workflow_runner(request):
        captured.append(request)
        return {"status": "success", "source_generation_type": "text2image2video"}

    result = run_scheduled_job(config, workflow_runner=workflow_runner)

    assert result["status"] == "success"
    assert len(captured) == 1
    request = captured[0]
    assert request.repo_root == REPO_ROOT
    assert request.config_path == Path(config.config_path)
    assert request.generation.prompt == "Kirby neon short"
    assert request.generation.news_driven is True
    assert request.generation.news_history_path == "/tmp/kirby-news.json"
    assert request.review.dry_run_publish is True
    assert request.review.enable_review_loop is False
    assert isinstance(request.generation.rng, random.Random)


def test_user_given_quiet_scheduler_hour_when_running_safely_then_the_job_is_skipped(monkeypatch):
    """User Given the local hour is within quiet hours When the safe runner executes Then it skips the job."""

    monkeypatch.setenv("SCHEDULER_CHARACTER", "kirby")
    config = load_scheduler_config()
    called = []

    def job_runner(*_args, **_kwargs):
        called.append(True)
        return {"status": "success"}

    result = _run_scheduled_job_safe(config, current_hour=2, job_runner=job_runner)

    assert result == {"status": "skipped", "reason": "quiet_hours"}
    assert called == []


def test_user_given_quiet_hours_end_when_running_safely_then_the_job_is_run(monkeypatch):
    """User Given quiet hours have ended When the safe runner executes Then it runs the job."""

    monkeypatch.setenv("SCHEDULER_CHARACTER", "kirby")
    config = load_scheduler_config()
    captured = []

    def job_runner(received_config, *, rng):
        captured.append((received_config, rng))
        return {"status": "success"}

    result = _run_scheduled_job_safe(config, current_hour=6, job_runner=job_runner)

    assert result == {"status": "success"}
    assert captured == [(config, None)]
