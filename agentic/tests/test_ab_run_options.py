"""User scenarios for running two arms (free models vs Claude) on the same article and picks."""
from __future__ import annotations

import os

import pytest

import run_media_interface
from agentic.runtime.claude_brain import BRAIN_DIR_ENV
from agentic.tools.context_services import NewsContextService


class Boundary:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return {"status": "success", "run_id": f"run-{len(self.requests)}"}


def test_user_given_a_routing_seed_when_run_twice_then_both_runs_make_the_same_random_picks(capsys):
    """User Given the same --routing-seed When two runs start Then their routing randomness is identical."""
    first, second = Boundary(), Boundary()
    argv = ["--character", "kirby", "--routing-seed", "7"]
    run_media_interface.main(argv, workflow_runner=first)
    run_media_interface.main(argv, workflow_runner=second)
    picks = [[request.generation.rng.random() for _ in range(5)] for request in (first.requests[0], second.requests[0])]
    assert picks[0] == picks[1]


def test_user_given_claude_brain_when_run_starts_then_llm_calls_are_routed_to_that_directory(tmp_path, monkeypatch, capsys):
    """User Given --claude-brain DIR When the run starts Then every LLM call is answered through DIR."""
    monkeypatch.setenv(BRAIN_DIR_ENV, "")  # main() sets the variable; this makes pytest remove it afterwards
    run_media_interface.main(["--character", "kirby", "--claude-brain", str(tmp_path / "brain")], workflow_runner=Boundary())
    assert os.environ[BRAIN_DIR_ENV] == str((tmp_path / "brain").resolve())


def test_user_given_a_news_url_for_the_loop_strategy_when_run_then_it_is_refused(capsys):
    """User Given a story that invents its own premise When --news-url is supplied Then the CLI refuses instead of attaching news."""
    with pytest.raises(SystemExit):
        run_media_interface.main(
            ["--character", "kirby", "--generation-type", "native_h3_loop_story", "--news-url", "https://example.test/a"],
            workflow_runner=Boundary(),
        )


@pytest.mark.integration
def test_user_given_a_news_url_when_run_then_that_exact_article_drives_the_run(capsys):
    """User Given an article stored in the news database When --news-url names it Then the run is news-driven with that article's full text."""
    service = NewsContextService()
    if not service.is_configured():
        pytest.skip("news database is not configured")
    picked = service.select_news()
    assert picked is not None
    workflow = Boundary()
    run_media_interface.main(
        ["--character", "kirby", "--generation-type", "story_card", "--news-url", picked.url],
        workflow_runner=workflow,
    )
    generation = workflow.requests[0].generation
    assert generation.news_driven
    assert generation.news_context["url"] == picked.url
    assert generation.news_context["content"]
