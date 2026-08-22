from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentic.app.character_requests import CharacterGenerationOptions, CharacterReviewOptions, CharacterRuntimeOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import build_goal_payload_from_character_config


pytestmark = pytest.mark.integration


def test_formal_news_driven_payload_uses_live_provider(tmp_path: Path) -> None:
    if os.environ.get("AGENTIC_FORMAL_INTEGRATION") != "1":
        pytest.fail("Formal integration tests require AGENTIC_FORMAL_INTEGRATION=1")

    repo_root = Path(__file__).resolve().parents[3]
    history_path = tmp_path / "news-selection" / "kirby.json"
    output_dir = tmp_path / "output"
    payload = build_goal_payload_from_character_config(
        CharacterWorkflowRequest(
            repo_root=repo_root,
            config_path=repo_root / "configs" / "characters" / "kirby.yaml",
            generation=CharacterGenerationOptions(
                preferred_generation_type="text2video",
                news_driven=True,
                news_history_path=str(history_path),
                output_dir=str(output_dir),
            ),
            review=CharacterReviewOptions(publish_after_generate=False),
            runtime=CharacterRuntimeOptions(),
        )
    )

    news_context = payload["constraints"]["news_context"]
    assert news_context["title"]
    assert news_context["keyword"]
    assert payload["constraints"]["news_driven"] is True
    assert payload["constraints"]["prompt_mode"] == "llm"
    assert payload["prompt"]

    persisted = json.loads(history_path.read_text(encoding="utf-8"))
    assert persisted[0]["title"] == news_context["title"]
    assert persisted[0]["key"]
