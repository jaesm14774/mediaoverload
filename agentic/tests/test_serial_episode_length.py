import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from agentic.app.main import _apply_series_look
from agentic.app.serial_look import serial_series_look
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.planner import TaskPlanner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIRBY_CONFIG = PROJECT_ROOT.parent / "configs" / "characters" / "kirby.yaml"


def _planned_segments(duration_seconds: int) -> list[str]:
    planner = TaskPlanner(
        asset_registry=AssetRegistry(PROJECT_ROOT, asset_root=PROJECT_ROOT)
    )
    with TemporaryDirectory() as temporary:
        identity = Path(temporary) / "identity.png"
        identity.write_bytes(b"identity")
        goal = GoalRequest(
            prompt="Kirby follows a hum into a cave",
            media_type="long_video",
            duration_seconds=duration_seconds,
            style="watercolor",
            constraints={
                "serial_mode": True,
                "series_id": "kirby-drama",
                "character": "Kirby",
                "reference_manifest": [
                    {"path": str(identity), "type": "image", "role": "subject"}
                ],
                "longvideo_review_policy": "every_segment",
                "require_human_review": True,
            },
        )
        plan = planner.build_plan(goal)
    return sorted(node.node_id for node in plan.nodes if node.node_id.startswith("segment-review-"))


@pytest.mark.parametrize(
    ("duration_seconds", "clips"), [(15, 1), (30, 2), (60, 4), (180, 12)]
)
def test_user_given_an_episode_length_when_it_is_planned_then_it_uses_the_fewest_clips_that_cover_it(
    duration_seconds, clips
):
    """User Given an episode of a given length When it is planned Then it is cut into the fewest clips H3 can render, so a one-minute episode is four clips with three hand-offs rather than six clips with five, and the character has fewer chances to drift."""
    assert len(_planned_segments(duration_seconds)) == clips


def test_user_given_a_series_when_any_episode_runs_then_it_uses_the_same_look_the_references_were_made_in():
    """User Given a series whose references were made in its locked look When one of its episodes runs Then the episode's style and canvas are that same look."""
    generation = yaml.safe_load(KIRBY_CONFIG.read_text(encoding="utf-8"))["generation"]
    args = argparse.Namespace(character="Kirby", character_config=str(KIRBY_CONFIG))

    _apply_series_look(args, {"series_id": "auto-kirby-series-7"})

    look = serial_series_look(generation, "auto-kirby-series-7")
    assert args.style == look["style"]["label"]
    assert (args.width, args.height) == (look["canvas"]["width"], look["canvas"]["height"])
    assert args.serial_style_profile == look["style"]
