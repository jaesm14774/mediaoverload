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
    return sorted(
        node.node_id
        for node in _plan(duration_seconds).nodes
        if node.node_id.startswith("segment-review-")
    )


def _plan(duration_seconds: int):
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
        return planner.build_plan(goal)


@pytest.mark.parametrize(
    ("duration_seconds", "clips"), [(10, 1), (30, 3), (40, 4), (60, 6)]
)
def test_user_given_a_short_episode_when_it_is_planned_then_every_clip_is_ten_seconds_and_carries_one_beat(
    duration_seconds, clips
):
    """User Given a 30-40 second episode When it is planned Then it is cut into ten-second clips, so the script has exactly one beat per clip and no clip has to stretch a thin beat over fifteen seconds."""
    plan = _plan(duration_seconds)

    assert len(_planned_segments(duration_seconds)) == clips
    assert plan.metadata["segment_count"] == clips
    render_lengths = {
        node.inputs["length"]
        for node in plan.nodes
        if node.node_id.startswith("segment-video-")
    }
    assert render_lengths == {240}


def test_user_given_a_series_when_any_episode_runs_then_it_uses_the_same_look_the_references_were_made_in(tmp_path):
    """User Given a series whose references were made in its locked look When one of its episodes runs Then the episode's style and canvas are that same look."""
    generation = yaml.safe_load(KIRBY_CONFIG.read_text(encoding="utf-8"))["generation"]
    args = argparse.Namespace(character="Kirby", character_config=str(KIRBY_CONFIG))

    _apply_series_look(args, {"series_id": "auto-kirby-series-7"}, tmp_path)

    look = serial_series_look(generation, "auto-kirby-series-7", output_root=tmp_path)
    assert args.style == look["style"]["label"]
    assert (args.width, args.height) == (look["canvas"]["width"], look["canvas"]["height"])
    assert args.serial_style_profile == look["style"]
