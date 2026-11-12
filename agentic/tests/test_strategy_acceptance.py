from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

from agentic.runtime.observability import RunRecorder

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "strategy_acceptance.py"
spec = importlib.util.spec_from_file_location("strategy_acceptance_script", SCRIPT)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def _run(root: Path, run_id: str, strategy: str, **payload) -> None:
    RunRecorder(root, run_id).finalize({"source_generation_type": strategy, **payload})


def _published(root, run_id, strategy):
    _run(root, run_id, strategy, status="success", stage_status={"publish": {"status": "success"}})


def _regenerate(root, run_id, strategy):
    _run(root, run_id, strategy, status="regeneration_requested", stage_status={"publish": {"status": "regeneration_requested"}})


def _rejected(root, run_id, strategy):
    _run(
        root, run_id, strategy, status="failed",
        failure_reason="Blocked workflow because the Discord reviewer rejected the candidates.",
        stage_status={"publish": {"status": "failed"}},
    )


def test_user_given_reviewed_runs_when_acceptance_is_tallied_then_each_strategy_shows_published_over_human_decisions():
    """User Given finished runs of two strategies When acceptance is tallied Then each strategy's rate is its published runs over the runs a human decided, and regenerate counts as a decline."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for index in range(4):
            _published(root, f"a{index}", "story_card")
        _regenerate(root, "a4", "story_card")
        _published(root, "b0", "text2img")
        _rejected(root, "b1", "text2img")
        _regenerate(root, "b2", "text2img")

        rows = {row["strategy"]: row for row in acceptance.tally(root)}

    assert rows["story_card"]["accepted"] == 4 and rows["story_card"]["declined"] == 1
    assert rows["story_card"]["rate"] >= 0.8
    assert rows["text2img"]["accepted"] == 1 and rows["text2img"]["declined"] == 2
    assert rows["text2img"]["rate"] < 0.8


def test_user_given_runs_nobody_judged_when_acceptance_is_tallied_then_they_do_not_move_the_rate():
    """User Given a run that crashed and a review that timed out When acceptance is tallied Then both are listed as undecided and the rate only reflects human decisions."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _published(root, "a0", "motion_comic")
        _run(root, "a1", "motion_comic", status="failed", failure_reason="RuntimeError: ComfyUI is not reachable")
        _run(
            root, "a2", "motion_comic", status="failed",
            failure_reason="Blocked workflow because Discord review did not complete successfully.",
            stage_status={"publish": {"status": "failed"}},
        )

        (row,) = acceptance.tally(root)

    assert (row["accepted"], row["declined"], row["undecided"]) == (1, 0, 2)
    assert row["rate"] == 1.0
