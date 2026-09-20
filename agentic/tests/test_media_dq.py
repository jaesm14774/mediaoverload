from pathlib import Path

from PIL import Image
import pytest

from agentic.assets.image_input import inspect_image_input
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.media_dq import check_image_contract, expected_subject_count, validate_subject_counts
from agentic.runtime.prompt_engine import PromptEngine
from agentic.skills.editing import EditingSkills
from agentic.tools.media_services import MediaServiceTools
from agentic.runtime.registry import ToolRegistry


@pytest.mark.parametrize("counts", [[True], ["1"], [None], [], [1, 1], [-1]])
def test_user_given_subject_count_evidence_is_malformed_when_validating_then_it_is_rejected(counts):
    """User Given subject-count evidence is malformed When validating Then it is rejected without coercion."""

    assert validate_subject_counts({"counts": counts}, 1, 1)["passed"] is False


def test_user_given_exact_subject_counts_when_validating_then_only_the_explicit_numeric_contract_passes():
    """User Given exact subject-count evidence When validating Then only the explicit numeric contract passes."""

    assert validate_subject_counts({"counts": [2, 2], "score": 0, "status": "fail"}, 2, 2)["passed"] is True
    assert validate_subject_counts({"counts": [2, 3]}, 2, 2)["passed"] is False


def test_user_given_character_name_without_a_count_when_resolving_expected_subjects_then_the_count_is_not_guessed():
    """User Given only a character name When resolving expected subjects Then no count is guessed."""

    assert expected_subject_count({"character": "Kirby"}) is None
    assert expected_subject_count({"expected_subject_count": 2}) == 2
    assert expected_subject_count({"expected_subject_count": 0}) == 0
    with pytest.raises(ValueError):
        expected_subject_count({"expected_subject_count": True})


def test_user_given_image_files_when_checking_the_media_contract_then_decode_dimensions_and_ratio_are_enforced(tmp_path):
    """User Given image files When checking the media contract Then decoding, dimensions, and ratio are enforced."""

    path = tmp_path / "frame.png"
    Image.new("RGB", (320, 180), "blue").save(path)

    assert check_image_contract(str(path), {"canvas_aspect_ratio": "16:9"})["passed"]
    assert not check_image_contract(str(path), {"canvas_width": 321})["passed"]
    assert not check_image_contract(str(path), {"canvas_aspect_ratio": "4:5"})["passed"]
    path.write_bytes(b"corrupt")
    assert not check_image_contract(str(path), {})["passed"]


def test_user_given_h3_input_when_the_file_is_decodable_then_only_technical_input_rules_apply(tmp_path):
    """User Given H3 image input When the file is checked Then only technical input rules apply."""

    path = tmp_path / "example.png"
    Image.new("RGB", (608, 352), "blue").save(path)
    assert inspect_image_input(path).passed
    Image.new("RGB", (16, 16), "pink").save(path)
    assert not inspect_image_input(path).passed


@pytest.mark.parametrize("technical_passed", [True, False])
def test_user_given_an_edit_candidate_when_hard_video_qa_runs_then_materialization_follows_the_qa_result(
    tmp_path, technical_passed
):
    """User Given an edit candidate When hard video QA runs Then materialization follows the QA result."""

    tools = ToolRegistry()

    def compose(_payload):
        return {"video_path": str(tmp_path / "edited.mp4"), "manifest_path": "", "contact_sheet_path": ""}

    def video_qa(_payload):
        return {"passed": technical_passed}

    def materialize(_payload):
        return {"video_path": str(tmp_path / "edited.mp4"), "manifest_path": "", "contact_sheet_path": ""}

    tools.register("media.compose_edit", compose, "compose")
    tools.register("media.video_qa", video_qa, "qa")
    tools.register("media.materialize_edit", materialize, "materialize")
    context = SkillContext(
        plan=ExecutionPlan(
            goal=GoalRequest(prompt="edit", media_type="image_sequence_edit"), workflow_name="test", nodes=[]
        ),
        node=ExecutionNode(
            node_id="edit",
            skill_name="media.compose_edit",
            inputs={"input_paths": ["a.mp4", "b.mp4"], "profile": "editorial_kinetic_v1"},
        ),
        state=RunState(goal={}, metadata={}, node_outputs={}),
    )

    result = EditingSkills(tools, tmp_path).compose_timeline(context)

    assert result.status == ("success" if technical_passed else "failed")
    assert "creative_review" not in result.outputs


@pytest.mark.parametrize("explicit_limit", [False, True])
def test_user_given_silent_video_when_audio_limit_is_explicit_then_only_that_limit_can_block(tmp_path, explicit_limit):
    """User Given a silent video When an audio limit is explicit Then only that limit can block the result."""

    class FakeFFmpeg:
        def probe_media(self, _path):
            return {
                "has_video": True,
                "has_audio": True,
                "duration": 5,
                "width": 320,
                "height": 180,
                "channels": 2,
            }

        def analyze_audio(self, _path, *, silence_threshold_db, silence_min_seconds):
            del silence_threshold_db, silence_min_seconds
            return {"mean_volume_db": -90, "max_volume_db": -80, "silence_ratio": 1}

    path = tmp_path / "silent.mp4"
    path.write_bytes(b"probe fixture")
    service = MediaServiceTools(tmp_path, ffmpeg=FakeFFmpeg())
    payload = {"video_path": str(path), "require_audio": True, "analyze_audio": not explicit_limit}
    if explicit_limit:
        payload["max_silence_ratio"] = 0.5

    result = service.video_qa(payload)

    assert result["passed"] is not explicit_limit
    assert result["audio_analysis"]["silence_ratio"] == 1


def test_user_given_human_review_notes_when_refining_a_prompt_then_the_review_result_is_returned():
    """User Given human review notes When refining a prompt Then the injected prompt service result is returned."""

    class FakeLLM:
        def refine_prompt_from_review(self, *_args, **_kwargs):
            return {"revised_prompt": "human requested revision"}

    result = PromptEngine(FakeLLM()).refine_prompt_from_review(
        GoalRequest(prompt="art", media_type="image"), "original", "change the hat", []
    )

    assert result == {"revised_prompt": "human requested revision"}
