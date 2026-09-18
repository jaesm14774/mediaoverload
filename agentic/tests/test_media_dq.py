from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from agentic.assets.image_input import inspect_image_input
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine, PromptGenerationError
from agentic.runtime.media_dq import check_image_contract, expected_subject_count, validate_subject_counts
from agentic.skills.editing import EditingSkills
from agentic.skills.agent_social import AgentSocialSkills
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry


@pytest.mark.parametrize("counts", [[True], ["1"], [None], [], [1, 1], [-1]])
def test_subject_count_never_coerces_unknown_or_malformed_evidence(counts):
    assert validate_subject_counts({"counts": counts}, 1, 1)["passed"] is False


def test_exact_count_is_the_only_visual_acceptance_condition():
    payload = {"counts": [2, 2], "score": 0, "status": "fail", "cute_hit": False}
    assert validate_subject_counts(payload, 2, 2)["passed"] is True
    assert validate_subject_counts({"counts": [2, 3]}, 2, 2)["passed"] is False


def test_subject_count_must_be_explicit_not_guessed_from_character_name():
    assert expected_subject_count({"character": "Kirby"}) is None
    assert expected_subject_count({"expected_subject_count": 2}) == 2
    assert expected_subject_count({"expected_subject_count": 0}) == 0
    with pytest.raises(ValueError):
        expected_subject_count({"expected_subject_count": True})


def test_image_contract_checks_decoding_dimensions_and_ratio(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (320, 180), "blue").save(path)
    assert check_image_contract(str(path), {"canvas_aspect_ratio": "16:9"})["passed"]
    assert not check_image_contract(str(path), {"canvas_width": 321})["passed"]
    assert not check_image_contract(str(path), {"canvas_aspect_ratio": "4:5"})["passed"]
    path.write_bytes(b"corrupt")
    assert not check_image_contract(str(path), {})["passed"]


def test_h3_input_has_no_color_identity_or_filename_gate(tmp_path):
    path = tmp_path / "example.png"
    Image.new("RGB", (608, 352), "blue").save(path)
    assert inspect_image_input(path).passed
    Image.new("RGB", (16, 16), "pink").save(path)
    assert not inspect_image_input(path).passed


def test_candidate_selection_has_no_score_threshold_or_editorial_llm(tmp_path):
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        Image.new("RGB", (320, 180)).save(path)
    engine = LLMPromptEngine(mode="template")
    engine._require_manager = Mock(side_effect=AssertionError("No editorial LLM"))
    result = engine.validate_image_candidates(
        GoalRequest(prompt="anything", media_type="image", constraints={"stage_probe_auto_select": True}),
        [str(path) for path in paths], "beautiful or ugly is a human decision", 1,
    )
    assert result["selected_assets"] == [str(paths[0])]
    assert all("score" not in item for item in result["ranked_candidates"])
    engine._require_manager.assert_not_called()


def test_bad_counts_are_excluded_and_unknown_counts_do_not_pass(tmp_path):
    paths = [tmp_path / "extra.png", tmp_path / "right.png"]
    for path in paths:
        Image.new("RGB", (320, 180)).save(path)
    engine = LLMPromptEngine(mode="template")
    engine._require_manager = Mock(return_value=object())
    engine._chat_json_with_recorder = Mock(side_effect=[{"counts": [3]}, {"counts": [2]}])
    goal = GoalRequest(prompt="two subjects", media_type="image", constraints={"expected_subject_count": 2})
    result = engine.validate_image_candidates(goal, [str(path) for path in paths], "", 10)
    assert result["selected_assets"] == [str(paths[1])]
    call = engine._chat_json_with_recorder.call_args
    assert call.kwargs["schema_name"] == "media_subject_counts"
    assert set(call.kwargs["schema"]["properties"]) == {"counts"}
    engine._chat_json_with_recorder = Mock(return_value={"counts": [None]})
    with pytest.raises(PromptGenerationError, match="asset_review_hard_gate"):
        engine.validate_image_candidates(goal, [str(paths[1])], "", 1)


@pytest.mark.parametrize("technical_passed", [True, False])
def test_editing_only_renders_once_and_materializes_after_hard_qa(tmp_path, technical_passed):
    calls = []

    def call(name, payload):
        calls.append(name)
        if name == "media.video_qa":
            return {"passed": technical_passed}
        return {"video_path": str(tmp_path / "edited.mp4"), "manifest_path": "", "contact_sheet_path": ""}

    tools = SimpleNamespace(call=call)
    context = SimpleNamespace(
        node=SimpleNamespace(inputs={"input_paths": ["a.mp4", "b.mp4"], "profile": "editorial_kinetic_v1"}),
        plan=SimpleNamespace(goal=GoalRequest(prompt="edit", media_type="image_sequence_edit")),
    )
    result = EditingSkills(tools, tmp_path).compose_timeline(context)
    assert result.status == ("success" if technical_passed else "failed")
    assert calls == ["media.compose_edit", "media.video_qa"] + (["media.materialize_edit"] if technical_passed else [])
    assert "creative_review" not in result.outputs


@pytest.mark.parametrize("scope", ["first_frame", "final_media", ""])
def test_every_image_review_branch_filters_hard_failures_before_discord(tmp_path, scope):
    good, bad = tmp_path / "wide.png", tmp_path / "square.png"
    Image.new("RGB", (640, 360), "blue").save(good)
    Image.new("RGB", (360, 360), "pink").save(bad)
    skills = AgentSocialSkills(ToolRegistry(), tmp_path)
    skills.prompt_engine.prepare_publish_caption = Mock(return_value={"caption": "caption", "hashtags": ""})
    decision = SimpleNamespace(
        review_mode="discord", status="approved", selected_paths=[str(good)],
        reviewer="human", session_id="test", session_path="", edited_text="", delivery={},
        fallback_reason="", edit_requested=False,
    )
    skills.discord_review.review_candidates = Mock(return_value=decision)
    goal = GoalRequest(prompt="art", media_type="image", constraints={
        "require_human_review": True, "canvas_width": 320, "canvas_height": 180,
    })
    node = ExecutionNode(node_id="review", skill_name="review.assets.select", depends_on=["render"],
                         inputs={"limit": 1, "review_scope": scope, "review_all_candidates": True})
    plan = ExecutionPlan(goal=goal, workflow_name="test", nodes=[node])
    state = RunState(goal={}, metadata={}, node_outputs={"render": {"saved_files": [str(bad), str(good)]}})
    result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))
    assert result.status == "success"
    assert skills.discord_review.review_candidates.call_args.kwargs["media_paths"] == [str(good)]
    assert result.outputs["selected_assets"] == [str(good)]
    assert result.outputs["hard_media_checks"]["rejected_asset_details"][0]["media_path"] == str(bad)


def test_human_rewrite_request_passes_through_without_inventing_failure_tags():
    from agentic.runtime.prompt_engine import PromptEngine
    llm = Mock()
    llm.refine_prompt_from_review.return_value = {"revised_prompt": "human requested revision"}
    engine = PromptEngine(llm)
    result = engine.refine_prompt_from_review(
        GoalRequest(prompt="art", media_type="image"), "original", "change the hat", [],
    )
    assert result == {"revised_prompt": "human requested revision"}


@pytest.mark.parametrize("explicit_limit", [False, True])
def test_silence_only_blocks_when_a_numeric_limit_is_explicit(tmp_path, explicit_limit):
    from agentic.tools.media_services import MediaServiceTools
    path = tmp_path / "silent.mp4"
    path.write_bytes(b"probe fixture")
    service = MediaServiceTools(tmp_path)
    service._ffmpeg = Mock()
    service._ffmpeg.probe_media.return_value = {
        "has_video": True, "has_audio": True, "duration": 5,
        "width": 320, "height": 180, "audio_channels": 2,
    }
    service._ffmpeg.analyze_audio.return_value = {
        "mean_volume_db": -90, "max_volume_db": -80, "silence_ratio": 1,
    }
    payload = {"video_path": str(path), "require_audio": True, "analyze_audio": not explicit_limit}
    if explicit_limit:
        payload["max_silence_ratio"] = 0.5
    result = service.video_qa(payload)
    assert result["passed"] is not explicit_limit
    assert result["audio_analysis"]["silence_ratio"] == 1
