from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.llm_engine import LLMPromptEngine, PromptGenerationError
from agentic.runtime.registry import ToolRegistry
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import format_native_h3_prompt
from agentic.tools.wan2gp import Wan2GPToolset


class RecordedTextModel:
    """LLM transport fake using the JSON response contract retained in run records."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        response = json.dumps(next(self.responses))
        if callable(kwargs.get("_response_validator")):
            kwargs["_response_validator"](response)
        return response


class CapturingVideoRenderer:
    """GPU boundary fake matching the saved_files contract of completed Comfy runs."""

    def __init__(self) -> None:
        self.payload: dict | None = None

    def __call__(self, payload: dict) -> dict:
        self.payload = payload
        return {"saved_files": [str(Path(payload["run_dir"]) / "video.mp4")]}


def segment_context(prompt: object) -> SkillContext:
    goal = GoalRequest(
        prompt="A paper boat reaches the harbor.", media_type="long_video", style="watercolor",
        constraints={"longvideo_production_profile": "text2longvideo"},
    )
    segment = {
        "segment_id": "segment-1", "visual": "Earlier planning text.",
        "action": "Planning action.", "camera": "Planning camera advice.",
    }
    node = ExecutionNode(
        node_id="segment-video-01", skill_name="longvideo.render_segment_video",
        inputs={"segment_index": 0, "recipe": "t2v", "workflow_name": "wan2gp_h3_t2va",
                "render_tool": "wan2gp.render_h3", "length": 120, "steps": 16},
        depends_on=["segment-prompt-01"],
    )
    state = RunState(goal={}, metadata={}, node_outputs={
        "script-plan": {"segments": [segment]}, "idea-brief": {"negative_prompt": ""},
        "segment-prompt-01": {"prompt": prompt},
    })
    return SkillContext(plan=ExecutionPlan(goal=goal, workflow_name="longvideo", nodes=[node]), node=node, state=state)


@pytest.mark.parametrize("recipe", ["t2v", "anchor_first", "anchor_last", "anchor_first_last", "reference_bundle"])
def test_user_given_llm_scene_when_rendered_then_h3_receives_only_that_description(tmp_path: Path, recipe: str) -> None:
    """User Given a final LLM scene and separate planning notes, When the segment renders, Then H3 receives only the final scene description."""
    description = "A watercolor paper boat tilts into a gust, slides past a buoy, and settles in the harbor. The camera tracks alongside it."
    context = segment_context(description)
    context.node.inputs["recipe"] = recipe
    context.node.inputs["anchor_nodes"] = {"first": "first-frame", "last": "last-frame"}
    if recipe in {"anchor_first", "anchor_first_last"}:
        context.state.node_outputs["first-frame"] = {"frame_path": str(tmp_path / "first.png")}
    if recipe in {"anchor_last", "anchor_first_last"}:
        context.state.node_outputs["last-frame"] = {"frame_path": str(tmp_path / "last.png")}
    if recipe == "reference_bundle":
        context.node.inputs["reference_node"] = "reference-images"
        context.state.node_outputs["reference-images"] = {"reference_manifest": [{"image_path": str(tmp_path / "ref.png")}]}
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": description, "narration": "The boat reaches shelter."},
    ])))
    prepared = engine.prepare_segment(context.plan.goal, context.state["script-plan"]["segments"][0], "")
    context.state.node_outputs["segment-prompt-01"] = prepared
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    tools.register("wan2gp.render_h3", renderer, "Comfy GPU boundary")
    tools.register("wan2gp.render_h3", renderer, "Comfy GPU boundary")
    tools.register("wan2gp.render_h3", renderer, "Comfy GPU boundary")
    context.node.inputs["render_tool"] = "wan2gp.render_h3" if recipe == "t2v" else "wan2gp.render_h3"

    result = LongVideoSkills(tools, tmp_path).render_segment_video(context)

    assert result.status == "success"
    assert renderer.payload["prompt"] == description
    assert prepared["narration"] == "The boat reaches shelter."


def test_user_given_nested_llm_prompt_when_repaired_then_only_scene_text_is_accepted() -> None:
    """User Given an LLM prompt object like the retained failed run, When the response is repaired, Then the accepted prompt is description text rather than a serialized object."""
    description = "Kirby catches a drifting lantern and lifts it above the canal."
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": {"video_prompt": {"description": description}, "image_prompt_alternative": "Unused still."},
         "narration": {"voiceover_style": "Unused explanation."}},
        {"prompt": description, "narration": "The lantern is safe."},
    ])))

    result = engine.prepare_segment(GoalRequest(prompt="Save the lantern", media_type="long_video"),
                                    {"segment_id": "segment-1", "visual": "Lantern rescue"}, "")

    assert result["prompt"] == description
    assert result["narration"] == "The lantern is safe."


def test_user_given_nested_narration_when_repaired_then_voiceover_remains_text() -> None:
    """User Given valid scene text but nested narration, When the LLM response is repaired, Then both accepted output fields are text."""
    description = "Kirby lifts the lantern above the canal."
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": description, "narration": {"voiceover": "The lantern is safe."}},
        {"prompt": description, "narration": "The lantern is safe."},
    ])))
    result = engine.prepare_segment(GoalRequest(prompt="Lantern rescue", media_type="long_video"), {"segment_id": "segment-1", "visual": description}, "")
    assert result["prompt"] == description
    assert result["narration"] == "The lantern is safe."


def test_user_given_unrepairable_prompt_when_prepared_then_generation_stops() -> None:
    """User Given repeated non-text LLM prompts, When repairs cannot produce a description, Then preparation fails rather than returning a serialized response or a template."""
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": {"description": "Kirby jumps."}, "narration": "Kirby jumps."},
    ] * 3)))
    with pytest.raises(PromptGenerationError, match="prompt"):
        engine.prepare_segment(GoalRequest(prompt="Kirby jumps", media_type="long_video"), {"segment_id": "segment-1", "visual": "Kirby jumps."}, "")


@pytest.mark.parametrize("prompt", [None, {"description": "Kirby jumps."}, ["Kirby jumps."], "", "   ",
                                   "```json\n{\"prompt\": \"Kirby jumps.\"}\n```",
                                   "{'video_prompt': {'description': 'Kirby jumps.'}}",
                                   "Explanation: This prompt creates a compelling scene.",
                                   "Video action and camera direction: Follow these writing instructions."])
def test_user_given_invalid_prompt_when_render_requested_then_h3_is_not_submitted(tmp_path: Path, prompt: object) -> None:
    """User Given a non-description prompt, When rendering is requested, Then it fails before submission to the GPU provider."""
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    tools.register("wan2gp.render_h3", renderer, "Comfy GPU boundary")

    with pytest.raises(ValueError, match="prompt"):
        LongVideoSkills(tools, tmp_path).render_segment_video(segment_context(prompt))

    assert renderer.payload is None


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va", "ref2va"])
def test_user_given_nested_prompt_at_tool_entry_when_called_then_h3_rejects_it(tmp_path: Path, mode: str) -> None:
    """User Given a nested prompt at any H3 tool entry, When the tool is called, Then invalid content is rejected before contacting WanGP."""
    repo = Path(__file__).resolve().parents[2]
    toolset = Wan2GPToolset(AssetRegistry(repo / "agentic", asset_root=repo), tmp_path)

    with pytest.raises(ValueError, match="prompt"):
        toolset.execute({"workflow_name": f"wan2gp_h3_{mode}", "run_dir": str(tmp_path / "run"), "prompt": {"description": "Kirby runs."}})


def test_user_given_native_story_when_formatted_then_h3_gets_scene_without_editorial_context() -> None:
    """User Given native shots plus source and editorial records, When the H3 description is formatted, Then scene, camera and audio remain while source explanations and generic guidance stay out."""
    story = {
        "character": "Kirby", "base_prompt": "Kirby beside a canal.",
        "world": {"setting": "A moonlit canal.", "visual_language": "Watercolor with amber light."},
        "native_shots": [{"time": "0-15s", "action": "Kirby catches a lantern.",
                          "camera": "The camera tracks beside Kirby.", "state_change": "The lantern stops drifting."}],
        "native_audio": "Water ripples and a bright lantern chime.",
        "news_trace": {"source_fact": "SOURCE_RECORD_ONLY", "character_mapping": {"Kirby": "EDITORIAL_MAPPING_ONLY"}},
        "gag_card": {"prop_rule": "EDITORIAL_CARD_ONLY"},
    }

    prompt = format_native_h3_prompt(story)

    for description in ("moonlit canal", "Watercolor", "Kirby catches a lantern", "camera tracks", "lantern chime"):
        assert description in prompt
    for metadata in ("SOURCE_RECORD_ONLY", "EDITORIAL_MAPPING_ONLY", "EDITORIAL_CARD_ONLY",
                     "Video action and camera direction:", "Input relation:", "Follow the supplied creative brief"):
        assert metadata not in prompt
    assert story["news_trace"]["source_fact"] == "SOURCE_RECORD_ONLY"


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va", "ref2va"])
@pytest.mark.parametrize("invalid", [False, True])
def test_user_given_native_description_when_rendered_then_every_h3_mode_validates_it(tmp_path: Path, mode: str, invalid: bool) -> None:
    """User Given a native scene and separate reference notes, When any H3 mode renders, Then valid scene text stays unchanged and nested output is rejected before GPU submission."""
    description = "Kirby catches a drifting lantern as the camera tracks beside the canal. Water ripples."
    goal = GoalRequest(prompt="Lantern rescue", media_type="native_h3_story", duration_seconds=15)
    node = ExecutionNode(node_id="native-h3-render", skill_name="longvideo.render_native_h3",
                         inputs={"workflow_name": "minimax_h3_test", "use_last_frame": mode == "fl2va"})
    references = [{"path": str(tmp_path / "reference.png"), "type": "image", "tag": "subject_1", "role": "subject", "retention": "appearance",
                   "summary": "EDITORIAL_REFERENCE_ONLY", "retention_analysis": "EDITORIAL_REVIEW_ONLY"}]
    state = RunState(goal={}, metadata={}, node_outputs={
        "native-story-prompt": {"prompt": {"description": description} if invalid else description,
                                "native_audio": "EDITORIAL_AUDIO_NOTE_ONLY"},
        "native-keyframe-source-check": {"first_frame_path": str(tmp_path / "first.png"), "last_frame_path": str(tmp_path / "last.png")},
        "native-l2va-frame-source-check": {"last_frame_path": str(tmp_path / "last.png")},
        "native-ref2va-reference-check": {"reference_manifest": references},
    })
    context = SkillContext(plan=ExecutionPlan(goal=goal, workflow_name="native", nodes=[node]), node=node, state=state)
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    for tool_name in ("wan2gp.render_h3",):
        tools.register(tool_name, renderer, "Comfy GPU boundary")
    skills = LongVideoSkills(tools, tmp_path)
    render = {"t2va": skills.render_native_h3_t2v, "i2va": skills.render_native_h3,
              "fl2va": skills.render_native_h3, "l2va": skills.render_native_h3_l2va,
              "ref2va": skills.render_native_h3_ref2va}[mode]

    if invalid:
        with pytest.raises(ValueError, match="prompt"):
            render(context)
        assert renderer.payload is None
    else:
        assert render(context).status == "success"
        assert renderer.payload["prompt"] == description
        if mode == "ref2va":
            assert renderer.payload["reference_manifest"] == references
