from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.llm_client import PromptGenerationError
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import format_native_h3_prompt
from agentic.tools.comfy_h3 import ComfyH3Toolset


class RecordedTextModel:
    """LLM transport fake using the JSON response contract retained in run records."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
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
        inputs={"segment_index": 0, "recipe": "t2v", "workflow_name": "minimax_h3_t2va",
                "render_tool": "comfy.render_h3", "length": 120, "steps": 16,
                "h3_prompt_node": ""},
        depends_on=["segment-prompt-01"],
    )
    state = RunState(goal={}, metadata={}, node_outputs={
        "script-plan": {"segments": [segment]}, "idea-brief": {"negative_prompt": ""},
        "segment-prompt-01": {"prompt": prompt, "overall_soundscape": "Wind and water move softly.",
                              "non_diegetic_music": "N/A"},
    })
    return SkillContext(plan=ExecutionPlan(goal=goal, workflow_name="longvideo", nodes=[node]), node=node, state=state)


@pytest.mark.parametrize("recipe", ["t2v", "anchor_first", "anchor_last", "anchor_first_last", "reference_bundle"])
def test_user_given_llm_scene_when_rendered_then_h3_receives_mode_specific_scene_and_audio_fields(tmp_path: Path, recipe: str) -> None:
    """User Given a final LLM scene and conditioning recipe, When the segment renders, Then H3 receives the official mode fields and image alignment."""
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
        context.state.node_outputs["reference-images"] = {"reference_manifest": [{
            "path": str(tmp_path / "ref.png"), "type": "image", "prompt_label": "<Picture 1>",
            "role": "identity", "retention": "appearance",
        }]}
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": description, "narration": "The boat reaches shelter.",
         "overall_soundscape": "Wind and water move softly.", "non_diegetic_music": "N/A"},
    ])))
    prepared = engine.prepare_segment(context.plan.goal, context.state["script-plan"]["segments"][0], "")
    context.state.node_outputs["segment-prompt-01"] = prepared
    if recipe != "t2v":
        context.node.inputs["h3_prompt_node"] = "segment-h3-prompt-01"
        context.state.node_outputs["segment-h3-prompt-01"] = prepared
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    tools.register("comfy.render_h3", renderer, "Comfy GPU boundary")
    context.node.inputs["render_tool"] = "comfy.render_h3"

    result = LongVideoSkills(tools, tmp_path).render_segment_video(context)

    assert result.status == "success"
    assert "integrated_multimodal_description:" in renderer.payload["prompt"] or "subject_definitions:" in renderer.payload["prompt"]
    assert "overall_soundscape:" in renderer.payload["prompt"]
    assert "non_diegetic_music:" in renderer.payload["prompt"]
    if recipe == "anchor_first":
        assert "<Picture 1> (from [Shot 1]) is fully referenced" in renderer.payload["prompt"]
    elif recipe == "anchor_last":
        assert "<Picture 1> (from [Shot 1]) aligns with the 5.00-second mark" in renderer.payload["prompt"]
    elif recipe == "anchor_first_last":
        assert "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark" in renderer.payload["prompt"]
        assert "<Picture 2> (from [Shot 1]) aligns with the 5.00-second mark" in renderer.payload["prompt"]
    elif recipe == "reference_bundle":
        assert "subject_definitions:" in renderer.payload["prompt"]
    assert description in renderer.payload["prompt"]
    assert prepared["narration"] == "The boat reaches shelter."


def test_user_given_a_reviewed_keyframe_and_action_prompt_when_short_h3_renders_then_the_first_request_uses_the_official_i2va_contract(tmp_path: Path) -> None:
    """User Given a reviewed keyframe and action-only short-video prompt When the first H3 I2VA request is submitted Then the prompt uses official image alignment and separate audio fields without motion-style rules overriding the action."""
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    tools.register("comfy.render_h3", renderer, "Comfy GPU boundary")
    image_path = tmp_path / "selected-first-frame.png"
    image_path.write_bytes(b"test frame at the external image boundary")
    action = (
        "Kirby coils into a tight ball, rockets diagonally toward the fuel can, "
        "strikes its rim, and rebounds into a balanced landing as the camera tracks upward."
    )
    node = ExecutionNode(
        node_id="animate-video",
        skill_name="image.animate",
        inputs={"workflow_name": "minimax_h3_i2va", "image_path": str(image_path), "length": 240},
        depends_on=["idea-brief"],
    )
    context = SkillContext(
        plan=ExecutionPlan(
            goal=GoalRequest(
                prompt="Animate Kirby reaching the fuel can",
                media_type="text2img2video",
                duration_seconds=10,
                style="simple doodle",
                constraints={
                    "visual_style_profile": {
                        "prompt": "wobbly ballpoint lines",
                        "motion": "slow contemplative drift",
                    }
                },
            ),
            workflow_name="short_video",
            nodes=[node],
        ),
        node=node,
        state=RunState(
            goal={},
            metadata={},
            node_outputs={
                "idea-brief": {
                    "prompt": action,
                    "overall_soundscape": "Air rushes, the metal rim rings, and Kirby's feet tap the can.",
                    "non_diegetic_music": "Fast taiko strikes build beneath a rising brass phrase.",
                }
            },
        ),
    )

    AgentMediaSkills(tools, tmp_path / "runs").animate_image(context)

    assert renderer.payload is not None
    h3_prompt = renderer.payload["prompt"]
    assert h3_prompt.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    )
    assert "integrated_multimodal_description: [Shot 1]" in h3_prompt
    assert "Preserve its visible subject identity, pose, silhouette, colors, composition, and existing props." in h3_prompt
    assert action in h3_prompt
    assert "overall_soundscape: Air rushes" in h3_prompt
    assert "non_diegetic_music: Fast taiko" in h3_prompt
    assert "Selected visual style rules take priority" not in h3_prompt
    assert "slow contemplative drift" not in h3_prompt


def test_user_given_conditioning_images_and_scene_draft_when_h3_prompt_is_prepared_then_vision_model_receives_images(tmp_path: Path) -> None:
    """User Given approved first and last frames with a scene draft, When the H3 prompt is prepared, Then the vision model receives both frames in role order and returns separate audio fields."""
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    first_frame.write_bytes(b"first frame test fixture")
    last_frame.write_bytes(b"last frame test fixture")
    vision_model = RecordedTextModel([
        {
            "prompt": "The masked knight turns toward the warning monitor and lifts one hand to the visor.",
            "overall_soundscape": "One monitor beep over quiet clinic room tone.",
            "non_diegetic_music": "N/A",
        }
    ])
    engine = LLMPromptEngine(
        mode="llm",
        manager=SimpleNamespace(text_model=RecordedTextModel([]), vision_model=vision_model),
    )

    prepared = engine.prepare_h3_segment(
        GoalRequest(prompt="A knight sees a warning", media_type="long_video", style="comics"),
        {"segment_id": "segment-1", "visual": "The knight reacts to the monitor."},
        draft_prompt="A masked knight reacts to the monitor.",
        overall_soundscape="Clinic room tone.",
        non_diegetic_music="N/A",
        duration_seconds=5.0,
        image_references=[
            {"path": str(first_frame), "label": "<Picture 1>", "role": "first-frame input"},
            {"path": str(last_frame), "label": "<Picture 2>", "role": "last-frame input"},
        ],
    )

    assert prepared["prompt"] == "The masked knight turns toward the warning monitor and lifts one hand to the visor."
    assert prepared["overall_soundscape"] == "One monitor beep over quiet clinic room tone."
    assert vision_model.calls[0]["kwargs"]["images"] == [str(first_frame), str(last_frame)]
    assert "first-frame input" in str(vision_model.calls[0]["messages"][1]["content"])
    assert "interpret those poses as successive views of the same subject" in str(vision_model.calls[0]["messages"][1]["content"])
    assert "ordered chain of positive physical-action beats" in str(vision_model.calls[0]["messages"][1]["content"])
    assert "From the first moment" in str(vision_model.calls[0]["messages"][1]["content"])
    assert "reach the resolved endpoint only in the final tenth" in str(vision_model.calls[0]["messages"][1]["content"])


def test_user_given_nested_llm_prompt_when_repaired_then_only_scene_text_is_accepted() -> None:
    """User Given an LLM prompt object like the retained failed run, When the response is repaired, Then the accepted prompt is description text rather than a serialized object."""
    description = "Kirby catches a drifting lantern and lifts it above the canal."
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": {"video_prompt": {"description": description}, "image_prompt_alternative": "Unused still."},
         "narration": {"voiceover_style": "Unused explanation."}},
        {"prompt": description, "narration": "The lantern is safe.",
         "overall_soundscape": "Canal water moves softly.", "non_diegetic_music": "N/A"},
    ])))

    result = engine.prepare_segment(GoalRequest(prompt="Save the lantern", media_type="long_video"),
                                    {"segment_id": "segment-1", "visual": "Lantern rescue"}, "")

    assert result["prompt"] == description
    assert result["narration"] == "The lantern is safe."


def test_user_given_nested_narration_when_repaired_then_voiceover_remains_text() -> None:
    """User Given valid scene text but nested narration, When the LLM response is repaired, Then both accepted output fields are text."""
    description = "Kirby lifts the lantern above the canal."
    engine = LLMPromptEngine(mode="llm", manager=SimpleNamespace(text_model=RecordedTextModel([
        {"prompt": description, "narration": {"voiceover": "The lantern is safe."},
         "overall_soundscape": "Canal water moves softly.", "non_diegetic_music": "N/A"},
        {"prompt": description, "narration": "The lantern is safe.",
         "overall_soundscape": "Canal water moves softly.", "non_diegetic_music": "N/A"},
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


@pytest.mark.parametrize("prompt", [None, {"description": "Kirby jumps."}, ["Kirby jumps."], "", "   "])
def test_user_given_invalid_prompt_when_render_requested_then_h3_is_not_submitted(tmp_path: Path, prompt: object) -> None:
    """User Given an empty or non-text prompt, When rendering is requested, Then it fails before submission to the GPU provider."""
    renderer = CapturingVideoRenderer()
    tools = ToolRegistry()
    tools.register("comfy.render_h3", renderer, "Comfy GPU boundary")

    with pytest.raises(ValueError, match="prompt"):
        LongVideoSkills(tools, tmp_path).render_segment_video(segment_context(prompt))

    assert renderer.payload is None


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va", "ref2va"])
def test_user_given_nested_prompt_at_tool_entry_when_called_then_h3_rejects_it(tmp_path: Path, mode: str) -> None:
    """User Given a nested prompt at any H3 tool entry, When the tool is called, Then invalid content is rejected before contacting ComfyUI."""
    repo = Path(__file__).resolve().parents[2]
    toolset = ComfyH3Toolset(AssetRegistry(repo / "agentic", asset_root=repo), tmp_path)

    with pytest.raises(ValueError, match="prompt"):
        toolset.execute({"workflow_name": f"minimax_h3_{mode}", "run_dir": str(tmp_path / "run"), "prompt": {"description": "Kirby runs."}})


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
    """User Given a native scene and separate reference notes, When any H3 mode renders, Then the official mode prompt is delivered and nested output is rejected before GPU submission."""
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
    for tool_name in ("comfy.render_h3",):
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
        assert "overall_soundscape:" in renderer.payload["prompt"]
        assert "non_diegetic_music:" in renderer.payload["prompt"]
        if mode == "i2va":
            assert "<Picture 1> (from [Shot 1]) is fully referenced" in renderer.payload["prompt"]
        elif mode == "fl2va":
            assert "<Picture 2> (from [Shot 1]) aligns" in renderer.payload["prompt"]
        elif mode == "l2va":
            assert "<Picture 1> (from [Shot 1]) aligns" in renderer.payload["prompt"]
        elif mode == "ref2va":
            assert "subject_definitions:" in renderer.payload["prompt"]
        assert description in renderer.payload["prompt"]
        if mode == "ref2va":
            assert renderer.payload["reference_manifest"] == references
