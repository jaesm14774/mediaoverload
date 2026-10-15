"""FL2VA ending behavior at the planner, prompt, and ComfyUI tool boundaries.

The image provider is injected here; the real ComfyUI contract is exercised by
the accompanying live ending-generation probe, rather than pretending that a
local test proves GPU image quality.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from agentic.app.main import build_runtime
from agentic.runtime.contracts import RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import StoryboardError, format_native_h3_keyframe_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]


def story():
    return {
        "character": "Kirby",
        "render_mode": "first_last_frame_to_video",
        "world": {
            "setting": "an illustrated garden table",
            "visual_language": "hand-inked 2D cartoon with coral and teal accents",
        },
        "native_shots": [
            {
                "action": "Kirby holds an invitation card with curious eyes.",
                "state_change": "The card is in Kirby's hands and his gaze is on it.",
            },
            {
                "action": "Kirby declines the invitation and sets the card aside.",
                "state_change": "Kirby's hands are empty, the card rests on the table edge, and his eyes look down with concern.",
                "camera": "side view showing Kirby's empty hands and the card on the table edge",
            },
        ],
    }


def plan(tmp_path):
    planner, _, _ = build_runtime(REPO_ROOT, output_root=tmp_path)
    goal = planner.create_goal(
        prompt="Kirby declines an invitation at a garden table",
        media_type="native_h3_fl2va_story",
        duration_seconds=15,
        style="hand-inked 2D cartoon",
        constraints={
            "character": "Kirby",
            "native_h3_use_last_frame": True,
            "storyboard_path": "configs/storyboards/native_h3_15s.yaml",
            "native_h3_keyframe_candidate_count": 6,
            "require_human_review": True,
            "visual_style_profile": {
                "prompt": "hand-inked 2D cartoon with coral and teal accents",
            },
        },
    )
    return planner.build_plan(goal)


class ImageProvider:
    """A file-producing fake at the registered ComfyUI public tool boundary."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.requests = []

    def generate(self, payload):
        self.requests.append(("text_to_image", payload))
        assert payload["width"] > 0 and payload["height"] > 0
        return self._save("ending", payload)

    def refine(self, payload):
        self.requests.append(("image_to_image", payload))
        assert Path(payload["image_path"]).is_file()
        return self._save("refined", payload)

    def _save(self, prefix, payload):
        files = []
        for index in range(payload["image_count"]):
            path = self.tmp_path / f"{prefix}_{index}.png"
            Image.new("RGB", (64, 36), "pink").save(path)
            files.append(str(path))
        return {"saved_files": files, "workflow_name": payload["workflow_name"]}


@pytest.mark.parametrize("explicit_opening_path", [False, True])
def test_user_gets_independently_drawn_ending_candidates(tmp_path, explicit_opening_path):
    """User Given an approved opening When FL2VA generates ending candidates Then it draws the final state from text with the same style and no opening image input."""
    execution_plan = plan(tmp_path)
    ending = next(n for n in execution_plan.nodes if n.node_id == "native-ending-keyframe")
    opening_path = tmp_path / "approved_opening.png"
    Image.new("RGB", (64, 36), "green").save(opening_path)
    if explicit_opening_path:
        # The opt-out must also discard an explicit input, not just dependencies.
        ending.inputs["use_prior_frame"] = False
        ending.inputs["prior_frame_path"] = str(opening_path)
    provider = ImageProvider(tmp_path)
    tools = ToolRegistry()
    tools.register("comfy.workflow.text_to_image", provider.generate, "Generate image candidates")
    tools.register("comfy.workflow.image_to_image", provider.refine, "Refine an input image")
    final_prompt = format_native_h3_keyframe_prompt(story(), frame="ending")
    state = RunState(goal={}, metadata={}, node_outputs={
        **{dependency: {} for dependency in ending.depends_on},
        "native-opening-review": {"selected_assets": [str(opening_path)]},
        "native-story-prompt": {"ending_keyframe_prompt": final_prompt},
    })

    result = AgentMediaSkills(tools, tmp_path).generate_keyframe(
        SkillContext(execution_plan, ending, state)
    )

    assert result.status == "success"
    assert len(result.outputs["saved_files"]) == 6
    for method, payload in provider.requests:
        assert method == "text_to_image"
        assert "image_path" not in payload
        assert "hand-inked 2D cartoon with coral and teal accents" in payload["prompt"]
        assert "Kirby's hands are empty" in payload["prompt"]
    assert all(Path(f).is_file() and f != str(opening_path) for f in result.outputs["saved_files"])


def test_user_sees_the_final_scene_with_its_own_camera():
    """User Given a solo story with a changed final pose and camera When the ending is described Then the illustration shows that final scene without locking it to the opening shot."""
    prompt = format_native_h3_keyframe_prompt(story(), frame="ending")

    assert "Kirby's hands are empty" in prompt
    assert "Camera: side view showing Kirby's empty hands" in prompt
    assert "hand-inked 2D cartoon" in prompt
    assert "an illustrated garden table" in prompt
    assert "both full figures" not in prompt
    assert "supplied opening image" not in prompt


def test_user_cannot_render_an_unchanged_fl2va_ending():
    """User Given identical opening and final states When FL2VA prepares the ending Then it refuses that story instead of generating another version of the opening."""
    unchanged = deepcopy(story())
    unchanged["native_shots"][-1]["state_change"] = unchanged["native_shots"][0]["state_change"]

    with pytest.raises(StoryboardError, match="distinct"):
        format_native_h3_keyframe_prompt(unchanged, frame="ending")


def test_user_sees_only_the_final_state_in_the_ending_camera():
    """User Given a camera plan that starts with an earlier pose When FL2VA describes the ending still Then only the final camera and final pose reach image generation."""
    resolved = story()
    resolved["native_shots"][-1]["camera"] = (
        "Opening framing: Kirby holds the invitation card. "
        "The camera settles on Kirby's empty hands and the card resting on the table edge."
    )
    prompt = format_native_h3_keyframe_prompt(resolved, frame="ending")

    assert "Kirby's empty hands" in prompt
    assert "Kirby holds the invitation card" not in prompt


def test_user_gets_a_final_prompt_without_opening_composition_locks(tmp_path):
    """User Given a resolved FL2VA story When the story skill prepares its ending prompt Then it preserves the final pose and art direction without ordering a redraw of the opening."""
    class ResolvedStoryService:
        def resolve(self, base_storyboard, **kwargs):
            resolved = deepcopy(base_storyboard)
            resolved.update(story())
            return resolved, {
                "prompt_mode": "llm",
                "source": "news",
                "creative_seed": "invitation",
                "news_context": kwargs["news_context"],
            }

    execution_plan = plan(tmp_path)
    node = next(n for n in execution_plan.nodes if n.node_id == "native-story-prompt")
    state = RunState(goal={}, metadata={})
    result = LongVideoSkills(
        ToolRegistry(), tmp_path, story_service=ResolvedStoryService()
    ).prepare_native_h3_story(SkillContext(execution_plan, node, state))
    prompt = result.outputs["ending_keyframe_prompt"]

    assert "Kirby's hands are empty" in prompt
    assert "hand-inked 2D cartoon" in prompt
    assert "supplied opening image" not in prompt
    assert "camera axis, framing, perspective" not in prompt


def test_user_can_still_refine_an_image_for_other_continuity_work(tmp_path):
    """User Given a continuity refinement outside FL2VA ending generation When the caller requests an input image Then the image provider receives that source image."""
    execution_plan = plan(tmp_path)
    ending = next(n for n in execution_plan.nodes if n.node_id == "native-ending-keyframe")
    ending.inputs["use_prior_frame"] = True
    opening_path = tmp_path / "continuity_source.png"
    Image.new("RGB", (64, 36), "green").save(opening_path)
    ending.inputs["prior_frame_path"] = str(opening_path)
    execution_plan.goal.media_type = "text2image2video"
    provider = ImageProvider(tmp_path)
    tools = ToolRegistry()
    tools.register("comfy.workflow.text_to_image", provider.generate, "Generate image candidates")
    tools.register("comfy.workflow.image_to_image", provider.refine, "Refine an input image")
    state = RunState(goal={}, metadata={}, node_outputs={
        **{dependency: {} for dependency in ending.depends_on},
        "native-story-prompt": {"ending_keyframe_prompt": "Kirby lifts a lantern"},
    })

    result = AgentMediaSkills(tools, tmp_path).generate_keyframe(
        SkillContext(execution_plan, ending, state)
    )

    assert result.status == "success"
    assert all(method == "image_to_image" and p["image_path"] == str(opening_path)
               for method, p in provider.requests)
