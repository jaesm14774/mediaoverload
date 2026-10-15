"""Renderer prompt contracts; the external GPU is outside these scenarios."""

from pathlib import Path

from PIL import Image
import pytest

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.prompting import build_goal_brief
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.comfy_image import ComfyImageSkills


PROFILE = {
    "label": "hand-inked cartoon",
    "prompt": "variable-weight ink contours and clean cel-shaded colors",
    "composition": "a full illustrated garden with layered scenery",
    "palette": "coral and cyan with clear subject separation",
    "motion": "stable contours through anticipation and follow-through",
    "avoid": "isolated stickers or cutouts, empty backgrounds",
}


@pytest.mark.parametrize("media_type", ["sticker_pack", "animated_sticker", "game_sprite"])
@pytest.mark.parametrize("renderer", ["image", "keyframe"])
def test_user_given_scene_style_when_rendering_an_isolated_asset_then_background_requirements_override_scene_guidance(
    tmp_path, media_type, renderer,
):
    """User Given a style with full-scene rules When a sticker or sprite reaches the image renderer Then subject appearance is preserved without scene composition or isolation exclusions overriding the required background."""
    goal = GoalRequest(
        prompt="Kirby waves on a flat uniform chroma-key background #18e06f",
        media_type=media_type, style=PROFILE["label"],
        constraints={"visual_style_profile": dict(PROFILE), "negative_prompt": "watermark"},
    )
    brief = build_goal_brief(goal, goal.style, [])
    tools = ToolRegistry()
    submitted = []

    def render_image(payload):
        submitted.append(payload)
        path = Path(payload["run_dir"]) / "candidate.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (payload["width"], payload["height"]), "green").save(path)
        return {"saved_files": [str(path)]}

    tools.register("comfy.render_image", render_image, "Image provider boundary")
    tools.register("comfy.workflow.text_to_image", render_image, "Keyframe provider boundary")
    node = ExecutionNode(
        node_id="render", skill_name="image.render", depends_on=["brief"],
        inputs={"workflow_name": "krea2_turbo", "width": 64, "height": 40, "use_prior_frame": False},
    )
    context = SkillContext(
        ExecutionPlan(goal=goal, workflow_name="isolated-asset", nodes=[node]), node,
        RunState(goal={}, metadata={}, node_outputs={"brief": brief}),
    )
    result = (
        ComfyImageSkills(tools, tmp_path).render_image(context)
        if renderer == "image"
        else AgentMediaSkills(tools, tmp_path).generate_keyframe(context)
    )

    assert result.status == "success"
    assert submitted and all(Path(path).is_file() for path in result.outputs["saved_files"])
    for payload in submitted:
        prompt = payload["prompt"]
        assert PROFILE["prompt"] in prompt and PROFILE["palette"] in prompt
        assert "subject only" in prompt.casefold()
        assert "background requirements take priority" in prompt.casefold()
        assert "#18e06f" in prompt
        assert PROFILE["composition"] not in prompt
        assert PROFILE["avoid"] not in prompt
        assert PROFILE["avoid"] not in payload["negative_prompt"]
        assert "watermark" in payload["negative_prompt"]


def test_user_given_scene_style_when_rendering_a_scene_then_composition_and_exclusions_are_preserved():
    """User Given the same scene style When a full illustration is requested Then its composition and exclusions still reach the prompt and negative prompt."""
    goal = GoalRequest(
        prompt="Kirby waves in a garden", media_type="image", style=PROFILE["label"],
        constraints={"visual_style_profile": dict(PROFILE)},
    )
    brief = build_goal_brief(goal, goal.style, [])
    assert PROFILE["composition"] in brief["prompt"]
    assert PROFILE["avoid"] in brief["prompt"]
    assert PROFILE["avoid"] in brief["negative_prompt"]
