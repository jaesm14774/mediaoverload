"""News + character fusion story: one news item told as N keyframe beats joined by bridge clips.

Acceptance scenarios are written first. The LLM provider and the ComfyUI tool
are injected at their public boundaries; real GPU/LLM quality is covered by the
live run recorded in docs, not by these tests.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic.app.generation_routing import news_selection_options
from agentic.app.main import build_runtime
from agentic.runtime.contracts import RunState
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.prompt_tasks.news_fusion import generate_news_fusion_story
from agentic.runtime.strategies import (
    GENERATION_STRATEGIES,
    NEWS_GROUNDED_GENERATION_TYPES,
)
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.news_fusion import NewsFusionSkills
from provider_fakes import ScriptedLLMManager

REPO_ROOT = Path(__file__).resolve().parents[2]

NEWS = {
    "title": "嘉市斥資2335萬「縫合雙園」 百年公園今煥新竣工",
    "keyword": "嘉義市 公園",
    "content": "嘉義市政府今天宣布百年公園改建完成，新增空中步道連接兩座公園，市民可沿步道散步。",
}


def story_payload(beat_count: int = 3) -> dict:
    """A well-formed LLM answer: it invents a rainbow-rope prop the article never names."""
    return {
        "premise": "A chibi hero stitches two parks together with a glowing rope.",
        "source_anchors": [
            {"source_excerpt": "百年公園", "visual_mapping": "a century-old park gate"},
            {"source_excerpt": "空中步道", "visual_mapping": "a glowing rainbow rope bridge"},
        ],
        "beats": [
            {
                "keyframe_prompt": (
                    f"Beat {i + 1}: the hero beside a century-old park gate"
                    + (", holding a glowing rainbow rope bridge" if i else "")
                ),
                "clip_scene": f"The hero hops and tugs the rope, beat {i + 1}.",
                "soundscape": "park birds and a happy chirp",
                "music": "bouncy ukulele",
            }
            for i in range(beat_count)
        ],
        "bridges": [
            {
                "frame_prompt": f"The hero mid-air on the rope between the two spots, bridge {i + 1}",
                "first_scene": f"The hero leaps off and swings out on the rope, bridge {i + 1} first half.",
                "second_scene": f"The rope carries the hero down to the next spot, bridge {i + 1} second half.",
                "soundscape": "whoosh of rope",
                "music": "rising strings",
            }
            for i in range(beat_count - 1)
        ],
    }


MOUTHLESS = "Waddle Dee is a tan, round-bodied creature with chestnut-colored eyes, no mouth, small stubby arms, honey-colored feet"


def write_story(payload: dict, beat_count: int = 3, profile: str = "a round pink hero who loves snacks") -> dict:
    client = PromptClient(mode="llm", manager=ScriptedLLMManager([payload], repeat_last=True))
    return generate_news_fusion_story(
        client,
        character="Kirby",
        character_profile=profile,
        news_context=NEWS,
        beat_count=beat_count,
        style="cute 2D cartoon",
    )


def test_user_given_news_and_a_character_when_the_story_is_written_then_it_has_n_beats_and_n_minus_1_bridges():
    """User Given a news item and a character, When the story is written for 3 beats, Then it carries 3 beats, 2 bridges and anchors quoted from the article, and an invented fusion prop is allowed."""
    story = write_story(story_payload(3))

    assert len(story["beats"]) == 3
    assert len(story["bridges"]) == 2
    assert {a["source_excerpt"] for a in story["source_anchors"]} <= {"百年公園", "空中步道"}
    assert any("rainbow rope" in beat["keyframe_prompt"] for beat in story["beats"])


@pytest.mark.parametrize("beat_count", [2, 4])
def test_user_given_another_beat_count_when_the_story_is_written_then_the_beats_follow_it(beat_count):
    """User Given a requested beat count, When the story is written, Then the number of beats and bridges follows that count."""
    story = write_story(story_payload(beat_count), beat_count)

    assert len(story["beats"]) == beat_count
    assert len(story["bridges"]) == beat_count - 1


def test_user_given_a_story_with_the_wrong_beat_count_when_it_is_written_then_the_run_is_stopped():
    """User Given an LLM answer with 2 beats for a 3-beat request, When the story is written, Then it is rejected instead of padded from a template."""
    with pytest.raises(ValueError, match="beats"):
        write_story(story_payload(2), beat_count=3)


@pytest.mark.parametrize("missing", ["frame_prompt", "first_scene", "second_scene"])
def test_user_given_a_bridge_without_its_connecting_picture_or_halves_when_the_story_is_written_then_it_is_rejected(missing):
    """User Given a story whose bridge lacks the connecting-frame prompt or one of its two motions, When the story is written, Then it is rejected because the seam could not be rendered."""
    payload = story_payload(3)
    payload["bridges"][0][missing] = ""

    with pytest.raises(ValueError, match="bridges"):
        write_story(payload)


@pytest.mark.parametrize(
    "field_path",
    [("beats", 0, "clip_scene"), ("beats", 1, "keyframe_prompt"), ("beats", 2, "soundscape"),
     ("bridges", 0, "frame_prompt"), ("bridges", 1, "second_scene")],
)
def test_user_given_a_mouthless_character_when_the_story_gives_it_a_mouth_then_it_is_rejected(field_path):
    """User Given a character whose profile says it has no mouth, When any picture, motion or sound line of the story mentions a mouth, Then the story is rejected so no video is asked to draw one."""
    payload = story_payload(3)
    section, index, key = field_path
    payload[section][index][key] += " and opens its wide mouth to cheer"

    with pytest.raises(ValueError, match="mouth"):
        write_story(payload, profile=MOUTHLESS)


@pytest.mark.parametrize(
    ("addition", "profile"),
    [
        (" and laughs out loud with joy", MOUTHLESS),
        (" while singing a happy tune", MOUTHLESS),
        (" and gasps in surprise", MOUTHLESS),
        (" and runs across the stage", "Magolor is a short brown alien with no feet"),
        (" and stomps to the beat", "Magolor is a short brown alien with no feet"),
    ],
)
def test_user_given_a_character_without_a_body_part_when_the_story_needs_it_without_naming_it_then_it_is_rejected(
    addition, profile
):
    """User Given a character whose profile lacks a mouth or feet, When a motion line needs that part without naming it (laughing, singing, running), Then the story is rejected, because a live render drew a mouth from exactly such text."""
    payload = story_payload(3)
    for beat in payload["beats"]:
        beat["clip_scene"] = "The hero floats and tugs the rope"  # the sample story's hopping would itself need feet
    payload["beats"][0]["clip_scene"] += addition

    with pytest.raises(ValueError, match="profile says"):
        write_story(payload, profile=profile)


def test_user_given_a_mouthless_character_when_words_only_resemble_mouth_actions_then_the_story_is_accepted():
    """User Given a mouthless character, When the story says cheerful chimes, grinding gears or a single gadget, Then it is accepted: only real mouth actions are refused, and music direction is never judged."""
    payload = story_payload(3)
    payload["beats"][0]["soundscape"] = "cheerful chimes, grinding gears, a single cork pop"
    payload["beats"][0]["music"] = "cheerful singing strings"

    assert len(write_story(payload, profile=MOUTHLESS)["beats"]) == 3


def test_user_given_a_mouthless_character_when_the_story_uses_its_eyes_and_body_then_it_is_accepted():
    """User Given a character whose profile says it has no mouth, When the story shows joy through eyes, arms and props, Then the story is accepted."""
    story = write_story(story_payload(3), profile=MOUTHLESS)

    assert len(story["beats"]) == 3


def test_user_given_an_anchor_not_in_the_article_when_the_story_is_written_then_it_is_rejected():
    """User Given an anchor quote that the article never says, When the story is written, Then the story is rejected so the fusion stays grounded in the news."""
    payload = story_payload(3)
    payload["source_anchors"][0]["source_excerpt"] = "太空旅行"

    with pytest.raises(ValueError, match="source_excerpt"):
        write_story(payload)


def test_user_given_an_anchor_that_never_reaches_the_pictures_when_the_story_is_written_then_it_is_rejected():
    """User Given an anchor mapping that no keyframe or clip shows, When the story is written, Then the story is rejected."""
    payload = story_payload(3)
    payload["source_anchors"][1]["visual_mapping"] = "a submarine"

    with pytest.raises(ValueError, match="visual_mapping"):
        write_story(payload)


def test_user_when_the_strategy_is_registered_then_it_is_a_news_grounded_video_strategy():
    """User Given the strategy catalog, When news_fusion_story is looked up, Then it is a news-grounded video strategy."""
    strategy = GENERATION_STRATEGIES["news_fusion_story"]

    assert "news_fusion_story" in NEWS_GROUNDED_GENERATION_TYPES
    assert strategy.media_type == "news_fusion_story"


def build_plan(tmp_path, *, beats=3, human_review=False):
    planner, _, _ = build_runtime(REPO_ROOT, output_root=tmp_path)
    goal = planner.create_goal(
        prompt="",
        media_type="news_fusion_story",
        duration_seconds=15,
        style="cute 2D cartoon",
        constraints={
            "character": "Kirby",
            "news_fusion_beat_count": beats,
            "require_human_review": human_review,
        },
    )
    return planner.build_plan(goal)


def by_id(plan):
    return {node.node_id: node for node in plan.nodes}


@pytest.mark.parametrize("beats", [2, 3, 4])
def test_user_given_n_beats_when_the_plan_is_built_then_each_seam_has_two_bridge_halves_in_one_video(tmp_path, beats):
    """User Given N beats, When the plan is built, Then there are N keyframes, N clips and N-1 seams of two bridge halves, and the final video joins clip, first half, second half, next clip in story order."""
    plan = build_plan(tmp_path, beats=beats)
    nodes = by_id(plan)

    assert len([n for n in nodes if n.startswith("fusion-keyframe-")]) == beats
    assert len([n for n in nodes if n.startswith("fusion-clip-")]) == beats
    assert len([n for n in nodes if n.startswith("fusion-bridge-frame-")]) == beats - 1
    final = nodes["concat-final-video"]
    expected = []
    for index in range(1, beats + 1):
        expected.append(f"fusion-clip-{index:02d}")
        if index < beats:
            expected += [f"fusion-bridge-in-{index:02d}", f"fusion-bridge-out-{index:02d}"]
    assert final.inputs["video_node_ids"] == expected
    order = [n.node_id for n in plan.nodes]
    assert all(order.index(video) < order.index("concat-final-video") for video in expected)


def test_user_given_a_seam_when_the_plan_is_built_then_a_klein_frame_from_the_clip_end_sits_between_the_two_bridge_halves(tmp_path):
    """User Given three beats, When the plan is built, Then each seam gets a connecting picture edited by Klein from the previous clip's last frame, the first half travels clip end to that picture, and the second half travels that picture to the next keyframe, so no half has to jump across two scenes."""
    plan = build_plan(tmp_path, beats=3)
    nodes = by_id(plan)

    for index in (1, 2):
        suffix = f"{index:02d}"
        frame = nodes[f"fusion-bridge-frame-{suffix}"]
        assert frame.tool_name == "comfy.workflow.image_to_image"
        assert frame.inputs["identity_refine_workflow_name"] == plan.metadata["bridge_frame_workflow"]
        assert frame.inputs["prompt_key"] == f"bridge_{index}_frame_prompt"
        assert frame.depends_on[-1] == f"fusion-tail-{suffix}"
        assert nodes[f"fusion-bridge-in-{suffix}"].inputs["anchor_nodes"] == {
            "first": f"fusion-tail-{suffix}",
            "last": f"fusion-bridge-frame-{suffix}",
        }
        assert nodes[f"fusion-bridge-out-{suffix}"].inputs["anchor_nodes"] == {
            "first": f"fusion-bridge-frame-{suffix}",
            "last": f"fusion-keyframe-{index + 1:02d}",
        }
        assert nodes[f"fusion-tail-{suffix}"].inputs["source_video_node"] == f"fusion-clip-{suffix}"


def test_user_given_a_seam_frame_when_the_klein_node_runs_then_it_edits_the_clip_end_not_a_planning_asset(tmp_path):
    """User Given the planned connecting-frame node, When the keyframe skill runs it, Then Klein is asked to edit the previous clip's last frame with that seam's frame prompt, whatever other nodes the step depends on."""
    plan = build_plan(tmp_path, beats=3)
    node = by_id(plan)["fusion-bridge-frame-01"]
    story = story_payload(3)
    outputs = {
        dep: {"workflow_name": "x", "asset_status": []}
        for dep in node.depends_on
    }
    outputs["fusion-story"] = {"bridge_1_frame_prompt": story["bridges"][0]["frame_prompt"]}
    outputs["fusion-tail-01"] = {"frame_path": "tail1.png"}
    tools = RecordingTools()
    context = SimpleNamespace(
        plan=plan,
        node=node,
        state=RunState(goal={}, metadata={}, node_outputs=outputs),
    )

    AgentMediaSkills(tools, tmp_path).generate_keyframe(context)

    tool, payload = tools.calls[0]
    assert tool == "comfy.workflow.image_to_image"
    assert payload["image_path"] == "tail1.png"
    assert payload["workflow_name"] == plan.metadata["bridge_frame_workflow"]
    assert "mid-air on the rope" in payload["prompt"]


def test_user_given_beats_when_the_plan_is_built_then_later_keyframes_grow_from_the_previous_one(tmp_path):
    """User Given three beats, When the plan is built, Then beat 1 is drawn fresh and every later keyframe is edited from the previous keyframe so the character stays the same."""
    nodes = by_id(build_plan(tmp_path, beats=3))

    assert nodes["fusion-keyframe-01"].inputs.get("use_prior_frame") is False
    for index in (2, 3):
        keyframe = nodes[f"fusion-keyframe-{index:02d}"]
        assert keyframe.inputs["use_prior_frame"] is True
        assert keyframe.depends_on[0] == f"fusion-keyframe-{index - 1:02d}"
        assert keyframe.inputs["prompt_key"] == f"beat_{index}_keyframe_prompt"


def test_user_given_human_review_when_the_plan_is_built_then_no_video_renders_before_every_keyframe_is_approved(tmp_path):
    """User Given human review, When the plan is built, Then every keyframe has a review step and every clip anchors on the approved frames."""
    nodes = by_id(build_plan(tmp_path, beats=3, human_review=True))

    for index in (1, 2, 3):
        assert f"fusion-keyframe-review-{index:02d}" in nodes
    assert nodes["fusion-clip-01"].inputs["anchor_nodes"]["first"] == "fusion-keyframe-review-01"
    assert nodes["fusion-bridge-out-01"].inputs["anchor_nodes"]["last"] == "fusion-keyframe-review-02"
    # A connecting frame exists only after its clip renders, so it cannot be reviewed up front.
    assert not [n for n in nodes if n.startswith("fusion-bridge-frame-review")]
    plan_order = list(nodes)
    for index in (1, 2, 3):
        assert plan_order.index(f"fusion-keyframe-review-{index:02d}") < plan_order.index("fusion-clip-01")


class RecordingTools:
    """A file-free fake at the public comfy.render_h3 tool boundary."""

    def __init__(self):
        self.calls = []

    def call(self, tool_name, payload):
        self.calls.append((tool_name, copy.deepcopy(payload)))
        return {"saved_files": ["rendered.mp4"]}


def mouthless_constraints():
    return {
        "character": "Waddle Dee",
        "subject_context": {
            "subjects": [{"name": "Waddle Dee", "profile": {"role_description": MOUTHLESS}}],
        },
    }


def render(tmp_path, node_inputs, outputs, constraints=None):
    tools = RecordingTools()
    node = SimpleNamespace(inputs=node_inputs, depends_on=[])
    context = SimpleNamespace(
        plan=SimpleNamespace(
            goal=SimpleNamespace(prompt="", constraints=constraints or {"character": "Kirby"}, media_type="news_fusion_story")
        ),
        node=node,
        state=RunState(goal={}, metadata={}, node_outputs=outputs),
    )
    result = NewsFusionSkills(tools, tmp_path).render_clip(context)
    return tools.calls, result


def story_output():
    story = story_payload(3)
    return {
        "beats": story["beats"],
        "bridges": story["bridges"],
    }


def test_user_given_a_clip_when_it_is_rendered_then_h3_starts_from_its_keyframe(tmp_path):
    """User Given a beat clip, When it is rendered, Then H3 is asked to animate from that beat's keyframe with that beat's scene and sound."""
    calls, result = render(
        tmp_path,
        {"kind": "clip", "workflow_name": "minimax_h3_i2va", "index": 0, "story_node": "fusion-story",
         "anchor_nodes": {"first": "fusion-keyframe-01"}, "width": 384, "height": 736, "length": 124},
        {"fusion-story": story_output(), "fusion-keyframe-01": {"selected_assets": ["k1.png"]}},
    )

    tool, payload = calls[0]
    assert tool == "comfy.render_h3"
    assert payload["h3_mode"] == "i2va"
    assert payload["workflow_name"] == "minimax_h3_i2va"
    assert payload["image_path"] == "k1.png" and "last_image_path" not in payload
    assert "beat 1" in payload["prompt"] and "park birds" in payload["prompt"]
    assert result.status == "success"


@pytest.mark.parametrize(
    ("kind", "first", "last", "first_path", "last_path", "half"),
    [
        ("bridge_in", "fusion-tail-01", "fusion-bridge-frame-01", "tail1.png", "mid1.png", "first half"),
        ("bridge_out", "fusion-bridge-frame-01", "fusion-keyframe-02", "mid1.png", "k2.png", "second half"),
    ],
)
def test_user_given_a_bridge_half_when_it_is_rendered_then_h3_connects_its_two_frames(
    tmp_path, kind, first, last, first_path, last_path, half
):
    """User Given a bridge half, When it is rendered, Then H3 travels between that half's two frames using that half's own motion."""
    calls, _ = render(
        tmp_path,
        {"kind": kind, "workflow_name": "minimax_h3_fl2va", "index": 0, "story_node": "fusion-story",
         "anchor_nodes": {"first": first, "last": last},
         "width": 384, "height": 736, "length": 107},
        {"fusion-story": story_output(), "fusion-tail-01": {"frame_path": "tail1.png"},
         "fusion-bridge-frame-01": {"saved_files": ["mid1.png"]},
         "fusion-keyframe-02": {"selected_assets": ["k2.png"]}},
    )

    _, payload = calls[0]
    assert payload["h3_mode"] == "fl2va"
    assert payload["image_path"] == first_path
    assert payload["last_image_path"] == last_path
    assert f"bridge 1 {half}" in payload["prompt"]


@pytest.mark.parametrize(
    ("kind", "anchors"),
    [
        ("clip", {"first": "fusion-keyframe-01"}),
        ("bridge_in", {"first": "fusion-tail-01", "last": "fusion-bridge-frame-01"}),
        ("bridge_out", {"first": "fusion-bridge-frame-01", "last": "fusion-keyframe-02"}),
    ],
)
def test_user_given_a_mouthless_character_when_any_video_is_rendered_then_h3_is_told_the_character_has_no_mouth(
    tmp_path, kind, anchors
):
    """User Given a character whose profile says it has no mouth, When a beat clip or either bridge half is rendered, Then the H3 prompt carries that anatomy so the video does not grow a mouth."""
    calls, _ = render(
        tmp_path,
        {"kind": kind, "workflow_name": "minimax_h3_i2va", "index": 0, "story_node": "fusion-story",
         "anchor_nodes": anchors, "width": 384, "height": 736, "length": 124},
        {"fusion-story": story_output(), "fusion-keyframe-01": {"selected_assets": ["k1.png"]},
         "fusion-tail-01": {"frame_path": "tail1.png"}, "fusion-bridge-frame-01": {"saved_files": ["mid1.png"]},
         "fusion-keyframe-02": {"selected_assets": ["k2.png"]}},
        constraints=mouthless_constraints(),
    )

    prompt = calls[0][1]["prompt"]
    assert "no mouth" in prompt and "honey-colored feet" in prompt


def test_user_given_a_mouthless_character_when_the_seam_frame_is_edited_then_klein_is_told_the_anatomy(tmp_path):
    """User Given a mouthless character, When Klein paints the connecting frame, Then its prompt carries the same anatomy so the frame never gives the character a mouth."""
    plan = build_plan(tmp_path, beats=3)
    plan.goal.constraints.update(mouthless_constraints())
    node = by_id(plan)["fusion-bridge-frame-01"]
    outputs = {dep: {"workflow_name": "x", "asset_status": []} for dep in node.depends_on}
    outputs["fusion-story"] = {"bridge_1_frame_prompt": "The hero mid-air on the rope"}
    outputs["fusion-tail-01"] = {"frame_path": "tail1.png"}
    tools = RecordingTools()
    context = SimpleNamespace(plan=plan, node=node, state=RunState(goal={}, metadata={}, node_outputs=outputs))

    AgentMediaSkills(tools, tmp_path).generate_keyframe(context)

    assert "no mouth" in tools.calls[0][1]["prompt"]


def test_user_given_a_fusion_run_when_news_is_picked_then_it_is_a_random_item_outside_crime_and_geopolitics():
    """User Given a news-fusion run, When its news is picked, Then the pick is one random item that is never from crime, accident or geopolitical categories, while other strategies keep the default pick."""
    options = news_selection_options("news_fusion_story")

    assert options["random_inspiration"] is True
    assert {"社會", "國際", "全球", "選舉"} <= set(options["exclude_categories"])
    assert "政治" in options["exclude_categories"]  # the shared finance and politics filter still applies
    assert news_selection_options("native_h3_story") == {}


def test_user_given_a_finished_fusion_run_when_publish_copy_is_prepared_then_it_tells_the_fused_story():
    """User Given a finished fusion run, When publish copy is prepared, Then the copy source names the news item and the story beats instead of an empty goal prompt."""
    from agentic.app.character_publishing import _resolve_publish_prompt

    story = story_output()
    story["premise"] = "A kitchen fire drill becomes a hide and seek game."
    result = {"state": {"node_outputs": {"fusion-story": {**story, "news_context": {"title": "Fire drill news"}}}}}

    text, source = _resolve_publish_prompt(result, fallback_prompt="")

    assert source == "news_fusion_story"
    assert "Fire drill news" in text and "hide and seek" in text and "Beat 1:" in text and "Beat 3:" in text


def test_user_given_a_planned_fusion_run_when_a_planned_clip_is_rendered_then_its_planned_workflow_runs(tmp_path):
    """User Given a planned fusion run, When the planned first clip and first bridge render, Then each renders the workflow the plan selected and passes its own QA frame rate."""
    plan = build_plan(tmp_path)
    nodes = by_id(plan)
    outputs = {"fusion-story": story_output(), "fusion-keyframe-01": {"selected_assets": ["k1.png"]},
               "fusion-keyframe-02": {"selected_assets": ["k2.png"]}, "fusion-tail-01": {"frame_path": "tail1.png"},
               "fusion-bridge-frame-01": {"saved_files": ["mid1.png"]}}

    clip_calls, _ = render(tmp_path, nodes["fusion-clip-01"].inputs, outputs)
    in_calls, _ = render(tmp_path, nodes["fusion-bridge-in-01"].inputs, outputs)
    out_calls, _ = render(tmp_path, nodes["fusion-bridge-out-01"].inputs, outputs)

    assert clip_calls[0][1]["workflow_name"] == plan.metadata["selected_workflow"]
    assert in_calls[0][1]["workflow_name"] == plan.metadata["bridge_workflow"]
    assert out_calls[0][1]["workflow_name"] == plan.metadata["bridge_workflow"]
    assert nodes["longvideo-video-qa"].inputs["expected_fps"] == 24
