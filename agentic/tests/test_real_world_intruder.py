"""Real-world intruder: a drawn character wanders into a real photograph, and a kid's taped sketch of it sits in the corner.

Acceptance scenarios are written first. The LLM provider and the ComfyUI tool are
injected at their public boundaries; the sketch-inset compositing runs on real
FFmpeg. Real GPU/LLM quality is covered by the live run recorded in docs.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from agentic.app.character_publishing import _resolve_publish_prompt
from agentic.app.generation_routing import news_selection_options
from agentic.app.main import build_runtime
from agentic.runtime.contracts import MIN_VIDEO_DURATION_SECONDS, RunState
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.prompt_tasks.real_world_intruder import (
    CLIP_SECONDS,
    generate_real_world_intruder_story,
    keyframe_prompt,
    sketch_prompt,
)
from agentic.runtime.strategies import GENERATION_STRATEGIES, NEWS_GROUNDED_GENERATION_TYPES
from agentic.skills.real_world_intruder import RealWorldIntruderSkills
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.sketch_inset_adapter import SketchInsetAdapter
from provider_fakes import ScriptedLLMManager

REPO_ROOT = Path(__file__).resolve().parents[2]

NEWS = {
    "title": "嘉義市夜市新推出草莓冰沙 排隊人潮繞整條街",
    "keyword": "夜市 草莓冰沙",
    "content": "嘉義市文化路夜市今天推出草莓冰沙，攤位前擺滿新鮮草莓，排隊人潮繞整條街。",
}
PINK = "Kirby is a round pink puffball with two short stubby arms, two round red feet, big dark-blue oval eyes and rosy cheeks"
MOUTHLESS = "Waddle Dee is a tan, round creature with no mouth, small stubby arms, honey-colored feet"


def story_payload() -> dict:
    return {
        "premise": "A pink puffball wanders onto a night-market counter and tries to move a real strawberry.",
        "source_anchors": [{"source_excerpt": "新鮮草莓", "visual_mapping": "a glossy heap of fresh strawberries"}],
        "real_scene": "a night-market stall counter lit by warm bulbs, a glossy heap of fresh strawberries on a steel tray",
        "setup_beat": "The character shoves one strawberry across the steel tray with both stubby arms, straining.",
        "escalation_beat": "The strawberry wobbles and squishes with a juicy splat; the character bounces back and flings its arms up.",
        "payoff_beat": "A string of bulbs above the stall pops with a bang and sparks; the character leaps in fright, tumbles off the strawberry and the camera snaps in close on its wide eyes as it keeps flailing.",
        "soundscape": "distant night-market chatter, soft wet squish, tiny bouncy boing",
        "music": "N/A",
    }


def write_story(payload: dict, profile: str = PINK) -> dict:
    client = PromptClient(mode="llm", manager=ScriptedLLMManager([payload], repeat_last=True))
    return generate_real_world_intruder_story(
        client, character="Kirby", character_profile=profile, news_context=NEWS
    )


# ---- the story ----


def test_user_given_news_and_a_character_when_the_story_is_written_then_it_has_a_real_scene_one_gag_and_news_anchors():
    """User Given a news item and a drawn character, When the story is written, Then it names one real photographable scene, one gag for the clip, a soundscape, and anchors quoted from the article."""
    story = write_story(story_payload())

    assert all(story[key] for key in ("real_scene", "setup_beat", "escalation_beat", "payoff_beat", "soundscape"))
    assert {a["source_excerpt"] for a in story["source_anchors"]} <= {"新鮮草莓"}
    assert "strawberries" in story["real_scene"]


@pytest.mark.parametrize(
    "missing", ["real_scene", "setup_beat", "escalation_beat", "payoff_beat", "soundscape", "premise"]
)
def test_user_given_a_story_missing_a_part_when_it_is_written_then_it_is_rejected(missing):
    """User Given an LLM answer without the scene, the gag, the sound or the premise, When the story is written, Then it is rejected instead of padded from a template."""
    payload = story_payload()
    payload[missing] = ""

    with pytest.raises(ValueError, match=missing):
        write_story(payload)


def test_user_given_an_anchor_not_in_the_article_when_the_story_is_written_then_it_is_rejected():
    """User Given an anchor quote the article never says, When the story is written, Then the story is rejected so the scene stays grounded in the news."""
    payload = story_payload()
    payload["source_anchors"][0]["source_excerpt"] = "太空旅行"

    with pytest.raises(ValueError, match="source_excerpt"):
        write_story(payload)


@pytest.mark.parametrize("field", ["real_scene", "setup_beat", "escalation_beat", "payoff_beat", "soundscape"])
def test_user_given_a_mouthless_character_when_the_story_needs_a_mouth_then_it_is_rejected(field):
    """User Given a character whose profile says it has no mouth, When any scene, gag or sound line makes it laugh or eat, Then the story is rejected so no video is asked to draw a mouth."""
    payload = story_payload()
    payload[field] += " and it laughs out loud"

    with pytest.raises(ValueError, match="mouth"):
        write_story(payload, profile=MOUTHLESS)


@pytest.mark.parametrize("beat", ["escalation_beat", "payoff_beat"])
@pytest.mark.parametrize("stillness", ["then holds still and admires the result", "and freezes in place", "then rests quietly", "and stays motionless"])
def test_user_given_a_beat_that_ends_in_stillness_when_the_story_is_written_then_it_is_rejected(beat, stillness):
    """User Given an escalation or payoff that has the character hold still, freeze or rest, When the story is written, Then it is rejected, because a live run that ended with 'holds still' left the last four seconds of the clip static."""
    payload = story_payload()
    payload[beat] += f" The character {stillness}."

    with pytest.raises(ValueError, match=beat):
        write_story(payload)


def test_user_given_a_story_request_when_it_is_sent_then_it_asks_for_a_twist_and_a_big_reaction_in_the_last_beat():
    """User Given the writer request, When it is sent, Then the last beat is asked to be an unexpected real-world event plus a big whole-body reaction that keeps moving to the last frame, and the writer is told which stillness words are rejected."""
    manager = ScriptedLLMManager([story_payload()], repeat_last=True)
    generate_real_world_intruder_story(
        PromptClient(mode="llm", manager=manager), character="Kirby", character_profile=PINK, news_context=NEWS
    )
    request = next(m["content"] for m in manager.text_model.requests[0]["messages"] if m["role"] == "user")

    assert "payoff_beat" in request and "unexpected" in request and "reaction" in request
    assert "grammatical subject" in request  # the character, not the event, carries the payoff sentence
    assert "holds still" in request


# ---- the pictures ----


def test_user_given_a_story_when_the_keyframe_prompt_is_built_then_the_photo_is_real_and_only_the_character_is_drawn():
    """User Given a written story, When the keyframe prompt is built, Then it asks for a real phone photograph of the scene, one small flat 2D character in its own style carrying its own anatomy, and nothing else drawn."""
    prompt = keyframe_prompt(write_story(story_payload()), character="Kirby", character_profile=PINK).lower()

    assert "real photograph" in prompt
    assert "night-market stall" in prompt
    assert "2d" in prompt and "flat" in prompt
    assert "round pink puffball" in prompt and "stubby arms" in prompt
    assert "only drawn" in prompt or "only drawn thing" in prompt


def test_user_given_a_character_when_the_sketch_prompt_is_built_then_it_is_a_childlike_marker_drawing_of_that_character():
    """User Given a character profile, When the sketch prompt is built, Then it asks for a crude childlike black-marker drawing on crumpled paper that carries the character's own features, so the sketch reads as the same character."""
    prompt = sketch_prompt(character="Kirby", character_profile=PINK).lower()

    assert "childlike" in prompt and "marker" in prompt and "crumpled" in prompt
    assert "round pink puffball" in prompt and "stubby arms" in prompt and "feet" in prompt


# ---- the plan ----


def build_plan(tmp_path, *, human_review=False, duration_seconds=10):
    planner, _, _ = build_runtime(REPO_ROOT, output_root=tmp_path)
    goal = planner.create_goal(
        prompt="",
        media_type="real_world_intruder",
        duration_seconds=duration_seconds,
        style="cute 2D cartoon",
        constraints={"character": "Kirby", "require_human_review": human_review},
    )
    return planner.build_plan(goal)


def by_id(plan):
    return {node.node_id: node for node in plan.nodes}


def index_of(plan, node_id):
    return [node.node_id for node in plan.nodes].index(node_id)


def test_user_when_the_strategy_is_registered_then_it_is_a_news_grounded_video_strategy_with_random_news():
    """User Given the strategy catalog, When real_world_intruder is looked up, Then it is a news-grounded video strategy whose news is a random item outside crime and geopolitics."""
    strategy = GENERATION_STRATEGIES["real_world_intruder"]
    options = news_selection_options("real_world_intruder")

    assert "real_world_intruder" in NEWS_GROUNDED_GENERATION_TYPES
    assert strategy.media_type == "real_world_intruder"
    assert options["random_inspiration"] is True
    assert {"社會", "國際"} <= set(options["exclude_categories"])


def test_user_given_a_run_when_the_plan_is_built_then_the_picture_and_sketch_exist_before_one_clip_that_meets_the_duration_floor(tmp_path):
    """User Given a run, When the plan is built, Then the real-scene keyframe and the kid's sketch are drawn before a single image-to-video clip of at least the pipeline's minimum duration, and the clip is composed with the sketch before QA."""
    plan = build_plan(tmp_path)
    nodes = by_id(plan)

    assert nodes["intruder-keyframe"].tool_name == "comfy.workflow.text_to_image"
    assert nodes["intruder-sketch"].tool_name == "comfy.workflow.text_to_image"
    clip = nodes["intruder-clip"]
    assert clip.tool_name == "comfy.render_h3"
    assert clip.inputs["length"] / 24 >= MIN_VIDEO_DURATION_SECONDS
    assert round(clip.inputs["length"] / 24) == CLIP_SECONDS  # the length the writer plans the gag for
    assert nodes["intruder-compose"].inputs["clip_node"] == "intruder-clip"
    assert nodes["intruder-compose"].inputs["sketch_node"] == "intruder-sketch"
    for earlier, later in (
        ("intruder-keyframe", "intruder-clip"),
        ("intruder-sketch", "intruder-compose"),
        ("intruder-clip", "intruder-compose"),
        ("intruder-compose", "native-h3-qa"),
    ):
        assert index_of(plan, earlier) < index_of(plan, later)


def test_user_given_a_long_default_duration_when_the_plan_is_built_then_the_clip_is_still_the_length_the_gag_was_written_for(tmp_path):
    """User Given a character whose default video duration is 30 seconds, When the plan is built, Then the single clip is still the short length the writer planned the gag for, not stretched to fill the duration."""
    clip = by_id(build_plan(tmp_path, duration_seconds=30))["intruder-clip"]

    assert round(clip.inputs["length"] / 24) == CLIP_SECONDS


def test_user_given_a_finished_plan_when_it_is_packaged_then_the_package_carries_the_composed_video(tmp_path):
    """User Given the plan, When the video is quality-checked, previewed and packaged, Then all three read the video that already carries the sketch, never the bare clip."""
    nodes = by_id(build_plan(tmp_path))

    def ancestors(node_id):
        found, pending = set(), [node_id]
        while pending:
            for parent in nodes[pending.pop()].depends_on:
                if parent not in found:
                    found.add(parent)
                    pending.append(parent)
        return found

    for node_id in ("native-h3-qa", "native-h3-package"):
        video_node = nodes[node_id].inputs["video_node"]
        assert video_node != "intruder-clip"
        assert video_node == "intruder-compose" or "intruder-compose" in ancestors(video_node)
    assert nodes["native-h3-qa"].inputs["render_node"] == "intruder-clip"
    assert nodes["native-h3-package"].inputs["render_node"] == "intruder-clip"
    assert nodes["native-h3-qa"].inputs["expected_fps"] % 24 == 0  # 24 fps, doubled only when the VFX pass interpolates


def test_user_given_human_review_when_the_plan_is_built_then_no_video_renders_before_the_keyframe_is_approved(tmp_path):
    """User Given human review, When the plan is built, Then the keyframe has a review step and the clip starts from the approved frame."""
    plan = build_plan(tmp_path, human_review=True)
    nodes = by_id(plan)

    assert nodes["intruder-clip"].inputs["anchor_node"] == "intruder-keyframe-review"
    assert index_of(plan, "intruder-keyframe-review") < index_of(plan, "intruder-clip")


def test_user_given_no_human_review_when_the_plan_is_built_then_the_clip_starts_from_the_keyframe(tmp_path):
    """User Given no human review, When the plan is built, Then the clip starts from the drawn keyframe directly."""
    nodes = by_id(build_plan(tmp_path))

    assert "intruder-keyframe-review" not in nodes
    assert nodes["intruder-clip"].inputs["anchor_node"] == "intruder-keyframe"


# ---- the skills ----


class RecordingTools:
    """A file-free fake at the public comfy tool boundary."""

    def __init__(self):
        self.calls = []

    def call(self, tool_name, payload):
        self.calls.append((tool_name, copy.deepcopy(payload)))
        return {"saved_files": ["rendered.mp4"]}


def skill_context(tmp_path, inputs, outputs, profile=PINK):
    goal = SimpleNamespace(
        prompt="",
        media_type="real_world_intruder",
        style="cute 2D cartoon",
        constraints={
            "character": "Kirby",
            "subject_context": {"subjects": [{"name": "Kirby", "profile": {"role_description": profile}}]},
        },
    )
    return SimpleNamespace(
        plan=SimpleNamespace(goal=goal),
        node=SimpleNamespace(inputs=inputs, depends_on=[]),
        state=RunState(goal={}, metadata={}, node_outputs=outputs),
    )


def stored_story():
    story = write_story(story_payload())
    return {**story, "keyframe_prompt": keyframe_prompt(story, character="Kirby", character_profile=PINK),
            "sketch_prompt": sketch_prompt(character="Kirby", character_profile=PINK)}


def test_user_given_a_story_when_the_keyframe_is_drawn_then_krea_gets_the_photo_prompt_without_a_cartoon_style_on_top(tmp_path):
    """User Given the written story, When the keyframe node runs, Then the image tool receives exactly the real-photo prompt and the run's cartoon style is not appended, so the room stays a photograph."""
    tools = RecordingTools()
    context = skill_context(
        tmp_path,
        {"workflow_name": "krea2_turbo", "story_node": "intruder-story", "prompt_key": "keyframe_prompt",
         "width": 896, "height": 1536, "suffix": "keyframe"},
        {"intruder-story": stored_story()},
    )

    RealWorldIntruderSkills(tools, tmp_path).render_image(context)

    tool, payload = tools.calls[0]
    assert tool == "comfy.workflow.text_to_image"
    assert "real photograph" in payload["prompt"].lower()
    assert "cute 2D cartoon" not in payload["prompt"]
    assert (payload["width"], payload["height"]) == (896, 1536)


def test_user_given_a_keyframe_when_the_clip_is_rendered_then_h3_starts_from_it_with_a_locked_camera_and_the_gag(tmp_path):
    """User Given an approved keyframe, When the clip is rendered, Then H3 animates it image-to-video with a locked camera, a small character that never grows, and the story's gag and sound."""
    tools = RecordingTools()
    context = skill_context(
        tmp_path,
        {"kind": "clip", "workflow_name": "minimax_h3_i2va", "story_node": "intruder-story",
         "anchor_node": "intruder-keyframe", "width": 448, "height": 768, "length": 242},
        {"intruder-story": stored_story(), "intruder-keyframe": {"selected_assets": ["k.png"]}},
    )

    result = RealWorldIntruderSkills(tools, tmp_path).render_clip(context)

    tool, payload = tools.calls[0]
    assert tool == "comfy.render_h3"
    assert payload["h3_mode"] == "i2va" and payload["image_path"] == "k.png"
    prompt = payload["prompt"].lower()
    assert "camera never moves" in prompt and "never zooms" in prompt
    assert "one fifth of the frame height" in prompt
    assert "shoves one strawberry" in prompt and "night-market chatter" in prompt
    assert result.status == "success"


def test_user_given_a_story_when_the_clip_is_rendered_then_h3_gets_three_timed_shots_that_end_on_the_payoff(tmp_path):
    """User Given a three-beat story, When the clip is rendered, Then the H3 prompt carries the setup, escalation and payoff as three timed shots in order, the payoff starts in the last third of the clip, and the clip is told to keep moving until its last frame."""
    tools = RecordingTools()
    context = skill_context(
        tmp_path,
        {"kind": "clip", "workflow_name": "minimax_h3_i2va", "story_node": "intruder-story",
         "anchor_node": "intruder-keyframe", "width": 448, "height": 768, "length": 243},
        {"intruder-story": stored_story(), "intruder-keyframe": {"selected_assets": ["k.png"]}},
    )

    RealWorldIntruderSkills(tools, tmp_path).render_clip(context)

    prompt = tools.calls[0][1]["prompt"]
    first, second, third = (prompt.index(f"[Shot {n}]") for n in (1, 2, 3))
    assert first < prompt.index("shoves one strawberry") < second < prompt.index("juicy splat") < third < prompt.index("bulbs above the stall")
    assert "00:03" in prompt[second:third] and "00:07" in prompt[third:]
    assert "until the last frame" in prompt.lower()
    assert "still mid-reaction on the last frame" in prompt[third:]


def test_user_given_a_mouthless_character_when_the_clip_is_rendered_then_h3_is_told_the_anatomy(tmp_path):
    """User Given a character whose profile says it has no mouth, When the clip is rendered, Then the H3 prompt carries that anatomy so the video does not grow a mouth."""
    tools = RecordingTools()
    context = skill_context(
        tmp_path,
        {"kind": "clip", "workflow_name": "minimax_h3_i2va", "story_node": "intruder-story",
         "anchor_node": "intruder-keyframe", "width": 448, "height": 768, "length": 242},
        {"intruder-story": stored_story(), "intruder-keyframe": {"selected_assets": ["k.png"]}},
        profile=MOUTHLESS,
    )

    RealWorldIntruderSkills(tools, tmp_path).render_clip(context)

    assert "no mouth" in tools.calls[0][1]["prompt"]


def test_user_given_a_finished_run_when_publish_copy_is_prepared_then_it_tells_the_intrusion():
    """User Given a finished run, When publish copy is prepared, Then the copy source names the news item, the scene and the gag instead of an empty goal prompt."""
    story = write_story(story_payload())
    result = {"state": {"node_outputs": {"intruder-story": {**story, "news_context": {"title": "Strawberry smoothie news"}}}}}

    text, source = _resolve_publish_prompt(result, fallback_prompt="")

    assert source == "real_world_intruder"
    assert "Strawberry smoothie news" in text and "night-market stall" in text
    assert "shoves one strawberry" in text and "bulbs above the stall" in text


# ---- the corner sketch, on real FFmpeg ----

FPS = 24
SECONDS = 6
SIZE = (320, 560)


def make_source(path: Path, size: tuple[int, int] = SIZE) -> None:
    """A moving test picture with a stereo tone, so a copied audio track and a changed picture are both observable."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size[0]}x{size[1]}:rate={FPS}:duration={SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}",
            "-filter_complex", "[1:a]aformat=channel_layouts=stereo[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
        ],
        check=True,
    )


def make_sketch(path: Path) -> None:
    image = Image.new("RGB", (512, 512), (250, 250, 245))
    ImageDraw.Draw(image).ellipse([120, 120, 392, 392], outline=(0, 0, 0), width=14)
    image.save(path)


def frame_at(video: Path, second: float, directory: Path) -> np.ndarray:
    target = directory / f"{video.stem}_{second}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(video), "-frames:v", "1", str(target)], check=True
    )
    return np.asarray(Image.open(target).convert("RGB"), dtype=np.int16)


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("inset")
    source, sketch, output = directory / "source.mp4", directory / "sketch.png", directory / "composed.mp4"
    make_source(source)
    make_sketch(sketch)
    result = SketchInsetAdapter(FFmpegAdapter()).compose(
        video_path=str(source), sketch_path=str(sketch), output_path=str(output)
    )
    return SimpleNamespace(directory=directory, source=source, output=output, result=result)


def test_user_given_a_clip_and_a_sketch_when_they_are_composed_then_the_picture_audio_and_length_are_unchanged(composed):
    """User Given a rendered clip and a taped sketch, When they are composed, Then the video keeps its canvas, frame rate, duration and stereo audio, so every later QA still passes."""
    source = FFmpegAdapter().probe_media(str(composed.source))
    output = FFmpegAdapter().probe_media(str(composed.output))

    assert (output["width"], output["height"]) == (source["width"], source["height"])
    assert abs(float(output["frame_rate"]) - float(source["frame_rate"])) < 0.15
    assert abs(float(output["duration"]) - float(source["duration"])) < 0.1
    assert output["has_audio"] and output["channels"] == source["channels"] == 2


def test_user_given_a_composed_clip_when_it_opens_then_the_sketch_sits_in_the_top_left_corner_and_nothing_else_changes(composed):
    """User Given a composed clip, When its first seconds are watched, Then the top-left corner shows the sketch and the rest of the picture is the original."""
    before = frame_at(composed.source, 1.0, composed.directory)
    after = frame_at(composed.output, 1.0, composed.directory)
    height, width = before.shape[:2]
    corner = (slice(0, int(height * 0.22)), slice(0, int(width * 0.4)))
    elsewhere = (slice(int(height * 0.4), height), slice(int(width * 0.4), width))

    assert np.abs(after[corner] - before[corner]).mean() > 10
    assert np.abs(after[elsewhere] - before[elsewhere]).mean() < 3


def test_user_given_a_composed_clip_when_it_ends_then_the_sketch_has_gone_and_the_scene_is_clean(composed):
    """User Given a composed clip, When its last second plays, Then the sketch has faded away and the top-left corner is the original picture again."""
    before = frame_at(composed.source, SECONDS - 0.7, composed.directory)
    after = frame_at(composed.output, SECONDS - 0.7, composed.directory)

    assert np.abs(after - before).mean() < 3


def test_user_given_a_landscape_clip_when_the_sketch_is_taped_then_it_stays_a_small_corner_note(tmp_path):
    """User Given a 16:9 clip, When the sketch is taped in, Then the sketch covers a small corner of the frame, not a large part of the scene's height."""
    source, sketch, output = tmp_path / "wide.mp4", tmp_path / "sketch.png", tmp_path / "wide_out.mp4"
    make_source(source, (640, 360))
    make_sketch(sketch)
    SketchInsetAdapter(FFmpegAdapter()).compose(video_path=str(source), sketch_path=str(sketch), output_path=str(output))

    before = frame_at(source, 1.0, tmp_path)
    after = frame_at(output, 1.0, tmp_path)
    changed_rows = np.where(np.abs(after - before).mean(axis=(1, 2)) > 8)[0]

    assert len(changed_rows) > 0
    assert changed_rows.max() < before.shape[0] * 0.35
