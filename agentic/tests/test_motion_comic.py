"""User-facing strategy acceptance; real Pillow/FFmpeg, injected provider boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import random

from PIL import Image
import pytest
import yaml

from agentic.app.character_payload import build_goal_payload_from_character_config
from agentic.app.character_publishing import (
    _extract_publish_visual_grounding,
    _resolve_publish_prompt,
    collect_media_paths_from_run_result,
)
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest, RunState, SkillContext
from agentic.runtime.creativity import RetryPolicy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.motion_comics import validate_motion_comic_story
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.story_cards import load_story_card_font, resolve_story_card_font
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.motion_comic import register_motion_comic_skills
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.media_services import register_media_service_tools
from agentic.tools.motion_comic_adapter import wrap_caption
from character_workflow_helpers import make_character_workflow_request
from provider_fakes import ScriptedLLMManager

REPO = Path(__file__).resolve().parents[2]


def story_response():
    return {
        "title": "最後一口",
        "visual_style": "Flat pastel gouache with dark plum outlines.",
        "continuity": "One round pink hero on the right, one white plate on the left, a wooden table and ivory wall; same camera and scale.",
        "panels": [
            {
                "role": "setup",
                "caption": "只吃一小口。",
                "image_prompt": "The hero prepares one small forkful beside an almost whole cake.",
                "visible_state": "Almost whole strawberry cake and a tiny forkful.",
                "cause": "The hero promises to take only one mouthful.",
            },
            {
                "role": "escalation",
                "caption": "再一小口就好。",
                "image_prompt": "The hero eats another forkful with half the cake missing.",
                "visible_state": "A large part of the same cake is missing and the fork touches the mouth.",
                "cause": "Enjoying the first mouthful makes the hero continue eating.",
            },
            {
                "role": "payoff",
                "caption": "……怎麼沒了？",
                "image_prompt": "The hero looks at an empty plate, cream on the lips.",
                "visible_state": "No cake remains; cream around the hero's mouth.",
                "cause": "Repeated eating has consumed the whole cake, breaking the promise.",
            },
        ],
    }


class ImageFixtureProvider:
    """Comfy T2I saved-file contract; image meaning is outside this local test.

    Real Comfy responses are checked against this consumer contract in the live
    strategy smoke run. Fixtures are genuine image files, not mocked media.
    """

    def __init__(self, failure=""):
        self.failure = failure

    def __call__(self, payload):
        registry = AssetRegistry(REPO)
        assert registry.validate_workflow(payload["workflow_name"])["valid"]
        assert payload["image_count"] == 1
        assert payload["steps"] > 0 and isinstance(payload["seed"], int)
        directory = Path(payload["run_dir"])
        directory.mkdir(parents=True)
        path = directory / "generated.png"
        Image.new("RGB", (payload["width"], payload["height"]), "#FFC0CB").save(path)
        if directory.name == "panel-2" and self.failure == "corrupt":
            path.write_bytes(b"interrupted image download")
        saved = (
            []
            if directory.name == "panel-2" and self.failure == "empty"
            else [str(path)]
        )
        summary = directory / "comfy_summary.json"
        summary.write_text(
            json.dumps({"payload": payload, "saved_files": [str(path)]}),
            encoding="utf-8",
        )
        return {
            "run_dir": str(directory),
            "saved_files": saved,
            "summary_path": str(summary),
            "workflow_name": payload["workflow_name"],
            "memory_retry_count": 0,
        }


def runtime(output, response=None, *, mode="llm", image_failure=""):
    tools, skills = ToolRegistry(), SkillRegistry()
    register_media_service_tools(tools, output)
    tools.register(
        "asset.ensure_workflow_ready",
        lambda p: AssetRegistry(REPO).validate_workflow(p["workflow_name"]),
        "Real manifest validation; external assets outside this test",
    )
    tools.register(
        "comfy.workflow.text_to_image",
        ImageFixtureProvider(image_failure),
        "Injected Comfy image provider",
    )
    engine = LLMPromptEngine(
        mode=mode,
        manager=ScriptedLLMManager([response or story_response()], repeat_last=True),
    )
    register_agent_primitive_skills(skills, tools, output, prompt_engine=engine)
    register_motion_comic_skills(skills, tools, output, prompt_engine=engine)
    runner = WorkflowRunner(skills, retry_policy=RetryPolicy(max_attempts=1))
    return TaskPlanner(AssetRegistry(REPO)), runner


def goal():
    return GoalRequest(
        prompt="A hero promises to eat one tiny mouthful",
        media_type="motion_comic",
        duration_seconds=9,
        style="pastel comic",
        constraints={
            "character": "Kirby",
            "seed": 610110,
            "motion_comic": {
                "image_width": 320,
                "image_height": 240,
                "width": 240,
                "height": 426,
            },
        },
    )


def test_user_given_theme_and_character_when_motion_comic_runs_without_a_story_file_then_a_generated_story_and_playable_comic_are_saved(
    tmp_path,
):
    """User Given a theme and character When motion_comic runs without a story path Then LLM-authored three-beat content, real source images, exact captions and a playable comic are saved with provenance and pending visual review."""
    planner, runner = runtime(tmp_path)
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "success", result.to_dict()
    outputs = result.state["motion-comic-compose"]
    recipe = json.loads(Path(outputs["story_path"]).read_text(encoding="utf-8"))
    assert recipe["title"] == story_response()["title"]
    assert [p["caption"] for p in recipe["panels"]] == [
        p["caption"] for p in story_response()["panels"]
    ]
    assert all(Path(p["image_path"]).is_file() for p in recipe["panels"])
    assert len(recipe["panels"]) == 3
    probe = FFmpegAdapter().probe_media(outputs["final_video_path"])
    assert probe["has_video"] and probe["has_audio"]
    assert (probe["width"], probe["height"]) == (240, 426)
    assert abs(probe["duration"] - 9) < 0.05 and abs(probe["frame_rate"] - 24) < 0.01
    assert outputs["perceptual_review"]["status"] == "awaiting_review"
    assert result.state["motion-comic-write"]["prompt_mode"] == "llm"
    assert Path(outputs["summary_path"]).is_file()


def test_user_given_a_reference_derived_style_profile_when_motion_comic_renders_llm_panels_then_each_krea_prompt_retains_the_profile_rules(
    tmp_path,
):
    """User Given a reference-derived style profile and an LLM story with a short style label When motion-comic panels are rendered Then each Krea prompt retains the profile's medium, composition, palette, and avoidance rules."""
    request = goal()
    profile = {
        "label": "hand-inked cartoon", "prompt": "bold ink outlines and crisp cel shading",
        "composition": "one readable focal action", "palette": "coral and cyan accents",
        "avoid": "fake lettering and interface overlays",
    }
    request.constraints["visual_style_profile"] = profile
    planner, runner = runtime(tmp_path)
    plan = planner.build_plan(request)
    node = next(node for node in plan.nodes if node.skill_name == "media.motion_comic.panels")
    state = RunState(goal={"prompt": request.prompt}, metadata=plan.metadata)
    written = story_response()
    written["run_dir"] = str(tmp_path)
    state["motion-comic-write"] = written

    result = runner.skill_registry.get(node.skill_name).handler(SkillContext(plan, node, state))

    assert result.status == "success", result.to_dict()
    panel_records = result.outputs["generation"]
    assert len(panel_records) == 3
    for record in panel_records:
        prompt = record["payload"]["prompt"]
        assert profile["label"] in prompt
        assert story_response()["visual_style"] not in prompt
        for key in ("prompt", "composition", "palette", "avoid"):
            assert profile[key] in prompt


@pytest.mark.parametrize(
    ("canvas_width", "canvas_height"), [(240, 426), (576, 1024)]
)
def test_user_given_long_but_valid_captions_when_comic_uses_a_portrait_canvas_then_text_reflows_and_video_is_composed(
    tmp_path, canvas_width, canvas_height
):
    """User Given valid captions near the story contract limit When a comic uses minimum or default portrait canvas dimensions Then the compositor measures the actual font and panel space, reflows the captions, and saves a playable video."""
    response = deepcopy(story_response())
    response["panels"][0]["caption"] = "陽光照進明亮的療養院，MetaKnight站在一"
    response["panels"][1]["caption"] = "AI語音傳出溫暖的聲音，長者的表情開始緩和"
    response["panels"][2]["caption"] = "長者露出開心微笑，MetaKnight的守護姿態"
    request = goal()
    request.constraints["motion_comic"].update(
        {"width": canvas_width, "height": canvas_height}
    )
    planner, runner = runtime(tmp_path, response)

    result = runner.run(planner.build_plan(request))

    assert result.status == "success", result.to_dict()
    outputs = result.state["motion-comic-compose"]
    page_paths = sorted(
        (Path(outputs["story_path"]).parent / "render").glob("page-*.png")
    )
    assert len(page_paths) == 3
    assert all(
        Image.open(path).size == (canvas_width, canvas_height) for path in page_paths
    )
    probe = FFmpegAdapter().probe_media(outputs["final_video_path"])
    assert probe["has_video"] and (probe["width"], probe["height"]) == (
        canvas_width,
        canvas_height,
    )


def test_user_given_a_caption_with_a_latin_name_when_text_wraps_then_the_name_stays_intact_on_a_line():
    """User Given a caption containing a Latin character name When measured text wraps Then the name remains intact when it fits within one line."""
    font = load_story_card_font(resolve_story_card_font(), 27)

    lines = wrap_caption("陽光照進明亮的療養院，MetaKnight站在一", font, 215)

    assert any("MetaKnight" in line for line in lines)


def test_user_given_a_rendered_comic_when_preparing_social_review_then_only_the_final_video_and_its_actual_story_are_selected(
    tmp_path,
):
    """User Given generated panels and a complete comic When its output enters social review preparation Then only the final MP4 is selected and caption generation receives the rendered story and contact sheet; provider dispatch is outside this local boundary."""
    planner, runner = runtime(tmp_path)
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "success", result.to_dict()
    data = result.to_dict()
    assert collect_media_paths_from_run_result(data) == [
        result.state["motion-comic-compose"]["final_video_path"]
    ]
    publish_context, source = _resolve_publish_prompt(
        data, fallback_prompt="An unrelated earlier theme"
    )
    assert source == "motion_comic"
    for panel in story_response()["panels"]:
        assert panel["caption"] in publish_context
        assert panel["visible_state"] in publish_context
    assert "An unrelated earlier theme" not in publish_context
    grounding = _extract_publish_visual_grounding(data)
    assert Path(grounding["contact_sheet_path"]).is_file()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_beat",
        "wrong_order",
        "unsupported_caption",
        "non_english_direction",
        "missing_state",
    ],
)
def test_user_given_an_invalid_generated_story_when_the_strategy_runs_then_it_stops_before_images_or_video_are_created(
    tmp_path, damage
):
    """User Given a generated story violating the three-beat text/visual contract When the strategy runs Then it fails before rendering instead of using a stock story or publishing partial content."""
    response = deepcopy(story_response())
    if damage == "missing_beat":
        response["panels"].pop()
    elif damage == "wrong_order":
        response["panels"].reverse()
    elif damage == "unsupported_caption":
        response["panels"][0]["caption"] = "字" * 25
    elif damage == "non_english_direction":
        response["panels"][0]["image_prompt"] = "角色吃蛋糕"
    else:
        response["panels"][1]["visible_state"] = ""
    planner, runner = runtime(tmp_path, response)
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "failed"
    assert not list(tmp_path.rglob("*.png")) and not list(tmp_path.rglob("*.mp4"))


def test_user_given_template_mode_when_motion_comic_is_requested_then_no_canned_story_or_generated_media_is_used(
    tmp_path,
):
    """User Given no LLM generation mode When motion_comic is requested Then it fails explicitly without falling back to a canned story or creating media."""
    planner, runner = runtime(tmp_path, mode="template")
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "failed"
    assert any("LLM" in log for record in result.records for log in record.logs)
    assert not list(tmp_path.rglob("*.mp4"))


@pytest.mark.parametrize("failure", ["empty", "corrupt"])
def test_user_given_an_incomplete_image_when_generating_a_comic_then_it_preserves_the_draft_and_successful_panel_but_creates_no_final_video(
    tmp_path, failure
):
    """User Given an empty or corrupt second image response When a comic is generated Then the run fails, keeps the LLM draft and earlier image evidence, and never produces a partial publishable video."""
    planner, runner = runtime(tmp_path, image_failure=failure)
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "failed"
    assert list(tmp_path.rglob("story_draft.json"))
    manifests = list(tmp_path.rglob("panel_generation.json"))
    assert manifests
    earlier = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert earlier[0]["role"] == "setup"
    with Image.open(earlier[0]["result"]["saved_files"][0]) as image:
        image.verify()
    assert not any(
        Path(path).suffix == ".mp4"
        for path in collect_media_paths_from_run_result(result.to_dict())
    )
    assert not list(tmp_path.rglob("*.mp4"))


def test_user_given_no_story_or_theme_when_the_llm_writes_a_comic_then_the_run_can_render_the_generated_story(
    tmp_path,
):
    """User Given a resolved character and no theme or story file When the LLM supplies a complete original story Then that story becomes a playable comic and the runtime does not require local recipe input."""
    request = goal()
    request.prompt = ""
    request.constraints.pop("seed")
    planner, runner = runtime(tmp_path)
    result = runner.run(planner.build_plan(request))
    assert result.status == "success", result.to_dict()
    assert Path(result.state["motion-comic-compose"]["final_video_path"]).is_file()
    assert result.state["motion-comic-write"]["title"] == story_response()["title"]


def test_user_given_shared_story_notes_containing_an_initial_prop_state_when_rendering_the_payoff_then_only_its_own_state_is_sent_to_the_image_provider(
    tmp_path,
):
    """User Given LLM continuity notes accidentally describing the original whole cake When the empty-plate payoff is rendered Then image directions use its own empty state plus resolved identity and style, preventing shared planning notes from restoring the food."""
    response = story_response()
    response["continuity"] = (
        "A white plate holding an entire strawberry cake in every scene."
    )
    request = goal()
    request.constraints["seed"] = 0
    request.constraints["character_profile"] = {
        "role_description": "a round pink hero with red feet"
    }
    planner, runner = runtime(tmp_path, response)
    result = runner.run(planner.build_plan(request))
    assert result.status == "success", result.to_dict()
    generations = result.state["motion-comic-panels"]["generation"]
    payoff_prompt = generations[-1]["payload"]["prompt"]
    assert response["panels"][-1]["visible_state"] in payoff_prompt
    assert "entire strawberry cake in every scene" not in payoff_prompt
    assert "a round pink hero with red feet" in payoff_prompt
    assert response["visual_style"] in payoff_prompt
    assert generations[0]["payload"]["seed"] == 0


def test_user_given_a_motion_comic_weight_when_character_routing_runs_then_the_strategy_has_defaults_and_a_complete_registered_plan(
    tmp_path,
):
    """User Given positive motion_comic random weight When a character request selects its route Then nine-second portrait defaults, one image per beat and registered runtime steps require no story file or video model."""
    config = yaml.safe_load(
        (REPO / "configs/characters/kirby.yaml").read_text(encoding="utf-8")
    )
    assert config["generation"]["generation_type_weights"]["motion_comic"] > 0
    config["generation"]["generation_type_weights"] = {"motion_comic": 1}
    config["generation"]["subject_mode"] = "single"
    config["character"] = {"name": "Kirby"}
    path = tmp_path / "character.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    payload = build_goal_payload_from_character_config(
        make_character_workflow_request(
            REPO,
            path,
            prompt="A hero promises to take one mouthful",
            rng=random.Random(610110),
        )
    )
    assert payload["source_generation_type"] == "motion_comic"
    assert payload["duration_seconds"] == 9
    assert payload["constraints"]["video_speed"] == {"enabled": False, "factor": 1.0}
    assert payload["constraints"]["image_workflow_name"] == "krea2_turbo"
    assert not payload["constraints"]["video_workflow_name"]
    planner, runner = runtime(tmp_path / "runtime")
    plan = planner.build_plan(
        GoalRequest(
            **{
                key: payload[key]
                for key in (
                    "prompt",
                    "media_type",
                    "duration_seconds",
                    "style",
                    "constraints",
                )
            }
        )
    )
    assert all(runner.skill_registry.get(node.skill_name) for node in plan.nodes)
    assert "story_path" not in payload["constraints"]


def test_user_given_the_story_text_when_validating_it_then_visible_states_and_causal_transitions_survive_for_review():
    """User Given valid bilingual story content When its public contract is validated Then the title, captions and visible-state/cause evidence remain available for later image review."""
    normalized = validate_motion_comic_story(story_response())
    assert normalized == story_response()
