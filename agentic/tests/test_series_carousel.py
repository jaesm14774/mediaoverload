"""User-facing series_carousel acceptance: one shared world, each character its own moment and spot."""

import json
from pathlib import Path
import random

from PIL import Image
import pytest

from agentic.app.character_payload import build_goal_payload_from_character_config
from agentic.app.character_publishing import (
    _resolve_publish_prompt,
    collect_media_paths_from_run_result,
)
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.creativity import RetryPolicy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.runtime.series_carousel import select_series_members, series_prompt_sets
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.comfy_image import register_comfy_image_skills
from agentic.tools.media_services import register_media_service_tools
from character_workflow_helpers import make_character_workflow_request
from provider_fakes import ScriptedLLMManager

REPO = Path(__file__).resolve().parents[2]

CAST = [
    {
        "name": "Kirby",
        "role_description": "Kirby is a small pink spherical creature with two stubby arms, two red feet, oval eyes, and rosy cheeks.",
        "keywords": "pink, spherical body",
    },
    {
        "name": "Waddle Dee",
        "role_description": "Waddle Dee is a tan, round-bodied creature with a pear-shaped face, no mouth, and honey-colored feet.",
        "keywords": "tan, no mouth",
    },
    {
        "name": "Magolor",
        "role_description": "Magolor is a short brown alien with no feet, detached hands in cream mittens, and a hood.",
        "keywords": "brown alien, no feet",
    },
    {
        "name": "Parasol Waddle Dee",
        "role_description": "Parasol Waddle Dee is a tan, round-bodied Waddle Dee with no mouth and a striped parasol.",
        "keywords": "tan, parasol",
    },
]
SCENE = (
    "Soft hand-drawn pastel illustration with paper grain. A small wooden bed sits under a round moon window "
    "while night light pools across the room. No writing anywhere."
)
PAGES = [
    {
        "name": "Kirby",
        "moment": "Kirby lies tucked under a star blanket up to the chin, fast asleep.",
        "placement": "bottom-left",
    },
    {
        "name": "Waddle Dee",
        "moment": "Waddle Dee peeks over the bed frame with drowsy eyes.",
        "placement": "top-right",
    },
    {
        "name": "Magolor",
        "moment": "Magolor floats just above the pillow with its hood pulled low.",
        "placement": "center",
    },
    {
        "name": "Parasol Waddle Dee",
        "moment": "Parasol Waddle Dee holds its striped parasol open like a canopy over the bed.",
        "placement": "bottom-right",
    },
]


def writer_response(scene=SCENE, pages=PAGES):
    return {"scene": scene, "pages": [dict(page) for page in pages]}


def with_page(index, **changes):
    pages = [dict(page) for page in PAGES]
    pages[index].update(changes)
    return pages


def identity(member):
    """The part of the roster description an image prompt must carry for this member."""
    return member["role_description"].split(" is ", 1)[1].rstrip(".")


class RecordingImageProvider:
    """Comfy T2I saved-file contract; every request is kept for inspection."""

    def __init__(self, empty_page=""):
        self.payloads = []
        self.empty_page = empty_page

    def __call__(self, payload):
        self.payloads.append(payload)
        directory = Path(payload["run_dir"])
        directory.mkdir(parents=True)
        path = directory / "generated.png"
        Image.new("RGB", (payload["width"], payload["height"]), "#FFC0CB").save(path)
        summary = directory / "comfy_summary.json"
        summary.write_text(json.dumps({"saved_files": [str(path)]}), encoding="utf-8")
        return {
            "run_dir": str(directory),
            "saved_files": [] if directory.name == self.empty_page else [str(path)],
            "summary_path": str(summary),
            "workflow_name": payload["workflow_name"],
            "memory_retry_count": 0,
        }


def runtime(output, response, *, mode="llm", empty_page=""):
    tools, skills = ToolRegistry(), SkillRegistry()
    register_media_service_tools(tools, output)
    tools.register(
        "asset.ensure_workflow_ready",
        lambda p: AssetRegistry(REPO).validate_workflow(p["workflow_name"]),
        "Real manifest validation; external assets outside this test",
    )
    provider = RecordingImageProvider(empty_page)
    tools.register("comfy.workflow.text_to_image", provider, "Injected Comfy image provider")
    engine = LLMPromptEngine(mode=mode, manager=ScriptedLLMManager([response], repeat_last=True))
    register_agent_primitive_skills(skills, tools, output, prompt_engine=engine)
    register_comfy_image_skills(skills, tools, output)
    runner = WorkflowRunner(skills, retry_policy=RetryPolicy(max_attempts=1))
    return TaskPlanner(AssetRegistry(REPO)), runner, provider


def goal(*, seed=610110, members=CAST):
    return GoalRequest(
        prompt="Kirby falls asleep under a star blanket after a long day",
        media_type="carousel",
        style="pastel storybook",
        constraints={
            "character": "Kirby",
            "seed": seed,
            "series_members": [dict(member) for member in members],
            "image_width": 320,
            "image_height": 400,
        },
    )


def test_user_given_a_cast_and_a_scene_when_series_carousel_runs_then_one_page_per_character_is_packaged_in_the_same_world(
    tmp_path,
):
    """User Given four group members and one shared world When series_carousel runs Then one page per member is packaged and every page prompt carries the same world."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal()))

    assert result.status == "success", result.to_dict()
    saved = result.state["package-carousel"]["saved_files"]
    assert len(saved) == len(CAST) and all(Path(path).is_file() for path in saved)
    assert all(SCENE in payload["prompt"] for payload in provider.payloads)


def test_user_given_a_cast_when_series_carousel_runs_then_every_page_shows_its_own_moment_and_the_set_is_not_one_composition(
    tmp_path,
):
    """User Given a writer that gives each member a different moment and spot When series_carousel runs Then each page prompt carries only its own member's moment and the pages do not all sit in one place."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal()))

    assert result.status == "success", result.to_dict()
    prompts = [payload["prompt"] for payload in provider.payloads]
    for prompt, page in zip(prompts, PAGES, strict=True):
        assert page["moment"] in prompt
        assert not any(other["moment"] in prompt for other in PAGES if other is not page)
    placements = [item["placement"] for item in result.state["series-prompts"]["prompt_sets"]]
    assert len(set(placements)) > 1 and placements.count("center") <= 1


def test_user_given_a_roster_with_verified_descriptions_when_series_carousel_runs_then_every_page_carries_its_own_roster_description_unchanged(
    tmp_path,
):
    """User Given members whose roster says Waddle Dee has no mouth and Magolor has no feet When series_carousel runs Then each page prompt carries that member's roster description word for word, and no other member's."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal()))

    assert result.status == "success", result.to_dict()
    prompts = [payload["prompt"] for payload in provider.payloads]
    for prompt, member in zip(prompts, CAST, strict=True):
        assert identity(member) in prompt
        assert member["name"] in prompt
        assert not any(
            identity(other) in prompt for other in CAST if other is not member
        )


def test_user_given_a_member_without_a_roster_description_when_series_carousel_runs_then_the_run_fails_before_any_page_is_rendered(
    tmp_path,
):
    """User Given a cast member whose roster has no description When series_carousel runs Then it fails instead of drawing a page from the bare name."""
    planner, runner, provider = runtime(tmp_path, writer_response())
    members = [dict(member) for member in CAST]
    members[1]["role_description"] = ""

    result = runner.run(planner.build_plan(goal(members=members)))

    assert result.status != "success"
    assert provider.payloads == []


def test_user_given_a_scene_when_the_writer_is_briefed_then_it_sees_every_member_description():
    """User Given a cast whose members differ in mouth and feet When the scene is requested Then the writer is shown every member's description."""
    manager = ScriptedLLMManager([writer_response()], repeat_last=True)
    engine = LLMPromptEngine(mode="llm", manager=manager)

    engine.build_series_template(goal(), [dict(member) for member in CAST])

    briefing = json.dumps(
        [request["messages"] for request in manager.text_model.requests], ensure_ascii=False
    )
    assert all(identity(member) in briefing for member in CAST)


def test_user_given_a_news_happening_when_the_scene_is_requested_then_the_writer_is_briefed_on_the_source_mapping():
    """User Given a run whose scene came from a news happening When the series scene is requested Then the writer sees that happening's source mapping and not only the finished scene text."""
    manager = ScriptedLLMManager([writer_response()], repeat_last=True)
    engine = LLMPromptEngine(mode="llm", manager=manager)
    news_goal = goal()
    news_goal.constraints["news_source_anchors"] = [
        {"source_excerpt": "rain kept falling", "visual_mapping": "keeps dancing as the rain keeps falling"}
    ]

    engine.build_series_template(news_goal, [dict(member) for member in CAST])

    briefing = json.dumps(
        [request["messages"] for request in manager.text_model.requests], ensure_ascii=False
    )
    assert "keeps dancing as the rain keeps falling" in briefing


def test_user_given_a_finished_series_when_it_is_published_then_every_page_goes_out_and_the_copy_is_briefed_on_the_whole_cast(
    tmp_path,
):
    """User Given a finished series When it goes to publish review Then every page is a publish candidate and the caption brief names each cast member instead of only the lead."""
    planner, runner, _ = runtime(tmp_path, writer_response())
    result = runner.run(planner.build_plan(goal()))
    assert result.status == "success", result.to_dict()
    data = result.to_dict()

    assert collect_media_paths_from_run_result(data) == result.state["package-carousel"]["saved_files"]
    brief, source = _resolve_publish_prompt(data, fallback_prompt="Kirby falls asleep")
    assert source == "series_carousel"
    assert all(member["name"] in brief for member in CAST)
    assert all(page["moment"] in brief for page in PAGES)


def test_user_given_a_requested_seed_when_a_series_renders_then_each_page_gets_its_own_seed_from_it(
    tmp_path,
):
    """User Given a requested seed When the series renders Then the first page uses it and no two pages start from the same noise, so their compositions can differ."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal(seed=610110)))

    assert result.status == "success", result.to_dict()
    seeds = [payload["seed"] for payload in provider.payloads]
    assert seeds[0] == 610110 and len(set(seeds)) == len(CAST)


@pytest.mark.parametrize(
    "response",
    [
        writer_response(scene=""),
        writer_response(scene="Kirby sleeps in the bed under a star blanket."),
        writer_response(pages=PAGES[:3]),
        writer_response(pages=PAGES[::-1]),
        writer_response(pages=with_page(0, moment="Magolor hugs a pillow.")),
        writer_response(pages=with_page(0, moment="A pink creature naps.")),
        writer_response(pages=with_page(1, placement="somewhere nice")),
        writer_response(pages=[{**page, "placement": "center"} for page in PAGES]),
        writer_response(pages=[{**page, "placement": "left"} for page in PAGES]),
        writer_response(scene="A small wooden bed under {character} and a round moon window."),
        writer_response(pages=with_page(3, moment="Waddle Dee holds a striped parasol.")),
    ],
    ids=[
        "empty-scene",
        "scene-names-a-member",
        "member-without-a-page",
        "pages-out-of-cast-order",
        "moment-about-another-member",
        "moment-never-names-its-member",
        "unknown-placement",
        "everyone-centred",
        "everyone-in-one-spot",
        "leftover-template-slot",
        "moment-names-a-member-by-a-shorter-name",
    ],
)
def test_user_given_a_series_that_cannot_be_drawn_when_the_scene_is_written_then_the_run_fails_before_any_page_is_rendered(
    tmp_path, response
):
    """User Given a writer result with an empty or member-naming scene, a page missing or misattributed, or every page stacked in one spot When series_carousel runs Then it fails and no page is rendered."""
    planner, runner, provider = runtime(tmp_path, response)

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"
    assert provider.payloads == []


def test_user_given_a_two_page_series_when_both_pages_sit_in_one_spot_then_the_scene_is_refused():
    """User Given the smallest carousel with both characters placed in the same spot When the pages are built Then it is refused so the pair cannot be one composition."""
    pair = [dict(member) for member in CAST[:2]]
    pages = [{**page, "placement": "left"} for page in PAGES[:2]]

    with pytest.raises(ValueError, match="stack"):
        series_prompt_sets(SCENE, pages, pair)


def test_user_given_a_page_that_renders_no_image_when_the_series_is_packaged_then_the_run_fails_instead_of_publishing_a_shorter_set(
    tmp_path,
):
    """User Given one page whose render returns no image When series_carousel runs Then the run fails rather than packaging fewer pages than the cast."""
    planner, runner, _ = runtime(tmp_path, writer_response(), empty_page="magolor")

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"


def test_user_given_no_llm_when_series_carousel_runs_then_it_refuses_instead_of_using_a_canned_scene(
    tmp_path,
):
    """User Given template mode When series_carousel runs Then no canned scene is invented and nothing is rendered."""
    planner, runner, provider = runtime(tmp_path, writer_response(), mode="template")

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"
    assert provider.payloads == []


def test_user_given_a_group_when_the_cast_is_chosen_then_the_primary_leads_and_every_member_is_distinct():
    """User Given a group of six active members When a four-page cast is chosen Then the primary is page one and the others are distinct group members."""
    group = [
        {"name": name, "role_description": "", "keywords": "", "status": 1, "weight": 1.0}
        for name in ("Kirby", "Waddle Dee", "Meta Knight", "King Dedede", "Bandana Dee", "Magolor")
    ]

    for seed in range(20):
        cast = select_series_members(
            group, primary="Meta Knight", page_count=4, rng=random.Random(seed)
        )
        names = [member["name"] for member in cast]
        assert names[0] == "Meta Knight"
        assert len(set(names)) == 4 and set(names) <= {member["name"] for member in group}


@pytest.mark.parametrize("page_count", [1, 11])
def test_user_given_a_page_count_a_carousel_cannot_hold_when_the_cast_is_chosen_then_it_fails_clearly(
    page_count,
):
    """User Given a page count outside what social carousels hold When the cast is chosen Then selection fails and says why."""
    group = [{"name": f"Member {index}", "status": 1, "weight": 1.0} for index in range(12)]

    with pytest.raises(ValueError, match="page count"):
        select_series_members(group, primary="Member 0", page_count=page_count, rng=random.Random(0))


def test_user_given_a_roster_that_lists_a_name_twice_when_the_cast_is_chosen_then_no_member_appears_twice():
    """User Given a group roster with a duplicated name When a cast is chosen Then every page still has a different character."""
    group = [
        {"name": name, "status": 1, "weight": 1.0}
        for name in ("Kirby", "Waddle Dee", "Waddle Dee", "Meta Knight", "Meta Knight", "King Dedede")
    ]

    for seed in range(20):
        names = [
            member["name"]
            for member in select_series_members(
                group, primary="Kirby", page_count=4, rng=random.Random(seed)
            )
        ]
        assert len(set(names)) == 4


def test_user_given_a_group_smaller_than_the_page_count_when_the_cast_is_chosen_then_it_fails_clearly():
    """User Given a group with fewer active members than pages When the cast is chosen Then selection fails and says why."""
    group = [
        {"name": name, "role_description": "", "keywords": "", "status": 1, "weight": 1.0}
        for name in ("Kirby", "Waddle Dee", "Meta Knight")
    ]

    with pytest.raises(ValueError, match="4 .*3"):
        select_series_members(group, primary="Kirby", page_count=4, rng=random.Random(0))


def test_user_given_a_kirby_group_when_series_carousel_is_requested_then_the_goal_carries_a_full_cast():
    """User Given a resolved Kirby group When the series_carousel goal is built Then it is a carousel goal whose cast starts with Kirby and has one distinct member per requested image."""
    group = [
        {"name": name, "role_description": f"{name} profile", "keywords": "k", "status": 1, "weight": 1.0}
        for name in ("Kirby", "Waddle Dee", "Meta Knight", "King Dedede", "Bandana Dee")
    ]
    selection = {
        "mode": "group",
        "group_name": "Kirby",
        "selected_character": "Kirby",
        "candidate_count": len(group),
        "candidates": group,
        "selected_profile": {"role_description": "Kirby profile", "keywords": "k"},
        "selection_source": "fixed_config_role",
    }
    payload = build_goal_payload_from_character_config(
        make_character_workflow_request(
            REPO,
            REPO / "configs" / "characters" / "kirby.yaml",
            prompt="Kirby falls asleep under a star blanket",
            preferred_generation_type="series_carousel",
            selected_character_name="Kirby",
            character_selection=selection,
            publish_after_generate=False,
        )
    )

    assert payload["media_type"] == "carousel"
    members = payload["constraints"]["series_members"]
    assert members[0]["name"] == "Kirby"
    assert len({member["name"] for member in members}) == len(members) == payload["routing"]["count_plan"]["image_count"]
