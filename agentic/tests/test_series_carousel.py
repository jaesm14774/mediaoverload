"""User-facing series_carousel acceptance: one fixed scene, only the character changes."""

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
from agentic.runtime.series_carousel import select_series_members
from agentic.skills.agent_primitives import register_agent_primitive_skills
from agentic.skills.comfy_image import register_comfy_image_skills
from agentic.tools.media_services import register_media_service_tools
from character_workflow_helpers import make_character_workflow_request
from provider_fakes import ScriptedLLMManager

REPO = Path(__file__).resolve().parents[2]

CAST = [
    {"name": "Kirby", "role_description": "A round pink hero who inhales enemies.", "keywords": "pink, round"},
    {"name": "Waddle Dee", "role_description": "A small orange helper.", "keywords": "orange, tan face"},
    {"name": "Meta Knight", "role_description": "A masked blue swordsman.", "keywords": "mask, cape"},
    {"name": "King Dedede", "role_description": "A plump penguin king.", "keywords": "blue, red cap"},
]
LOOKS = {
    "Kirby": "the small round pink puffball with rosy cheeks",
    "Waddle Dee": "the small round orange creature with a tan face",
    "Meta Knight": "the small round dark blue knight in a silver mask",
    "King Dedede": "the plump blue penguin king in a red cap",
}
TEMPLATE = (
    "Soft hand-drawn pastel illustration with paper grain. Seen straight on from the front, a small wooden bed sits "
    "low and centred under a round moon window. {character} lies tucked under a star blanket up to the chin, fast "
    "asleep. The sleeper is the only figure in the picture. No writing anywhere."
)


def writer_response(template=TEMPLATE, looks=None):
    looks = LOOKS if looks is None else looks
    return {
        "template": template,
        "looks": [{"name": name, "look": look} for name, look in looks.items()],
    }


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


def without_character(prompt, name):
    return prompt.replace(name, "").replace(LOOKS[name], "")


def test_user_given_a_cast_and_a_scene_when_series_carousel_runs_then_one_page_per_character_is_packaged_from_the_same_scene(
    tmp_path,
):
    """User Given four group members and one scene When series_carousel runs Then one page per member is packaged, each page names only its member, and the page prompts are otherwise identical."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal()))

    assert result.status == "success", result.to_dict()
    saved = result.state["package-carousel"]["saved_files"]
    assert len(saved) == len(CAST) and all(Path(path).is_file() for path in saved)
    prompts = [payload["prompt"] for payload in provider.payloads]
    names = [member["name"] for member in CAST]
    for prompt, name in zip(prompts, names, strict=True):
        assert name in prompt
        assert not any(other in prompt for other in names if other != name)
    assert len({without_character(prompt, name) for prompt, name in zip(prompts, names, strict=True)}) == 1


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
    assert "{character}" not in brief


@pytest.mark.parametrize("requested_seed", [610110, None])
def test_user_given_a_series_when_it_renders_then_every_page_uses_the_same_seed(
    tmp_path, requested_seed
):
    """User Given a series with or without a requested seed When every page is rendered Then all render requests carry one identical seed so the composition holds."""
    planner, runner, provider = runtime(tmp_path, writer_response())

    result = runner.run(planner.build_plan(goal(seed=requested_seed)))

    assert result.status == "success", result.to_dict()
    seeds = {payload["seed"] for payload in provider.payloads}
    assert len(provider.payloads) == len(CAST) and len(seeds) == 1
    if requested_seed is not None:
        assert seeds == {requested_seed}


@pytest.mark.parametrize(
    "template",
    [
        "A small wooden bed under a round moon window, a character asleep under a star blanket.",
        "{character} sleeps in the bed while {character} dreams of stars.",
        "Kirby sleeps in the bed under a star blanket while {character} watches.",
    ],
    ids=["no-slot", "two-slots", "leaves-a-member-name"],
)
def test_user_given_a_template_that_cannot_be_shared_when_the_scene_is_written_then_the_run_fails_before_any_page_is_rendered(
    tmp_path, template
):
    """User Given a scene template without exactly one character slot or one that still names a member When series_carousel runs Then it fails and no page is rendered."""
    planner, runner, provider = runtime(tmp_path, writer_response(template))

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"
    assert provider.payloads == []


def test_user_given_a_writer_that_skips_a_member_when_the_scene_is_written_then_the_run_fails_before_any_page_is_rendered(
    tmp_path,
):
    """User Given a writer that gives no look for one member When series_carousel runs Then it fails and no page is rendered."""
    looks = {name: look for name, look in LOOKS.items() if name != "King Dedede"}
    planner, runner, provider = runtime(tmp_path, writer_response(looks=looks))

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"
    assert provider.payloads == []


@pytest.mark.parametrize(
    "bad_look",
    [
        "the small round pink puffball just like Waddle Dee",
        "the small round pink puffball with a {character} slot",
        "the small round pink puffball with rosy cheeks and a very long list of other features",
    ],
    ids=["names-another-member", "contains-a-slot", "too-long"],
)
def test_user_given_a_look_that_would_change_the_composition_when_the_scene_is_written_then_the_run_fails_before_any_page_is_rendered(
    tmp_path, bad_look
):
    """User Given a look that names another member, holds a slot, or runs long When series_carousel runs Then it fails and no page is rendered."""
    looks = {**LOOKS, "Kirby": bad_look}
    planner, runner, provider = runtime(tmp_path, writer_response(looks=looks))

    result = runner.run(planner.build_plan(goal()))

    assert result.status != "success"
    assert provider.payloads == []


def test_user_given_a_page_that_renders_no_image_when_the_series_is_packaged_then_the_run_fails_instead_of_publishing_a_shorter_set(
    tmp_path,
):
    """User Given one page whose render returns no image When series_carousel runs Then the run fails rather than packaging fewer pages than the cast."""
    planner, runner, _ = runtime(tmp_path, writer_response(), empty_page="meta-knight")

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
