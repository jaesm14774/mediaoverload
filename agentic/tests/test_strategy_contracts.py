"""Acceptance boundaries: planning uses real manifests and registered runtime skills.

External GPU, LLM and social calls are deliberately outside these graph checks.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from agentic.app.main import build_runtime
from agentic.app.character_payload import build_goal_payload_from_character_config
from character_workflow_helpers import make_character_workflow_request
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.prompt_requests import GenerationRoutingRequest
from agentic.runtime.strategies import media_type_for_generation


REPO = Path(__file__).resolve().parents[2]
ROUTING = yaml.safe_load((REPO / "configs/routing.yaml").read_text(encoding="utf-8"))[
    "routing"
]


def test_user_given_an_unknown_generation_strategy_when_resolving_its_media_type_then_it_is_rejected():
    """User Given an unknown generation strategy When its planner media type is resolved Then it fails instead of silently routing to long video."""
    with pytest.raises(ValueError, match="Unsupported generation strategy: custom_strategy"):
        media_type_for_generation("custom_strategy")


@pytest.mark.parametrize(
    ("generation_type", "expected_media_type"),
    [
        ("text2longvideo", "long_video"),
        ("native_h3_story", "native_h3_story"),
        ("text2image2video", "text2img2video"),
    ],
)
def test_user_given_a_configured_generation_strategy_when_resolving_its_media_type_then_the_planner_contract_is_used(
    generation_type, expected_media_type
):
    """User Given a configured generation strategy When resolving its planner media type Then its declared planner contract is returned."""
    assert media_type_for_generation(generation_type) == expected_media_type


@pytest.mark.parametrize("generation_type", ROUTING["workflow_stage_candidates"])
def test_user_given_a_configured_strategy_when_planned_then_every_step_can_be_executed(
    generation_type, tmp_path
):
    """User Given a configured character strategy When its real payload is planned Then all skills/tools exist and dependencies form a complete acyclic graph."""
    planner, runner, _ = build_runtime(REPO, output_root=tmp_path)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (608, 352), "green").save(reference)
    payload = build_goal_payload_from_character_config(
        make_character_workflow_request(
            REPO,
            REPO / "configs/characters/kirby.yaml",
            prompt="A hero catches a falling seed",
            preferred_generation_type=generation_type,
            duration_seconds=15,
        )
    )
    goal = GoalRequest(
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
    goal.constraints["native_h3_reference_image_paths"] = [str(reference)]
    goal.constraints["skip_upscale_for_i2v"] = True
    plan = planner.build_plan(goal)
    ordered = plan.as_graph().topological_order()
    assert ordered
    assert len({node.node_id for node in ordered}) == len(plan.nodes)
    for node in ordered:
        assert node.skill_name in runner.skill_registry.list_names()
        if node.tool_name:
            assert node.tool_name in runner.tool_registry.list_names()


def test_user_given_an_unknown_media_type_when_planning_then_the_request_is_rejected():
    """User Given an unsupported media type When planning Then it fails clearly before selecting an unrelated generation workflow."""
    planner = TaskPlanner(AssetRegistry(REPO))
    with pytest.raises(ValueError, match="Unsupported media type"):
        planner.build_plan(GoalRequest(prompt="a hero", media_type="long_vdieo"))


@pytest.mark.parametrize("selected", ["missing-model", "minimax_h3_i2va"])
def test_user_given_an_explicit_incompatible_model_when_planning_then_it_is_not_substituted(
    selected,
):
    """User Given an absent or incompatible explicit image workflow When planning Then a configuration error names it instead of choosing another model."""
    planner = TaskPlanner(AssetRegistry(REPO))
    with pytest.raises(ValueError, match=selected):
        planner.build_plan(
            GoalRequest(
                prompt="a hero",
                media_type="image",
                constraints={"workflow_name": selected},
            )
        )


def test_user_given_story_cards_when_llm_routes_then_unrelated_video_counts_are_not_required():
    """User Given an image-only story-card strategy When the LLM returns its applicable counts Then unrelated video policies cannot invalidate the route."""

    # A deterministic ChatModel at the external provider boundary; the same
    # chat_completion interface is exercised over HTTP by provider contract tests.
    class RouteModel:
        def chat_completion(self, messages, **kwargs):
            return '{"generation_type":"story_card","workflow_plan":{"image_workflow_name":"krea2_turbo","video_workflow_name":"","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":1,"review_selection_limit":1},"reason":"written story cards"}'

    class RouteManager:
        text_model = RouteModel()
        vision_model = RouteModel()

    engine = LLMPromptEngine(mode="llm", manager=RouteManager())
    result = engine.route_generation_strategy(
        GenerationRoutingRequest(
            prompt="a story card",
            character="a hero",
            style="storybook",
            generation_type_candidates=("story_card",),
            workflow_stage_candidates={
                "story_card": {"image_workflow_name": ["krea2_turbo"]}
            },
            count_policies={
                "story_card": {
                    "image_count": {"min": 1, "max": 6},
                    "review_selection_limit": {"min": 1, "max": 6},
                    "video_count": {"min": 1, "max": 3},
                }
            },
        )
    )
    assert result["generation_type"] == "story_card"
    assert "video_count" not in result["count_plan"]


@pytest.mark.parametrize(
    "media_type,workflow",
    [("text2video", "minimax_h3_t2va"), ("game_sprite", "minimax_h3_i2va")],
)
def test_user_given_an_explicit_video_model_when_composed_planning_then_it_is_used_only_for_the_video_stage(
    media_type, workflow
):
    """User Given a selected primary video model When a composed strategy is planned Then the video uses it and its image stage keeps a compatible image model."""
    planner = TaskPlanner(AssetRegistry(REPO))
    plan = planner.build_plan(
        GoalRequest(
            prompt="A bouncing star",
            media_type=media_type,
            constraints={"workflow_name": workflow},
        )
    )
    assert workflow in plan.metadata["selected_workflows"]
    assert "krea2_turbo" in plan.metadata["selected_workflows"]


@pytest.mark.parametrize(
    "kind,expected",
    [("video", "final_video"), ("mixed", "final_media"), ("empty", "final_media")],
)
def test_user_given_a_publish_folder_when_planning_review_then_the_review_scope_matches_its_media(
    kind, expected, tmp_path
):
    """User Given a folder of videos, mixed media or no media When review is planned Then the scope matches the actual artifact types."""
    if kind != "empty":
        (tmp_path / "clip.mp4").write_bytes(
            b"scope-only media listing; this test does not decode"
        )
    if kind == "mixed":
        Image.new("RGB", (32, 32), "green").save(tmp_path / "still.png")
    (tmp_path / "subfolder").mkdir()
    plan = TaskPlanner(AssetRegistry(REPO)).build_plan(
        GoalRequest(
            prompt="Review existing media",
            media_type="publish_review",
            constraints={"input_dir": str(tmp_path), "dry_run": True},
        )
    )
    review = next(
        node for node in plan.nodes if node.skill_name == "review.assets.select"
    )
    assert review.inputs["review_scope"] == expected
