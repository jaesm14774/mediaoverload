from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.pair_interaction import PAIR_INTERACTION_CRAFT
from agentic.runtime.prompt_tasks import scenes
from agentic.storyboard import load_storyboard

REPO_ROOT = Path(__file__).resolve().parents[2]

NEWS = {
    "title": "Geography week brings map games to schools",
    "keyword": "geography;map",
    "content": "Schools hold map games and field walks during geography week.",
}


def _pair_context(first: str = "KingDedede", second: str = "Lesser Slow Loris") -> dict:
    return {
        "mode": "two_character_interaction",
        "subjects": [
            {"role": "primary", "name": first, "profile": {"role_description": "a portly blue penguin with a wooden hammer"}},
            {"role": "secondary", "name": second, "profile": {"role_description": "a small round-eyed primate with long slow fingers"}},
        ],
        "interaction_contract": {"required": True, "same_frame": True},
    }


def _single_context() -> dict:
    return {
        "mode": "single",
        "subjects": [{"role": "primary", "name": "Kirby", "profile": {"role_description": "a round pink hero"}}],
        "interaction_contract": {"required": False},
    }


class _SetWriter:
    """Scripted prompt client that records what the writer was asked."""

    def __init__(self, payload):
        self.payload, self.sent = payload, []

    def require_manager(self):
        return object()

    def generate_json(self, manager, system, user_prompt, **kwargs):
        self.sent.append(user_prompt + "\n" + str(kwargs.get("output_contract") or ""))
        return kwargs["validate_response"](self.payload)

    def mark_payload(self, payload):
        return payload

    def template_fallback(self, fallback, exc):
        return fallback

    def generation_error(self, operation, exc=None):
        return RuntimeError(f"{operation}: {exc}")


def _moments(first: str, second: str) -> dict:
    beats = (
        ("curious", f"{first} peers down while {second} clings to its sleeve"),
        ("startled", f"{second} drops from {first}'s hand as {first} lunges to catch it"),
        ("relieved", f"{first} cradles {second} against its chest with both arms"),
        ("proud", f"{second} lifts one slow finger to tap {first}'s chin"),
        ("sleepy", f"{first} sits while {second} curls on its head"),
        ("content", f"{second} reaches the map pin and {first} lifts it higher"),
    )
    return {
        "creative_brief": "a pair and a map",
        "prompt_sets": [
            {
                "label": f"m{i}",
                "presentation": f"{feeling} beat {i}",
                "feeling": feeling,
                "pose": pose,
                "prompt": f"{pose}, soft pastel drawing of {first} and {second}.",
            }
            for i, (feeling, pose) in enumerate(beats, start=1)
        ],
    }


def _image_goal(context: dict, names: tuple[str, ...]) -> GoalRequest:
    return GoalRequest(
        prompt="A pair meets a map",
        media_type="image",
        style="pastel",
        constraints={
            "character": names[0],
            "subject_context": context,
            "news_grounding_required": True,
            "news_context": NEWS,
        },
    )


def test_user_given_two_cast_members_when_news_moments_are_planned_then_the_writer_is_asked_for_one_shared_beat_not_two_posed_characters():
    """User Given a news-driven image run with two selected characters When the six moments are planned Then the writer is told the two are co-leads who answer each other from their own natures, and each pose covers both bodies."""
    writer = _SetWriter(_moments("KingDedede", "Lesser Slow Loris"))

    scenes.expand_goal(
        writer, _image_goal(_pair_context(), ("KingDedede", "Lesser Slow Loris")), "pastel", []
    )

    asked = writer.sent[0]
    assert PAIR_INTERACTION_CRAFT in asked
    assert scenes.IMAGE_PAIR_MOMENT_SET_CONTRACT in asked
    assert scenes.IMAGE_MOMENT_SET_CONTRACT not in asked


def test_user_given_one_cast_member_when_news_moments_are_planned_then_no_pair_wording_reaches_the_writer():
    """User Given a news-driven image run with a single character When the six moments are planned Then the writer's request is free of any pair-interaction wording."""
    writer = _SetWriter(_moments("Kirby", "Kirby"))

    scenes.expand_goal(writer, _image_goal(_single_context(), ("Kirby",)), "pastel", [])

    asked = writer.sent[0]
    assert PAIR_INTERACTION_CRAFT not in asked
    assert scenes.IMAGE_MOMENT_SET_CONTRACT in asked
    assert scenes.IMAGE_PAIR_MOMENT_SET_CONTRACT not in asked


def test_user_given_the_same_character_twice_when_moments_are_planned_then_the_pair_is_still_a_pair():
    """User Given a draw that picked the same character for both slots When the six moments are planned Then the request is still the pair request, because that draw is expected."""
    writer = _SetWriter(_moments("Kirby", "Kirby"))

    scenes.expand_goal(writer, _image_goal(_pair_context("Kirby", "Kirby"), ("Kirby", "Kirby")), "pastel", [])

    assert PAIR_INTERACTION_CRAFT in writer.sent[0]


def _scene_client(sent: list[str]):
    class Client:
        def require_manager(self):
            return object()

        def generate_json(self, manager, system, user_prompt, **kwargs):
            sent.append(user_prompt)
            return {
                "suitable": True,
                "prompt": "KingDedede and Lesser Slow Loris trace a map together.",
                "creative_seed": "1",
                "source": "n/a",
                "source_anchors": [
                    {
                        "source_excerpt": "map games",
                        "visual_mapping": "KingDedede and Lesser Slow Loris trace a map together.",
                    }
                ],
            }

        def mark_payload(self, payload):
            return payload

    return Client()


def test_user_given_two_cast_members_when_the_news_happening_is_chosen_then_the_pair_lives_it_together():
    """User Given a news-driven image run with two selected characters When the one happening is chosen from the article Then the writer is shown both characters and asked for a happening they live together, not one character's."""
    sent: list[str] = []

    scenes.generate_autonomous_scene_prompt(
        _scene_client(sent),
        character="KingDedede",
        subject_context=_pair_context(),
        style="pastel",
        media_type="image",
        news_context=NEWS,
        news_grounding_required=True,
    )

    assert "Lesser Slow Loris" in sent[0]
    assert PAIR_INTERACTION_CRAFT in sent[0]
    assert "the character lives that happening itself" not in sent[0]


def test_user_given_one_cast_member_when_the_news_happening_is_chosen_then_it_is_still_one_characters_happening():
    """User Given a news-driven image run with a single character When the one happening is chosen Then the writer is asked for one character's happening exactly as before."""
    sent: list[str] = []

    scenes.generate_autonomous_scene_prompt(
        _scene_client(sent),
        character="Kirby",
        subject_context=_single_context(),
        style="pastel",
        media_type="image",
        news_context=NEWS,
        news_grounding_required=True,
    )

    assert PAIR_INTERACTION_CRAFT not in sent[0]
    assert "The character lives that happening itself" in sent[0]


class _StoryModel:
    last_success_model = "local-contract-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat_completion(self, *, messages, **_options) -> str:
        self.prompts.append(" ".join(str(item["content"]) for item in messages))
        return json.dumps(
            {
                "story": {
                    "native_shots": [
                        {"time": "0-5s", "action": "KingDedede eyes the map as Lesser Slow Loris clings on."},
                        {"time": "5-10s", "action": "Lesser Slow Loris tugs the map as KingDedede steadies it."},
                        {"time": "10-15s", "action": "KingDedede and Lesser Slow Loris look at the map together."},
                    ]
                }
            }
        )


def _native_prompts(context: dict, character: str, *, loop: bool) -> str:
    model = _StoryModel()
    engine = LLMPromptEngine(
        mode="llm", manager=SimpleNamespace(text_model=model, vision_model=model)
    )
    engine.generate_native_h3_storyboard(
        character=character,
        subject_context=context,
        style="polished 2D anime",
        duration_seconds=15,
        base_storyboard=load_storyboard(REPO_ROOT / "configs/storyboards/native_h3_15s.yaml"),
        news_context={},
        replayable_loop=loop,
    )
    return " ".join(model.prompts)


def test_user_given_two_cast_members_when_a_native_story_is_written_then_the_writer_gets_the_pair_craft():
    """User Given two selected characters and no news When a native H3 story is written, loop or not, Then the writer is told how the two act from their own natures and answer each other."""
    for loop in (True, False):
        prompt = _native_prompts(_pair_context(), "KingDedede and Lesser Slow Loris", loop=loop)
        assert PAIR_INTERACTION_CRAFT in prompt


def test_user_given_one_cast_member_when_a_native_story_is_written_then_no_pair_wording_reaches_the_writer():
    """User Given a single selected character When a native H3 story is written Then the writer's request carries no pair wording."""
    prompt = _native_prompts(_single_context(), "Kirby", loop=True)

    assert PAIR_INTERACTION_CRAFT not in prompt


def test_user_given_a_required_cast_that_is_not_exactly_two_when_the_goal_is_written_then_it_still_asks_for_physical_interaction():
    """User Given a required-interaction cast that does not hold exactly two named subjects When the scene is written Then the writer is still told to show their physical interaction, without the pair craft."""
    context = _pair_context()
    context["subjects"].append({"role": "extra", "name": "Magolor", "profile": {}})
    goal = _image_goal(context, ("KingDedede", "Lesser Slow Loris", "Magolor"))

    instruction = scenes._goal_subject_instruction(goal)

    assert "physical interaction" in instruction
    assert PAIR_INTERACTION_CRAFT not in instruction


def test_user_given_a_sticker_pack_with_a_pair_when_the_goal_is_written_then_no_pair_wording_reaches_the_writer():
    """User Given a sticker pack whose cast came out as a pair When the goal is written Then the writer gets no pair craft and no pair carousel, because a sticker is one isolated subject."""
    goal = GoalRequest(
        prompt="stickers",
        media_type="sticker_pack",
        style="pastel",
        constraints={"character": "KingDedede", "subject_context": _pair_context()},
    )

    assert PAIR_INTERACTION_CRAFT not in scenes._goal_subject_instruction(goal)
