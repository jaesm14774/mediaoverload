from agentic.runtime.llm_engine import LLMPromptEngine
from provider_fakes import ScriptedLLMManager


def _idea_request(**overrides) -> str:
    """Run the public idea generator against a scripted provider and return the prompt it sent."""
    manager = ScriptedLLMManager(
        [{"topic": "會唱歌的洞穴", "premise": "Kirby follows a hum into a cave and finds a glowing shell."}]
    )
    arguments = dict(
        character="Kirby",
        automation_mode="continue",
        story_context="Kirby wonders where the star candy comes from.",
    )
    arguments.update(overrides)
    idea = LLMPromptEngine(mode="llm", manager=manager).generate_serial_episode_idea(**arguments)
    assert idea["topic"]
    return "\n".join(
        str(message["content"])
        for message in manager.text_model.requests[0]["messages"]
    )


def test_user_given_a_new_series_when_its_first_episode_is_ideated_then_it_is_only_the_opening_and_leaves_the_series_question_open():
    """User Given an approved series story When episode 1 is ideated Then the writer is asked for an opening beat that never finishes the series goal, so episode 2 still has something to pursue."""
    prompt = _idea_request(automation_mode="new")

    assert "開場" in prompt or "opening" in prompt.casefold()
    assert "do not complete" in prompt.casefold()
    assert "hook" in prompt.casefold()


def test_user_given_aired_episodes_when_the_next_idea_is_requested_then_their_actions_are_shown_and_must_not_be_repeated():
    """User Given two aired episodes When the next episode is ideated Then the prompt lists what they already did and forbids repeating the same errand, so a delivered letter is never delivered again."""
    prompt = _idea_request(
        previous_episodes=[
            "Episode 1: Kirby delivered a letter to a friend's door.",
            "Episode 2: Kirby chased a hat across the harbor.",
        ]
    )

    assert "delivered a letter" in prompt
    assert "chased a hat" in prompt
    assert "do not repeat" in prompt.casefold()
    assert "different place" in prompt.casefold()


def test_user_given_a_continuing_series_when_an_idea_is_requested_then_the_episode_ends_on_a_hook_and_never_wraps_up_the_series():
    """User Given a continuing series When the next idea is requested Then the writer must end on an unanswered hook and must not conclude the series."""
    prompt = _idea_request()

    assert "hook" in prompt.casefold()
    assert "not conclude" in prompt.casefold() or "do not finish" in prompt.casefold()
