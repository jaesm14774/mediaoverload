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


def test_user_given_a_ledger_and_recap_when_the_next_idea_is_requested_then_it_must_grow_from_the_last_hook_and_advance_an_open_thread():
    """User Given the recap of aired episodes and the open threads When the next episode is ideated Then the prompt shows both and asks the writer to start from the goal the last episode handed on, advance an open thread and change one visible state, instead of inventing an unrelated errand."""
    prompt = _idea_request(
        recap_text="第 1 集《信》：Kirby 撿到一封沒有地址的信｜交給下一集：追著星形印記去郵局",
        ledger_text="目前開放中的線索：\n- T1 主線｜這封信到底是寫給誰的（第 1 集埋下，第 1 集最後被碰到）",
        roster_text="- 小明：戴草帽的青年",
    )

    assert "追著星形印記去郵局" in prompt
    assert "這封信到底是寫給誰的" in prompt
    assert "小明" in prompt
    assert "goal the last episode handed on" in prompt.casefold()
    assert "open thread" in prompt.casefold()
    assert "visible state" in prompt.casefold()
    assert "different place" not in prompt.casefold()


def test_user_given_a_continuing_series_when_an_idea_is_requested_then_the_episode_ends_on_a_hook_and_never_wraps_up_the_series():
    """User Given a continuing series When the next idea is requested Then the writer must end on an unanswered hook and must not conclude the series."""
    prompt = _idea_request()

    assert "hook" in prompt.casefold()
    assert "not conclude" in prompt.casefold() or "do not finish" in prompt.casefold()


def test_user_given_a_protagonist_and_story_when_recurring_characters_are_proposed_then_the_validated_list_comes_back():
    """User Given an approved story and protagonist card When the writer proposes recurring characters Then the public engine returns the validated characters, at most two."""
    manager = ScriptedLLMManager(
        [{"companions": [{"name": "小明", "appearance": "戴草帽的青年"}]}]
    )

    companions = LLMPromptEngine(mode="llm", manager=manager).propose_serial_companions(
        story="Kirby wonders where the star candy comes from.",
        character_card="Kirby is a round pink creature.",
        protagonist_name="Kirby",
    )

    assert companions == [{"name": "小明", "appearance": "戴草帽的青年"}]
