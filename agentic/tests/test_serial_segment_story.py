from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from provider_fakes import ScriptedLLMManager


def _segment_request(*, serial: bool, segment_count: int = 4) -> str:
    segments = [
        {
            "segment_id": f"s{index}",
            "visual": "Kirby hops in the meadow",
            "narration": "",
            "action": "hop",
            "cause": "wind",
            "effect": "leaf spins",
            "camera": "locked",
        }
        for index in range(segment_count)
    ]
    manager = ScriptedLLMManager([{"segments": segments}])
    goal = GoalRequest(
        prompt="四個節拍：開場、問題、升級、解決與鉤子。",
        media_type="long_video",
        duration_seconds=60,
        style="watercolor",
        constraints={"character": "Kirby", **({"serial_mode": True} if serial else {})},
    )
    LLMPromptEngine(mode="llm", manager=manager).segment_story(
        goal, "brief", segment_count, "playful"
    )
    return "\n".join(
        str(message["content"]) for message in manager.text_model.requests[0]["messages"]
    )


def test_user_given_an_episode_outline_with_one_beat_per_clip_when_clips_are_written_then_each_clip_performs_exactly_its_own_beat():
    """User Given a serial episode outline with a beat for every clip When the clip writer runs Then it is told to give each clip exactly its own beat and invent nothing between them, so the outline is not re-imagined."""
    prompt = _segment_request(serial=True)

    assert "Segment i performs only beat i" in prompt
    assert "do not add a beat the outline does not contain" in prompt.casefold()


def test_user_given_a_serial_episode_when_the_last_clip_is_written_then_it_ends_on_the_open_hook_rather_than_resolving_the_series():
    """User Given a serial episode When the final clip is written Then it shows the small result and then the hook image, and never closes the series question."""
    prompt = _segment_request(serial=True)

    assert "final segment shows the episode's small result and then its hook" in prompt
    assert "ending resolves or reframes the opening" not in prompt


def test_user_given_a_non_serial_story_when_clips_are_written_then_the_existing_resolve_rule_is_unchanged():
    """User Given a story that is not a serial episode When clips are written Then the original one-goal-one-obstacle-one-turn rule still applies."""
    prompt = _segment_request(serial=False)

    assert "ending resolves or reframes the opening" in prompt
    assert "Segment i performs only beat i" not in prompt
