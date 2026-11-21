import pytest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from provider_fakes import ScriptedLLMManager


def _segment_request(*, serial: bool, segment_count: int = 4) -> str:
    segments = [
        {
            "segment_id": f"s{index}",
            "visual": "Kirby hops in the meadow",
            "narration": "",
            "action": "Kirby sprints across the meadow toward the hill and bounces off a stump",
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


def test_user_given_a_serial_episode_when_clips_are_written_then_each_ten_second_clip_gets_three_timed_beats_the_camera_can_see():
    """User Given a serial episode of ten-second clips When the clip writer runs Then each action is told to be three timed beats of visible movement that cover the whole clip, with the main move starting early and a change shown as a concrete picture, with no readable text, notes or phone messages, while a non-serial story is not given that limit."""
    serial = _segment_request(serial=True)
    other = _segment_request(serial=False)

    assert "ten-second clip" in serial
    assert "three timed beats" in serial
    assert "0.00-3.00 s" in serial
    assert "starts by about 4 seconds" in serial
    assert "a basket plainly empty" in serial
    assert "no readable text, notes, posters, signs, phone or screen messages" in serial
    assert "ten-second clip" not in other
    assert "three timed beats" not in other


def test_user_given_a_serial_episode_when_clips_are_written_then_the_writer_may_plan_a_tracking_close_up_or_cut_camera_but_other_stories_keep_the_locked_camera():
    """User Given a serial episode When the clip writer runs Then the camera field may plan a sideways tracking shot, a face close-up or up to three planned cuts and never a push-in, while a non-serial story still gets the locked-camera contract."""
    serial = _segment_request(serial=True).casefold()
    other = _segment_request(serial=False).casefold()

    assert "tracking sideways level with the action" in serial
    assert "close-up on the protagonist's face" in serial
    assert "begin with [shot 2] and [shot 3]" in serial
    assert "only when the camera field plans hard cuts" in serial
    assert "the camera field itself never contains a [shot] label" in serial
    assert "choose the camera from the beat" in serial
    assert "never a push-in, zoom, orbit or crane" in serial
    assert "the camera stays locked" not in serial
    assert "a locked camera and what stays in frame" not in serial
    assert "a locked camera and what stays in frame" in other
    assert "the camera stays locked" in other


def test_user_given_a_serial_episode_when_clips_are_written_then_every_clip_including_the_first_starts_moving_in_the_first_second_and_never_rests():
    """User Given a serial episode When the clip writer runs Then every clip, the opening one too, is told to start big continuous movement in its first second and never to use sleeping, resting or posing as the action, so no clip renders near-static."""
    prompt = _segment_request(serial=True).casefold()

    assert "including the first" in prompt
    assert "starting in the first second" in prompt
    assert "sleeping, resting" in prompt


def _written_actions(first_action: str, second_action: str, *, serial: bool) -> list[str]:
    def reply(action: str) -> dict:
        return {
            "segments": [
                {
                    "segment_id": "s0",
                    "visual": "Kirby in the meadow",
                    "narration": "",
                    "action": action,
                    "cause": "wind",
                    "effect": "a leaf spins",
                    "camera": "locked",
                }
            ]
        }

    manager = ScriptedLLMManager([reply(first_action), reply(second_action)])
    goal = GoalRequest(
        prompt="One beat.",
        media_type="long_video",
        duration_seconds=10,
        style="watercolor",
        constraints={"character": "Kirby", **({"serial_mode": True} if serial else {})},
    )
    segments = LLMPromptEngine(mode="llm", manager=manager).segment_story(goal, "brief", 1, "playful")
    return [segment["action"] for segment in segments]


CALM = "Kirby presses against the glass, sways gently, and stands ready while the backpack dangles."
LIVELY = "Kirby sprints across the plaza, leaps the bench, and bounces off the lamp post toward the gate."


def test_user_given_a_serial_clip_whose_planned_action_is_only_resting_when_clips_are_written_then_the_writer_is_made_to_write_a_moving_action():
    """User Given a serial beat the writer first plans as pressing, swaying and standing ready When clips are written Then that plan is refused and the moving action the writer gives next is the one used, so no clip is planned to stand still."""
    assert _written_actions(CALM, LIVELY, serial=True) == [LIVELY]


IN_PLACE = "Kirby bounces lightly on his heels, hops from side to side, and spins on the spot while blinking."
HOP_WITH_GOAL = "Kirby hops toward the gate, bounces off the lamp post, and spins up onto the bench."


def test_user_given_a_serial_clip_planned_as_hopping_on_the_spot_when_clips_are_written_then_it_is_refused_unless_the_hops_go_somewhere():
    """User Given a serial beat planned as bouncing, hopping and spinning in place When clips are written Then it is refused and the writer's next plan is used, while the same kind of hops given a destination are kept."""
    assert _written_actions(IN_PLACE, LIVELY, serial=True) == [LIVELY]
    assert _written_actions(HOP_WITH_GOAL, CALM, serial=True) == [HOP_WITH_GOAL]


def test_user_given_a_serial_clip_with_a_moving_action_when_clips_are_written_then_it_is_kept_as_written():
    """User Given a serial beat planned with running, leaping and bouncing When clips are written Then it is kept."""
    assert _written_actions(LIVELY, CALM, serial=True) == [LIVELY]


def test_user_given_a_story_that_is_not_serial_when_clips_are_written_then_a_calm_action_is_not_refused():
    """User Given a non-serial story with a calm action When clips are written Then it is kept, because the moving-action rule belongs to the serial strategy only."""
    assert _written_actions(CALM, LIVELY, serial=False) == [CALM]


def test_user_given_a_non_serial_story_when_clips_are_written_then_the_existing_resolve_rule_is_unchanged():
    """User Given a story that is not a serial episode When clips are written Then the original one-goal-one-obstacle-one-turn rule still applies."""
    prompt = _segment_request(serial=False)

    assert "ending resolves or reframes the opening" in prompt
    assert "Segment i performs only beat i" not in prompt


def test_user_given_a_serial_episode_when_clips_are_written_then_the_writer_is_not_also_told_the_camera_must_stay_locked_and_never_cut():
    """User Given a serial episode whose camera plan may track or cut When the clip writer runs Then its system message no longer says locked camera, no push-in, pan, orbit or cut, while other stories keep that default."""
    def system_message(serial: bool) -> str:
        goal = GoalRequest(
            prompt="One beat.",
            media_type="long_video",
            duration_seconds=10,
            style="watercolor",
            constraints={"character": "Kirby", **({"serial_mode": True} if serial else {})},
        )
        manager = ScriptedLLMManager([{"segments": [{"segment_id": "s0", "visual": "v", "narration": "", "action": LIVELY, "cause": "c", "effect": "e", "camera": "tracking"}]}])
        LLMPromptEngine(mode="llm", manager=manager).segment_story(goal, "brief", 1, "playful")
        return str(manager.text_model.requests[0]["messages"][0]["content"]).casefold()

    assert "no push-in, pan, orbit, or cut" not in system_message(serial=True)
    assert "no push-in, pan, orbit, or cut" in system_message(serial=False)


def test_user_given_a_serial_episode_when_the_writer_never_gives_a_moving_plan_then_the_episode_stops_instead_of_filming_a_clip_with_no_planned_action():
    """User Given a serial episode whose writer only ever plans resting actions When the clips are written Then writing the plan fails loudly, instead of silently falling back to a clip with no planned action, camera or ending."""
    calm_reply = {
        "segments": [
            {"segment_id": "s0", "visual": "v", "narration": "", "action": CALM, "cause": "c", "effect": "e", "camera": "locked"}
        ]
    }
    goal = GoalRequest(
        prompt="One beat.",
        media_type="long_video",
        duration_seconds=10,
        style="watercolor",
        constraints={"character": "Kirby", "serial_mode": True},
    )
    engine = LLMPromptEngine(mode="llm", manager=ScriptedLLMManager([calm_reply], repeat_last=True))

    with pytest.raises(ValueError, match="continuous movement"):
        engine.segment_story(goal, "brief", 1, "playful")


def test_user_given_a_serial_episode_with_a_recurring_cast_when_clips_are_written_then_every_beat_must_keep_the_whole_cast_in_the_action():
    """User Given a serial episode whose cast is listed When the clip writer runs Then each beat's action must name every recurring character and a tracking shot must carry the whole cast along, so the last frame that the next clip continues from still holds them, while other stories are not given that rule."""
    serial = _segment_request(serial=True).casefold()
    other = _segment_request(serial=False).casefold()

    assert "name every recurring character listed above in every beat" in serial
    assert "a tracking shot carries the whole cast" in serial
    assert "name every recurring character listed above" not in other


def _cast_goal() -> GoalRequest:
    return GoalRequest(
        prompt="四個節拍：開場、問題、升級、解決與鉤子。",
        media_type="long_video",
        duration_seconds=40,
        style="watercolor",
        constraints={
            "character": "Kirby",
            "serial_mode": True,
            "serial_cast": "- Momo：a yellow striped squirrel\n- 嘟嘟：a yellow duckling with a blue bow",
        },
    )


def _beat(action: str) -> dict:
    return {
        "segment_id": "s0", "visual": "Kirby runs", "narration": "", "action": action,
        "cause": "wind", "effect": "leaf spins", "camera": "tracking",
    }


def test_user_given_a_cast_when_a_planned_beat_leaves_a_recurring_character_out_then_the_beat_is_rewritten_until_the_whole_cast_is_in_it():
    """User Given a serial episode with Momo and 嘟嘟 in the cast, listed as the series ledger writes them (name, full-width colon, look) When the planner writes a beat that never names 嘟嘟 Then the plan is sent back for repair and the accepted beat names both, so the clip's last frame still holds the whole cast."""
    without_dede = _beat("Kirby sprints across the meadow past Momo and bounces off a stump")
    with_cast = _beat("Kirby sprints across the meadow past Momo and 嘟嘟 and bounces off a stump")
    manager = ScriptedLLMManager([{"segments": [without_dede]}, {"segments": [with_cast]}])

    segments = LLMPromptEngine(mode="llm", manager=manager).segment_story(_cast_goal(), "brief", 1, "playful")

    assert all(name in segments[0]["action"] for name in ("Momo", "嘟嘟"))


def test_user_given_no_language_model_when_a_serial_episode_is_planned_then_planning_stops_instead_of_filming_a_template_story():
    """User Given the language model is unavailable When a serial episode's clips are planned Then planning stops with an error and no template story is handed to the renderer."""
    with pytest.raises(Exception):
        LLMPromptEngine(mode="template").segment_story(_cast_goal(), "brief", 1, "playful")


def test_user_given_a_serial_plan_with_too_few_beats_when_it_is_written_then_it_is_sent_back_until_every_clip_has_its_own_beat():
    """User Given an episode of two clips When the planner first writes only one beat Then the plan is sent back and the accepted plan gives both clips an action, so no clip is rendered from just an opening look."""
    one = _beat("Kirby sprints across the meadow and bounces off a stump")
    two = _beat("Kirby leaps over a log and rolls into the pond")
    manager = ScriptedLLMManager([{"segments": [one]}, {"segments": [one, two]}])
    goal = GoalRequest(
        prompt="四個節拍：開場、問題、升級、解決與鉤子。", media_type="long_video", duration_seconds=20,
        style="watercolor", constraints={"character": "Kirby", "serial_mode": True},
    )

    segments = LLMPromptEngine(mode="llm", manager=manager).segment_story(goal, "brief", 2, "playful")

    assert len(segments) == 2 and all(segment["action"] for segment in segments)
