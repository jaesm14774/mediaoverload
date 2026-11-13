
from agentic.runtime.serial_episode_plan import (
    build_serial_episode_outline_prompts,
)


class TextModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []
        self.options = {}

    def chat_completion(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return self.response


def test_user_given_an_approved_story_and_character_when_planning_an_episode_then_the_outline_covers_arc_scenes_and_shot_states():
    """User Given the approved series story, character and episode direction When planning an episode Then the prompt requests causal episode beats followed by readable scenes and shots with opening, action, prop and ending states."""
    _, prompt = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a brave, absent-minded blue-gray seal with a red satchel.",
        episode_number=2,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide discovers the lantern is being blocked by salt crystals.",
        previous_episode_summaries=["Episode 1: Tide finds the cloud and learns the harbor has gone dark."],
        complexity="cinematic",
    )

    assert "returns a lost cloud" in prompt
    assert "red satchel" in prompt
    assert "Episode 1" in prompt
    assert "第 2 集" in prompt
    assert "開場狀態" in prompt
    assert "道具變化" in prompt
    assert "結尾狀態" in prompt



def test_user_given_a_cinematic_episode_when_planning_shots_then_the_outline_keeps_the_camera_steady():
    """User Given a cinematic episode When its shot list is requested Then the prompt keeps the camera fixed with at most one slow push-in, so motion comes from the character rather than the camera."""
    _, prompt = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a blue-gray seal.",
        episode_number=1,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide finds the blocked lantern.",
        previous_episode_summaries=[],
        complexity="cinematic",
    )

    assert "鏡頭固定" in prompt
    assert "最多一次緩慢推近" in prompt


def _simple_outline_prompt(**overrides) -> str:
    arguments = dict(
        approved_story="Kirby wonders where the star candy comes from.",
        approved_character="Kirby is a round pink creature with red feet.",
        episode_number=3,
        episode_topic="The humming cave",
        episode_direction="Kirby follows a hum into a cave.",
        previous_episode_summaries=[],
        complexity="simple",
    )
    arguments.update(overrides)
    _, prompt = build_serial_episode_outline_prompts(**arguments)
    return prompt


def test_user_given_a_one_minute_episode_when_planning_then_there_is_one_beat_for_every_clip_and_the_last_one_is_a_hook():
    """User Given an episode that renders as four clips When the outline is planned Then it asks for exactly one visible beat per clip and ends on an unanswered hook instead of a final resolution."""
    prompt = _simple_outline_prompt(beat_count=4)

    assert "四個節拍" in prompt
    assert "每個節拍對應一支影片" in prompt
    assert "鉤子" in prompt
    assert "不得解開長期懸念" in prompt


def test_user_given_earlier_episodes_when_planning_the_next_one_then_their_actions_are_listed_as_already_shown_and_not_to_be_repeated():
    """User Given episodes that already aired When the next outline is planned Then the prompt lists what each already did and forbids doing the same errand or setting again, so a new episode cannot replay an old one."""
    prompt = _simple_outline_prompt(
        earlier_episode_actions=[
            "第 1 集《信》：Kirby 把一封信送到朋友家門口。",
            "第 2 集《港口》：Kirby 在碼頭追一頂被風吹走的帽子。",
        ]
    )

    assert "已經演過" in prompt
    assert "把一封信送到朋友家門口" in prompt
    assert "追一頂被風吹走的帽子" in prompt
    assert "不得重複" in prompt


def test_user_given_a_series_route_when_planning_one_episode_then_only_that_episodes_part_is_performed():
    """User Given the whole route in the approved story When episode 3 is planned Then the prompt restricts the writer to episode 3's part and treats later episodes as future, so episode 1 can never act out the story's later events."""
    prompt = _simple_outline_prompt(episode_number=3)

    assert "只演第 3 集" in prompt
    assert "之後的集數是未來" in prompt


def test_user_given_a_cinematic_episode_when_planning_then_the_series_rules_still_apply():
    """User Given a cinematic episode When it is planned Then the hook, no-repeat and own-episode-only rules apply there too, not only to simple episodes."""
    prompt = _simple_outline_prompt(
        complexity="cinematic",
        earlier_episode_actions=["第 1 集《信》：Kirby 送信。"],
    )

    assert "鉤子" in prompt
    assert "已經演過" in prompt
    assert "只演第 3 集" in prompt


def test_user_given_an_approved_outline_when_it_is_summarised_for_later_episodes_then_only_its_one_line_story_is_kept():
    """User Given an approved outline with beats and a state card When later episodes need to know what it did Then the digest carries the episode number, topic and one-line story, not the whole outline."""
    from agentic.runtime.serial_episode_plan import episode_action_digest

    outline = (
        "一、本集一句話與系列位置\nKirby 把一封信送到朋友家門口，系列從此開始。\n\n"
        "二、四個節拍\n- 開始：背起背包\n- 問題：信掉進河裡\n"
    )

    digest = episode_action_digest(1, "信", outline)

    assert "第 1 集" in digest and "信" in digest
    assert "把一封信送到朋友家門口" in digest
    assert "信掉進河裡" not in digest
