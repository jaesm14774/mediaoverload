
from agentic.runtime.serial_character import (
    build_serial_character_prompts,
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


def test_user_given_character_preferences_when_drafting_a_card_then_the_preferences_and_approved_story_shape_the_readable_character_proposal():
    """User Given an approved story and character preferences When drafting a character card Then the prompt uses both, describes continuity anchors, and requests readable prose."""
    system_prompt, user_prompt = build_serial_character_prompts(
        approved_story="The keeper helps a lost cloud return to the sea.",
        user_preferences="主角是年輕的海豹郵差，勇敢但有點迷糊。",
    )

    assert "The keeper helps a lost cloud return to the sea." in user_prompt
    assert "年輕的海豹郵差" in user_prompt
    assert "外觀" in user_prompt
    assert "跨集固定" in system_prompt
    assert "外觀錨點" in system_prompt


def test_user_given_no_character_preferences_when_drafting_a_card_then_the_model_is_asked_to_invent_a_story_fitting_character():
    """User Given an approved story and no character preferences When drafting a character card Then the model invents an original protagonist that serves the story."""
    _, user_prompt = build_serial_character_prompts(
        approved_story="A clockmaker restores the town's lost seasons.",
        user_preferences="",
    )

    assert "沒有提供角色偏好" in user_prompt
    assert "自行創作" in user_prompt
    assert "外觀" in user_prompt


def test_user_given_a_simple_card_when_drafting_then_the_character_carries_a_standing_want_that_outlives_any_single_episode():
    """User Given a simple-mode character card When it is drafted Then the card asks for a standing want that later episodes can keep pursuing, not a plan for one episode's action."""
    _, user_prompt = build_serial_character_prompts(
        approved_story="The keeper helps a lost cloud return to the sea.",
        user_preferences="",
        complexity="simple",
    )

    assert "一直想要" in user_prompt
    assert "這集會怎麼行動" not in user_prompt


def test_user_given_an_approved_story_and_protagonist_when_recurring_characters_are_proposed_then_the_prompt_caps_them_at_two_and_forbids_the_protagonists_name():
    """User Given an approved story and protagonist card When recurring characters are proposed Then the prompt shows both, asks for at most two, and tells the writer not to reuse the protagonist."""
    from agentic.runtime.serial_character import build_serial_companion_proposal_prompts

    _, user_prompt = build_serial_companion_proposal_prompts(
        approved_story="Kirby wonders where the star candy comes from. 常駐角色：小明，戴草帽的青年。",
        approved_character="Kirby is a round pink creature.",
        protagonist_name="Kirby",
    )

    assert "小明" in user_prompt
    assert "round pink creature" in user_prompt
    assert "最多兩位" in user_prompt
    assert "不可以是 Kirby" in user_prompt


def test_user_given_a_story_that_names_its_recurring_characters_when_they_are_proposed_then_the_names_are_kept_exactly_as_the_story_writes_them():
    """User Given a story that names its recurring characters When they are proposed Then the prompt demands the story's own names, untranslated and unrewritten, so the cast the reviewer approved is the cast shown in the video."""
    from agentic.runtime.serial_character import build_serial_companion_proposal_prompts

    _, user_prompt = build_serial_companion_proposal_prompts(
        approved_story="常駐角色：班迪·瓦多魯迪，圓滾滾的藍色小人。",
        approved_character="Kirby is a round pink creature.",
        protagonist_name="Kirby",
    )

    assert "名字必須與故事裡寫的完全一致" in user_prompt
    assert "不翻譯" in user_prompt


def test_user_given_a_companion_proposal_when_it_breaks_a_rule_then_it_is_refused_and_a_valid_one_is_normalised():
    """User Given proposals from the writer When one has too many characters, the protagonist's name, a repeated name or no look Then it is refused, while zero, one or two valid characters are accepted."""
    import pytest

    from agentic.runtime.serial_character import validate_serial_companion_proposal

    valid = {"companions": [{"name": " 小明 ", "appearance": "戴草帽的青年"}]}
    assert validate_serial_companion_proposal(valid, protagonist_name="Kirby") == [
        {"name": "小明", "appearance": "戴草帽的青年"}
    ]
    assert validate_serial_companion_proposal({"companions": []}, protagonist_name="Kirby") == []
    too_many = {"companions": [{"name": name, "appearance": "外觀"} for name in ("甲", "乙", "丙")]}
    repeated = {"companions": [{"name": "甲", "appearance": "外觀"}, {"name": "甲", "appearance": "外觀"}]}
    protagonist = {"companions": [{"name": "kirby", "appearance": "外觀"}]}
    no_look = {"companions": [{"name": "甲", "appearance": " "}]}
    for payload in (too_many, repeated, protagonist, no_look):
        with pytest.raises(ValueError):
            validate_serial_companion_proposal(payload, protagonist_name="Kirby")


def test_user_given_a_recurring_character_when_its_identity_image_prompt_is_built_then_it_shows_only_that_character_in_the_series_style():
    """User Given a recurring character and the series style When its identity image prompt is built Then it asks for one centred character in that style and never mentions the protagonist."""
    from agentic.runtime.serial_character import build_serial_companion_reference_prompt

    prompt = build_serial_companion_reference_prompt(
        name="小明", appearance="戴草帽的青年", style_direction="蠟筆手繪"
    )

    assert "小明" in prompt and "戴草帽的青年" in prompt and "蠟筆手繪" in prompt
    assert "只畫一個角色" in prompt
    assert "Kirby" not in prompt
