
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

