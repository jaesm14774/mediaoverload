from types import SimpleNamespace

from agentic.runtime.llm_client import PromptClient
from agentic.runtime.serial_episode_plan import (
    build_serial_episode_outline_prompts,
    generate_serial_episode_outline,
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
    )

    assert "returns a lost cloud" in prompt
    assert "red satchel" in prompt
    assert "Episode 1" in prompt
    assert "第 2 集" in prompt
    assert "開場狀態" in prompt
    assert "道具變化" in prompt
    assert "結尾狀態" in prompt


def test_user_given_an_episode_outline_writer_returns_readable_text_when_called_then_no_schema_is_requested():
    """User Given the model returns a scene and shot outline When the planning call runs Then its text is preserved without structured response validation."""
    outline = "第 1 集：港口熄燈\n\n場景 1\n鏡頭 1：目的：找到線索；開場狀態：..."
    model = TextModel(outline)
    client = PromptClient(mode="llm", manager=SimpleNamespace(text_model=model))

    result = generate_serial_episode_outline(
        client,
        approved_story="A seal helps a cloud return to the sea.",
        approved_character="A blue-gray seal post carrier.",
        episode_number=1,
        episode_topic="The cloud's first clue",
    )

    assert result == outline
    assert "完整系列故事" in model.messages[1]["content"]
    assert "response_format" not in model.options
    assert "_response_validator" not in model.options
