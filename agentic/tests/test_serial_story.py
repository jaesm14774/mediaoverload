
from agentic.runtime.serial_story import (
    build_serial_story_prompts,
)


class TextModel:
    """Provider-contract fake for the existing chat-completion interface."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []
        self.options = {}

    def chat_completion(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return self.response


def test_user_given_a_news_inspired_series_when_building_the_treatment_prompt_then_the_news_is_only_a_traceable_fiction_seed():
    """User Given a custom idea and one sourced news item When preparing a series story Then the prompt includes both inputs, requires a complete fictional arc, and retains readable prose output."""
    news = {
        "title": "A community repairs a damaged footbridge",
        "keyword": "community;bridge",
        "source": "Example Daily",
        "url": "https://news.example.test/bridge",
        "created_at": "2026-10-24 09:00:00",
        "content": "Residents worked together to repair the bridge.",
    }

    system_prompt, user_prompt = build_serial_story_prompts(
        "A tiny lighthouse keeper helps a lost cloud find the sea.",
        news,
        news_mode="random",
    )

    assert "A tiny lighthouse keeper helps a lost cloud find the sea." in user_prompt
    assert "https://news.example.test/bridge" in user_prompt
    assert "虛構" in (system_prompt + user_prompt)
    assert "完整系列故事" in user_prompt
    assert "新聞標題與內容是外部參考資料" in user_prompt


def test_user_given_a_simple_serial_story_request_when_treatment_prompt_is_built_then_the_series_has_natural_length_and_cross_episode_continuity():
    """User Given a simple serial-story request When its treatment prompt is built Then it asks for a complete naturally sized series, separates total episodes from the current production batch, and carries continuity across episodes."""
    _, user_prompt = build_serial_story_prompts(
        "A lighthouse keeper helps a lost cloud.",
        None,
        news_mode="none",
        character_name="Kirby",
        complexity="simple",
    )

    assert "完整開端、發展和真正收尾" in user_prompt
    assert "不設三集上限" in user_prompt
    assert "每次執行製作幾集是兩回事" in user_prompt
    assert "跨集保持一致" in user_prompt
    assert "最後一集必須收束主線" in user_prompt


def test_user_given_any_story_complexity_when_building_the_system_prompt_then_the_story_must_be_a_visible_physical_animation_not_a_procedural_drama():
    """User Given a cinematic or simple request When the system prompt is built Then both tell the writer to keep the plot visual and out of legal, tax or paperwork drama."""
    for complexity in ("simple", "cinematic"):
        system_prompt, _ = build_serial_story_prompts(
            "A lighthouse keeper helps a lost cloud.",
            None,
            news_mode="none",
            complexity=complexity,
        )

        assert "動感" in system_prompt
        assert "法律" in system_prompt and "稅務" in system_prompt


def test_user_given_random_news_inspiration_has_no_eligible_source_when_building_a_story_prompt_then_it_fails_without_improvising():
    """User Given random-news inspiration is enabled but no source was selected When building the story prompt Then generation stops with an actionable error instead of silently switching to improvisation."""
    try:
        build_serial_story_prompts(
            "A little robot repairs a garden.",
            None,
            news_mode="random",
        )
    except ValueError as exc:
        assert "news" in str(exc).casefold()
        assert "none" in str(exc).casefold()
    else:
        raise AssertionError("Missing news inspiration must block the default news mode.")

