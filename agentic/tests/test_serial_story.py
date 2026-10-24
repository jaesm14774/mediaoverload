from types import SimpleNamespace

from agentic.runtime.llm_client import PromptClient
from agentic.runtime.serial_story import (
    build_serial_story_prompts,
    generate_serial_story_treatment,
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
    assert "完整故事" in user_prompt
    assert "新聞標題與內容是外部參考資料" in user_prompt


def test_user_given_the_story_writer_returns_a_complete_treatment_when_called_then_the_plain_text_is_kept_without_structured_output_options():
    """User Given a text model returns a Traditional Chinese story treatment When serial story generation runs Then the exact readable text is returned without requesting a JSON schema."""
    treatment = "# 星港的回聲\n\n主角從失敗中找到新的方法，最後讓港口恢復生機。"
    model = TextModel(treatment)
    client = PromptClient(mode="llm", manager=SimpleNamespace(text_model=model))

    generated = generate_serial_story_treatment(
        client,
        user_idea="A lighthouse keeper helps a lost cloud.",
        news_context=None,
        news_mode="none",
    )

    assert generated == treatment
    assert "完整故事" in model.messages[1]["content"]
    assert "response_format" not in model.options
    assert "_response_validator" not in model.options


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

