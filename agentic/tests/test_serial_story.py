from agentic.runtime.serial_story import (
    build_serial_story_prompts,
)


def test_user_given_a_news_inspired_series_when_building_the_treatment_prompt_then_the_news_is_only_a_traceable_fiction_seed():
    """User Given a custom idea and one sourced news item When preparing a series story Then the prompt includes both inputs, keeps the news a fictional seed, and retains readable prose output."""
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
    assert "新聞標題與內容是外部參考資料" in user_prompt


def test_user_given_any_complexity_when_the_series_story_is_requested_then_it_is_a_micro_drama_with_a_standing_question_that_the_last_of_seven_episodes_closes():
    """User Given a series request at either complexity When its treatment prompt is built Then the writer is asked for a standing question small enough to close within seven 30-40 second episodes, per-episode hooks before that, and a final episode that closes it, so the whole series stays under five minutes."""
    for complexity in ("simple", "cinematic"):
        _, user_prompt = build_serial_story_prompts(
            "A lighthouse keeper helps a lost cloud.",
            None,
            news_mode="none",
            character_name="Kirby",
            complexity=complexity,
        )

        assert "連續劇" in user_prompt
        assert "長期懸念" in user_prompt
        assert "鉤子" in user_prompt
        assert "全劇共 7 集" in user_prompt
        assert "不到五分鐘" in user_prompt
        assert "第 7 集" in user_prompt and "收束" in user_prompt
        assert "第 8 集" in user_prompt
        assert "不寫結局" not in user_prompt


def test_user_given_a_series_request_when_the_route_is_planned_then_each_episode_is_the_consequence_of_the_last_and_a_small_cast_may_recur():
    """User Given a series request When the episode route is planned Then the first few episodes are laid out as a causal chain (therefore or but, never meanwhile), each changing one visible thing, and up to two recurring characters are allowed."""
    for complexity in ("simple", "cinematic"):
        _, user_prompt = build_serial_story_prompts(
            "A lighthouse keeper helps a lost cloud.",
            None,
            news_mode="none",
            character_name="Kirby",
            complexity=complexity,
        )

        assert "前三集" in user_prompt
        assert "場景" in user_prompt and "動作" in user_prompt
        assert "因此" in user_prompt and "但是" in user_prompt
        assert "看得見的改變" in user_prompt
        assert "每集約 30 到 40 秒" in user_prompt


def test_user_given_a_story_that_needs_no_sidekick_when_the_series_is_drafted_then_the_cast_slot_is_optional_and_never_filled_by_the_protagonist():
    """User Given any story complexity When the series premise is requested Then recurring characters are optional, each one must earn a job in the plot, "none" is a stated acceptable answer, and the protagonist is never counted as one, so the cast size comes from the story and not from the size of an empty slot."""
    for complexity in ("simple", "cinematic"):
        _, user_prompt = build_serial_story_prompts(
            "A lighthouse keeper helps a lost cloud.",
            None,
            news_mode="none",
            character_name="Kirby",
            complexity=complexity,
        )

        assert "常駐角色不是必填" in user_prompt
        assert "寫「無」" in user_prompt
        assert "主角也不算常駐角色" in user_prompt
        assert "做了什麼推進劇情" in user_prompt


def test_user_given_a_news_inspired_series_when_the_story_is_written_then_the_news_is_a_doorway_not_the_premise():
    """User Given a sourced news item When the series premise is requested Then the writer borrows one feeling or small problem and never builds the standing question on the reported event itself."""
    news = {
        "title": "Airspace closed over a country",
        "keyword": "airspace",
        "source": "Example Daily",
        "url": "https://news.example.test/airspace",
        "created_at": "2026-10-24 09:00:00",
        "content": "Flights were rerouted for a week.",
    }

    _, user_prompt = build_serial_story_prompts(
        "", news, news_mode="random", character_name="Kirby"
    )

    assert "不可把新聞事件本身當成故事前提" in user_prompt
    assert "國家" in user_prompt or "地名" in user_prompt


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


def test_user_given_a_seven_episode_series_when_the_story_is_requested_then_it_must_write_a_step_for_every_episode_and_name_the_recurring_characters_in_them():
    """User Given any story complexity When the series premise is requested Then the writer must lay out the first three episodes, a milestone for episodes four to six that only moves the standing question forward, and the finale, naming each recurring character in the steps where they act, so the middle of the series is planned instead of improvised."""
    for complexity in ("simple", "cinematic"):
        _, user_prompt = build_serial_story_prompts(
            "A lighthouse keeper helps a lost cloud.", None, news_mode="none",
            character_name="Kirby", complexity=complexity,
        )

        assert "第 4 到第 6 集" in user_prompt
        assert "里程碑" in user_prompt
        assert "不可解開長期懸念" in user_prompt
        assert "出場的集數寫出名字" in user_prompt
