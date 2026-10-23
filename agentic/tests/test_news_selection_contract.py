"""Public behavior for evidence-backed news event selection."""

import inspect
import random
from datetime import datetime, timedelta

import pytest

from agentic.runtime.news_selection import select_news_candidate
from agentic.tools.context_services import NewsContextService
from scripts.run_longvideo_production_e2e import parse_args


def _news(title, source, created_at, slug, keyword="薔琵颱風"):
    return {
        "title": title,
        "keyword": keyword,
        "source": source,
        "url": f"https://news.example.test/{slug}",
        "created_at": created_at,
    }


def test_user_given_the_normal_news_entry_when_no_pool_limit_is_overridden_then_the_latest_5000_rows_are_requested():
    """User Given normal news-driven callers When they use the public selection default Then its candidate limit is 5000 and its eligibility window remains seven days."""
    parameters = inspect.signature(NewsContextService.select_news).parameters

    assert parameters["limit"].default == 5000
    assert parameters["lookback_days"].default == 7


def test_user_given_the_longvideo_news_entry_when_no_limit_is_supplied_then_it_uses_the_same_candidate_default():
    """User Given the production long-video CLI When its news limit is not overridden Then it requests the same latest 5000 candidates as normal news selection."""
    assert parse_args([]).news_limit == 5000


def test_user_given_a_rising_taiwan_event_with_distinct_action_angles_when_selecting_then_it_returns_a_traceable_hot_story():
    """User Given recent coverage from multiple publishers describes a Taiwan event and concrete actions When selecting news Then an eligible specific angle is returned with its momentum evidence."""
    now = datetime(2026, 10, 22, 18)
    event_rows = [
        _news(
            "薔琵颱風影響台北交通管制，公路局公布道路措施",
            "Outlet A",
            now,
            "storm-road-controls",
            "薔琵颱風;台北交通",
        ),
        _news(
            "天母萬聖節活動因薔琵颱風延期，主辦單位公告新日期",
            "Outlet B",
            now - timedelta(hours=2),
            "halloween-postponed",
            "薔琵颱風;天母萬聖節",
        ),
        _news(
            "薔琵颱風來襲，台北市公布交通與活動異動",
            "Outlet A",
            now - timedelta(hours=4),
            "taipei-event-changes",
            "薔琵颱風;台北交通",
        ),
    ]
    rows = [
        *event_rows,
        _news(
            "週末電影票價優惠活動開跑",
            "Outlet C",
            now - timedelta(hours=1),
            "movie-discount",
            "週末電影",
        ),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "hot_event"
    assert decision.selected in event_rows
    assert decision.selected["url"].startswith("https://")
    assert decision.evidence["news_6h"] >= 2
    assert decision.evidence["publishers_6h"] >= 2
    assert decision.evidence["news_supported"] is True
    assert decision.evidence["rising"] is True


def test_user_given_many_recent_articles_from_one_publisher_when_selecting_then_one_source_cannot_make_an_event_hot():
    """User Given one publisher repeats a Taiwan event across several recent articles When selecting news Then the event falls back because independent coverage is missing."""
    now = datetime(2026, 10, 22, 18)
    repeated_event = [
        _news(
            "薔琵颱風影響台北交通，公路局公布道路措施",
            "Outlet A",
            now,
            "single-source-roads",
        ),
        _news(
            "薔琵颱風造成台北活動延期，主辦單位公告調整",
            "Outlet A",
            now - timedelta(hours=1),
            "single-source-events",
        ),
        _news(
            "薔琵颱風最新消息：台北交通異動與活動措施",
            "Outlet A",
            now - timedelta(hours=2),
            "single-source-update",
        ),
    ]

    decision = select_news_candidate(
        repeated_event, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "regular_fallback"
    assert decision.selected in repeated_event
    assert decision.evidence["eligible_event_count"] == 0


def test_user_given_unrelated_events_with_a_shared_generic_keyword_when_selecting_then_they_stay_out_of_the_hot_pool():
    """User Given unrelated stories share only a generic keyword When selecting news Then they do not become one hot event."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news("台積電高雄廠擴大投資計畫", "Outlet A", now, "tsmc-investment", "今日焦點"),
        _news("中職金手套名單正式公布", "Outlet B", now - timedelta(hours=1), "cpbl-awards", "今日焦點"),
        _news("國旅連假旅客預訂量增加", "Outlet C", now - timedelta(hours=2), "travel-booking", "今日焦點"),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "regular_fallback"
    assert decision.selected in rows
    assert decision.evidence["eligible_event_count"] == 0


def test_user_given_news_without_a_publisher_or_article_url_when_selecting_then_it_is_rejected_as_untraceable():
    """User Given a candidate has no verifiable publisher and article URL When selecting news Then the selector rejects it instead of treating it as evidence."""
    with pytest.raises(ValueError, match="publisher and a traceable URL"):
        select_news_candidate(
            [{"title": "台北活動延期公告", "created_at": datetime(2026, 10, 22, 18)}],
            reference_time=datetime(2026, 10, 22, 18),
        )


def test_user_given_coverage_older_than_the_active_event_window_when_selecting_then_old_articles_do_not_make_it_hot():
    """User Given an event's only coverage is older than the selector's active event window When selecting news Then stale coverage cannot qualify it as hot."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news("薔琵颱風交通與活動異動一覽", "Outlet A", now - timedelta(days=4), "old-a"),
        _news("颱風薔琵影響台北交通活動調整", "Outlet B", now - timedelta(days=5), "old-b"),
        _news("薔琵襲台台北活動與交通最新異動", "Outlet C", now - timedelta(days=6), "old-c"),
        _news("週末電影票價優惠活動開跑", "Outlet D", now, "current-movie", "週末電影"),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "regular_fallback"
    assert decision.evidence["eligible_event_count"] == 0


def test_user_given_a_story_with_no_coverage_in_the_recent_24_hours_when_selecting_then_yesterday_only_coverage_is_not_hot():
    """User Given an event was covered only 25 to 27 hours ago When selecting news Then yesterday's coverage does not satisfy recent momentum."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news("薔琵颱風影響台北交通與活動異動", "Outlet A", now - timedelta(hours=25), "yesterday-a"),
        _news("颱風薔琵台北交通活動調整一覽", "Outlet B", now - timedelta(hours=26), "yesterday-b"),
        _news("薔琵颱風來襲台北交通措施更新", "Outlet C", now - timedelta(hours=27), "yesterday-c"),
        _news("本日股市收盤行情整理", "Outlet D", now, "market-close", "股市收盤"),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "regular_fallback"
    assert decision.evidence["eligible_event_count"] == 0
