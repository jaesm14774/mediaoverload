"""Public behavior for evidence-backed news event selection."""

import inspect
import os
import random
from datetime import datetime, timedelta

import pytest

from agentic.runtime.news_selection import (
    select_news_candidate,
    select_random_news_inspiration,
)
from agentic.tools.context_services import NewsContextService


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


def test_user_given_repeated_event_terms_across_publishers_when_selecting_then_it_returns_supported_hot_evidence():
    """User Given traceable publishers repeat concrete terms while covering one rising event When selecting news Then the event remains hot and its evidence still shows independent coverage."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news(
            "薔琵颱風來襲，台北交通管制加強，薔琵颱風道路措施更新",
            "Outlet A",
            now,
            "repeated-storm-roads",
            "薔琵颱風;台北交通;道路措施",
        ),
        _news(
            "薔琵颱風影響台北交通，公路局更新薔琵颱風道路管制措施",
            "Outlet B",
            now - timedelta(hours=2),
            "repeated-storm-update",
            "薔琵颱風;台北交通;道路管制",
        ),
        _news(
            "台北因薔琵颱風宣布交通與活動異動，薔琵颱風期間措施公布",
            "Outlet C",
            now - timedelta(hours=4),
            "repeated-storm-changes",
            "薔琵颱風;台北交通;活動異動",
        ),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "hot_event"
    assert decision.selected in rows
    assert decision.evidence["news_supported"] is True
    assert decision.evidence["publishers_6h"] >= 2


def test_user_given_case_and_punctuation_variants_of_one_taiwan_event_when_selecting_then_they_share_hot_evidence():
    """User Given traceable Taiwanese coverage varies Latin casing and punctuation When selecting news Then the related reports still form one independently supported hot event."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news(
            "台北 Typhoon Kirogi：國道因風雨管制，公路局公告替代道路",
            "Outlet A",
            now,
            "normalized-road-controls",
            "Kirogi;Taipei;road-control",
        ),
        _news(
            "花蓮 typhoon-kirogi／Taipei：消防署開放收容中心",
            "Outlet B",
            now - timedelta(hours=2),
            "normalized-shelter",
            "KIROGI;TAIPEI;shelter",
        ),
        _news(
            "台東 TYPHOON KIROGI—Taipei 港區封閉，海巡通知漁船返港",
            "Outlet C",
            now - timedelta(hours=4),
            "normalized-harbor",
            "kirogi;taipei;harbor",
        ),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "hot_event"
    assert decision.selected in rows
    assert decision.evidence["news_supported"] is True
    assert decision.evidence["publishers_6h"] >= 2


def test_user_given_distinct_typhoon_angles_share_a_rare_event_phrase_when_selecting_then_the_event_stays_hot():
    """User Given local reports cover distinct typhoon actions and share only a rare event phrase When selecting news Then the story remains hot with independent publisher evidence."""
    now = datetime(2026, 10, 22, 18)
    rows = [
        _news(
            "薔琵颱風來襲，台北交通管制加強，公路局公布封閉路段與市區替代道路措施",
            "Outlet A",
            now,
            "anchor-roads",
            "台北交通;道路管制",
        ),
        _news(
            "薔琵颱風降雨增強，花蓮居民疏散，消防局開放收容中心與山區收容中心",
            "Outlet B",
            now - timedelta(hours=2),
            "anchor-shelter",
            "花蓮山區;居民疏散",
        ),
        _news(
            "薔琵颱風沿海風浪升高，海巡署通知漁船返港並封港，港區漁船返港措施",
            "Outlet C",
            now - timedelta(hours=4),
            "anchor-coast",
            "台東沿海;漁船返港",
        ),
        _news(
            "台北市區替代道路措施公告，交通局調整路線與管制時段",
            "Outlet D",
            now - timedelta(hours=1),
            "decoy-road",
            "台北替代道路;交通調整",
        ),
        _news(
            "花蓮山區居民疏散與收容中心開放，縣府追加生活物資",
            "Outlet E",
            now - timedelta(hours=3),
            "decoy-shelter",
            "花蓮疏散;收容中心",
        ),
        _news(
            "港區漁船返港措施更新，海巡署維持封港並發布航行警告",
            "Outlet F",
            now - timedelta(hours=5),
            "decoy-coast",
            "漁船返港;港區封閉",
        ),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "hot_event"
    assert decision.selected in rows
    assert decision.evidence["news_supported"] is True
    assert decision.evidence["publishers_6h"] >= 2


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


def test_user_given_random_story_inspiration_when_selecting_then_it_chooses_uniformly_from_only_safe_traceable_articles():
    """User Given safe and unsafe recent article rows When choosing a random story seed Then the injected chooser receives only usable, brand-safe, traceable articles and the decision records its pool size."""
    allowed = _news("社區修復老橋讓居民重新通行", "Outlet A", "2026-10-24 09:00:00", "bridge")
    rejected = _news("政治人物宣布競選政策", "Outlet B", "2026-10-24 08:00:00", "politics")

    class ChooseLast:
        def __init__(self):
            self.seen = []

        def choice(self, values):
            self.seen = list(values)
            return values[-1]

    chooser = ChooseLast()
    decision = select_random_news_inspiration(
        [allowed, rejected, {"title": "untraceable", "keyword": "x"}],
        rng=chooser,
    )

    assert decision.reason == "random_inspiration"
    assert decision.selected is allowed
    assert chooser.seen == [allowed]
    assert decision.evidence["eligible_article_count"] == 1


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


def test_user_given_a_recent_hot_event_and_stale_news_tail_when_selecting_then_only_recent_event_evidence_is_returned():
    """User Given a supported recent event plus unrelated articles older than the event window When selecting news Then the selected story and complete evidence remain those of the recent event."""
    now = datetime(2026, 10, 22, 18)
    recent_rows = [
        _news(
            "薔琵颱風影響台北交通管制，公路局公布道路措施",
            "Outlet A",
            now,
            "storm-roads-a",
        ),
        _news(
            "薔琵颱風來襲，台北市公布交通與活動異動",
            "Outlet B",
            now - timedelta(hours=2),
            "storm-roads-b",
        ),
        _news(
            "台北薔琵颱風道路管制措施調整公告",
            "Outlet C",
            now - timedelta(hours=4),
            "storm-roads-c",
        ),
    ]
    stale_rows = [
        _news(
            "歐洲足球決賽延賽與球員傷勢消息整理",
            "Outlet D",
            now - timedelta(days=4),
            "stale-football",
            "歐洲足球",
        ),
        _news(
            "火星探測器傳回宇宙照片與任務資料更新",
            "Outlet E",
            now - timedelta(days=5),
            "stale-space",
            "火星探測",
        ),
        _news(
            "北海道觀光列車新增秋季班次公告",
            "Outlet F",
            now - timedelta(days=6),
            "stale-train",
            "北海道觀光",
        ),
    ]

    recent_decision = select_news_candidate(
        recent_rows, rng=random.Random(0), reference_time=now
    )
    decision_with_stale_tail = select_news_candidate(
        [*recent_rows, *stale_rows], rng=random.Random(0), reference_time=now
    )

    assert recent_decision.reason == "hot_event"
    assert decision_with_stale_tail.reason == recent_decision.reason
    assert decision_with_stale_tail.selected is recent_decision.selected
    assert decision_with_stale_tail.event_title == recent_decision.event_title
    assert decision_with_stale_tail.evidence == recent_decision.evidence


def test_user_given_fresh_multi_board_ptt_coverage_when_selecting_then_community_evidence_can_support_a_hot_event():
    """User Given one traceable Taiwan event story and fresh matching PTT coverage from multiple boards When selecting news Then community volume can support the hot event without independent multi-publisher coverage."""
    now = datetime(2026, 10, 22, 18)
    local_offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    ingested_at = now - local_offset - timedelta(hours=1)
    news = _news(
        "薔琵颱風影響台北交通，公路局公布道路措施",
        "Outlet A",
        now,
        "storm-road-controls",
        "薔琵颱風;台北交通",
    )
    ptt_rows = [
        {
            "title": title,
            "board": board,
            "created_at": now - timedelta(hours=1),
            "create_time": ingested_at,
            "comment_count": 40,
            "like_count": 12,
            "neutral_count": 2,
            "dislike_count": 1,
        }
        for title, board in (
            ("薔琵颱風台北交通管制討論與道路最新措施", "Gossiping"),
            ("台北薔琵颱風公路局交通公告調整", "Lifeismoney"),
            ("颱風薔琵路況更新，台北公路局公布影響", "Gossiping"),
        )
    ]

    decision = select_news_candidate(
        [news],
        ptt_rows=ptt_rows,
        news_volume={"6h": 10, "24h": 20, "previous_24h": 20},
        rng=random.Random(0),
        reference_time=now,
    )

    assert decision.reason == "hot_event"
    assert decision.selected is news
    assert decision.evidence["ptt_fresh"] is True
    assert decision.evidence["ptt_24h"] >= 3
    assert decision.evidence["ptt_boards_24h"] >= 2
    assert decision.evidence["ptt_supported"] is True
    assert decision.evidence["rising"] is True


def test_user_given_a_configured_news_store_when_selecting_then_selected_article_content_matches_the_stored_article():
    """User Given a configured MySQL news store with traceable articles When the public service selects an article Then its returned content matches that selected article's stored content."""
    service = NewsContextService()
    if not service.is_configured():
        pytest.skip("MySQL news credentials are not configured.")

    selection = service.select_news()
    if selection is None:
        pytest.skip("The configured news store has no eligible article rows.")

    import pymysql

    connection = pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT content
                FROM news_ch.news
                WHERE TRIM(article_url) = %s
                  AND TRIM(title) = %s
                  AND TRIM(COALESCE(keyword, '')) = %s
                  AND TRIM(COALESCE(category, '')) = %s
                  AND created_at = %s
                  AND TRIM(COALESCE(source, '')) = %s
                """,
                [
                    selection.url,
                    selection.title,
                    selection.keyword,
                    selection.category,
                    selection.created_at,
                    selection.source,
                ],
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    assert rows
    assert selection.content in {str(row.get("content") or "") for row in rows}


def test_user_given_related_articles_with_stale_and_invalid_timestamps_when_selecting_then_only_fresh_valid_coverage_counts():
    """User Given related reports include fresh, stale, and malformed timestamps When selecting a hot story Then only valid fresh coverage contributes to momentum evidence."""
    now = datetime(2026, 10, 22, 18)
    fresh_rows = [
        _news("薔琵颱風台北交通管制措施最新公告", "Outlet A", now.isoformat(sep=" "), "fresh-a"),
        _news("薔琵颱風台北交通管制措施更新公告", "Outlet B", (now - timedelta(hours=2)).isoformat(sep=" "), "fresh-b"),
        _news("薔琵颱風台北交通管制措施調整公告", "Outlet C", (now - timedelta(hours=4)).isoformat(sep=" "), "fresh-c"),
    ]
    rows = [
        *fresh_rows,
        _news("薔琵颱風台北交通管制措施補充", "Outlet D", "not-a-date", "invalid"),
        _news("薔琵颱風台北交通管制措施回顧", "Outlet E", (now - timedelta(hours=73)).isoformat(sep=" "), "stale"),
        _news("週末電影票價優惠活動開跑", "Outlet F", now.isoformat(sep=" "), "movie", "週末電影"),
    ]

    decision = select_news_candidate(
        rows, rng=random.Random(0), reference_time=now
    )

    assert decision.reason == "hot_event"
    assert decision.selected in fresh_rows
    assert decision.evidence["news_6h"] == len(fresh_rows)
    assert decision.evidence["news_supported"] is True
