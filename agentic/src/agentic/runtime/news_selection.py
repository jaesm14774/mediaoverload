"""Choose timely, Taiwan-relevant news events using news and PTT evidence."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor, log, sqrt
from typing import Any
from urllib.parse import urlparse

from agentic.runtime.news_grounding import is_brand_safe_news_selection


MAX_EVENT_GAP = timedelta(hours=72)
PTT_FRESHNESS_WINDOW = timedelta(hours=24)
EVENT_SIMILARITY_THRESHOLD = 0.34
PTT_SIMILARITY_THRESHOLD = 0.29
MAX_EVENT_ARTICLES = 40
MAX_DOCUMENT_FREQUENCY = 0.12
MIN_MAX_DOCUMENT_FREQUENCY = 3
TITLE_NGRAM_LENGTHS = (2, 3, 4)

TAIWAN_RELEVANCE_TERMS = (
    "台灣", "臺灣", "台湾", "全台", "全臺", "全國", "北市", "台北", "臺北",
    "新北", "基隆", "桃園", "新竹", "苗栗", "台中", "臺中", "彰化", "南投",
    "雲林", "嘉義", "台南", "臺南", "高雄", "屏東", "宜蘭", "花蓮", "台東",
    "臺東", "澎湖", "金門", "馬祖", "連江", "氣象署", "公路局", "台鐵", "臺鐵",
    "高鐵", "捷運", "交通部", "教育部", "衛福部", "勞動部", "消防署", "警政署",
    "行政院", "經濟部", "國道", "taiwan", "taiwanese", "taipei", "taoyuan",
    "hsinchu", "taichung", "tainan", "kaohsiung", "hualien", "keelung",
)

CONCRETE_ACTION_TERMS = (
    "異動", "延期", "改期", "取消", "停班", "停課", "停駛", "停航", "封路", "管制",
    "加開", "調整", "開放", "上路", "實施", "生效", "啟動", "升級", "解除", "警報",
    "預警", "影響", "報名", "開賣", "開幕", "開展", "開演", "登場", "推出", "公布",
    "公佈", "名單", "結果", "奪冠", "晉級", "奪牌", "漲價", "降價", "調漲", "調降",
    "申請", "受理", "預約", "公測", "下架", "中止", "封閉", "公告", "活動",
    "市集", "嘉年華", "燈會", "展覽", "遊行", "煙火", "慶典", "開跑", "減弱",
    "減輕", "走低", "轉弱", "增強", "回升", "下滑", "破紀錄", "創新高", "刷新",
    "降溫", "升溫", "豪雨", "降雨", "停電", "啟用", "重啟", "完工", "落幕",
)


@dataclass(frozen=True, slots=True)
class NewsCandidateDecision:
    selected: dict[str, Any]
    reason: str
    event_title: str = ""
    evidence: dict[str, int | float | bool] | None = None


def _normalize_text(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum()
    )


def _parse_created_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _keyword_terms(value: Any) -> list[str]:
    terms = re.split(r"[,，;；、|/\s]+", str(value or ""))
    return [term for term in (_normalize_text(item) for item in terms) if len(term) >= 2]


def _document_terms(row: dict[str, Any]) -> Counter[str]:
    terms: Counter[str] = Counter()
    title = _normalize_text(row.get("title"))
    for size in TITLE_NGRAM_LENGTHS:
        terms.update(
            f"t:{title[index:index + size]}"
            for index in range(max(0, len(title) - size + 1))
        )
    for keyword in _keyword_terms(row.get("keyword")):
        terms[f"kw:{keyword}"] += 1
        for size in (2, 3):
            terms.update(
                f"k:{keyword[index:index + size]}"
                for index in range(max(0, len(keyword) - size + 1))
            )
    return terms


def _build_vectors(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], list[float], dict[str, list[tuple[int, float]]]]:
    documents = [_document_terms(row) for row in rows]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())

    max_document_frequency = max(
        MIN_MAX_DOCUMENT_FREQUENCY,
        floor(len(rows) * MAX_DOCUMENT_FREQUENCY),
    )
    vectors: list[dict[str, float]] = []
    norms: list[float] = []
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, document in enumerate(documents):
        vector: dict[str, float] = {}
        for term, frequency in document.items():
            frequency_in_documents = document_frequency[term]
            if frequency_in_documents < 2 or frequency_in_documents > max_document_frequency:
                continue
            inverse_document_frequency = log(
                (len(rows) + 1) / (frequency_in_documents + 1)
            ) + 1
            term_frequency = 1 + log(frequency)
            feature_weight = 0.65 if term.startswith(("k:", "kw:")) else 1.0
            vector[term] = term_frequency * inverse_document_frequency * feature_weight
        norm = sqrt(sum(weight * weight for weight in vector.values()))
        vectors.append(vector)
        norms.append(norm)
        for term, weight in vector.items():
            postings[term].append((index, weight))
    return vectors, norms, postings


def _similarities(
    index: int,
    vectors: list[dict[str, float]],
    norms: list[float],
    postings: dict[str, list[tuple[int, float]]],
    *,
    max_index: int | None = None,
) -> dict[int, float]:
    if norms[index] == 0:
        return {}
    dots: dict[int, float] = defaultdict(float)
    for term, weight in vectors[index].items():
        for neighbor, neighbor_weight in postings[term]:
            if neighbor != index and (max_index is None or neighbor < max_index):
                dots[neighbor] += weight * neighbor_weight
    return {
        neighbor: dot / (norms[index] * norms[neighbor])
        for neighbor, dot in dots.items()
        if norms[neighbor]
    }


def _is_traceable(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "").strip()
    url = str(row.get("url") or row.get("article_url") or "").strip()
    parsed = urlparse(url)
    return bool(source and parsed.scheme in {"http", "https"} and parsed.netloc)


def is_traceable_news(row: dict[str, Any]) -> bool:
    """Return whether a news row has both a publisher and a usable article URL."""
    return _is_traceable(row)


def _editorial_fit(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(field) or "")
        for field in ("title", "keyword", "category")
    ).casefold()
    return (
        any(term in text for term in TAIWAN_RELEVANCE_TERMS)
        and any(term in text for term in CONCRETE_ACTION_TERMS)
        and _is_traceable(row)
    )


def _count_in_window(
    rows: list[dict[str, Any]],
    reference_time: datetime,
    start: timedelta,
    end: timedelta = timedelta(0),
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (created_at := _parse_created_at(row.get("created_at"))) is not None
        and end <= reference_time - created_at < start
    ]


def _make_events(
    news_rows: list[dict[str, Any]],
    vectors: list[dict[str, float]],
    norms: list[float],
    postings: dict[str, list[tuple[int, float]]],
    reference_time: datetime,
) -> list[dict[str, Any]]:
    # The vectorizer receives the news rows first. Keep the same order for its indices.
    ordered_indexes = sorted(
        range(len(news_rows)),
        key=lambda index: _parse_created_at(news_rows[index].get("created_at")) or datetime.min,
        reverse=True,
    )
    claimed: set[int] = set()
    events: list[dict[str, Any]] = []
    max_topic_anchor_frequency = max(3, floor(len(news_rows) * 0.02))
    for seed in ordered_indexes:
        if seed in claimed:
            continue
        seed_time = _parse_created_at(news_rows[seed].get("created_at"))
        if seed_time is None or norms[seed] == 0:
            continue
        similarities = _similarities(
            seed, vectors, norms, postings, max_index=len(news_rows)
        )
        matches: list[tuple[float, int]] = []
        for neighbor, similarity in similarities.items():
            if neighbor >= len(news_rows) or neighbor <= seed or neighbor in claimed:
                continue
            neighbor_time = _parse_created_at(news_rows[neighbor].get("created_at"))
            if (
                neighbor_time is None
                or seed_time - neighbor_time > MAX_EVENT_GAP
            ):
                continue
            shared_topic_anchor = any(
                term.startswith("kw:")
                or (term.startswith("t:") and len(term[2:]) >= 3)
                for term in vectors[seed].keys() & vectors[neighbor].keys()
                if len(postings[term]) <= max_topic_anchor_frequency
            )
            if similarity < EVENT_SIMILARITY_THRESHOLD and not shared_topic_anchor:
                continue
            matches.append((similarity, neighbor))
        matches.sort(reverse=True)
        members = [seed]
        for _similarity, neighbor in matches:
            if neighbor in claimed:
                continue
            members.append(neighbor)
            if len(members) >= MAX_EVENT_ARTICLES:
                break
        claimed.update(members)
        member_rows = [news_rows[index] for index in members]
        current_rows = [
            row for row in member_rows
            if (created_at := _parse_created_at(row.get("created_at"))) is not None
            and timedelta(0) <= reference_time - created_at <= MAX_EVENT_GAP
        ]
        if not current_rows:
            continue
        fit_rows = [row for row in current_rows if _editorial_fit(row)]
        events.append(
            {
                "rows": current_rows,
                "fit_rows": fit_rows,
                "vector_indexes": members[:3],
                "newest_at": seed_time,
                "title": str(news_rows[seed].get("title") or "").strip(),
            }
        )
    return events


def _match_ptt_to_events(
    events: list[dict[str, Any]],
    ptt_rows: list[dict[str, Any]],
    vectors: list[dict[str, float]],
    norms: list[float],
    news_count: int,
) -> None:
    event_norms: list[float] = []
    event_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event_index, event in enumerate(events):
        anchors = [vectors[index] for index in event["vector_indexes"] if norms[index]]
        combined: dict[str, float] = defaultdict(float)
        for vector in anchors:
            for term, weight in vector.items():
                combined[term] += weight / len(anchors)
        norm = sqrt(sum(weight * weight for weight in combined.values()))
        event_norms.append(norm)
        for term, weight in combined.items():
            event_postings[term].append((event_index, weight))

    if not events:
        return
    for ptt_index, row in enumerate(ptt_rows, start=news_count):
        ptt_time = _parse_created_at(row.get("created_at"))
        if ptt_time is None or norms[ptt_index] == 0:
            continue
        dots: dict[int, float] = defaultdict(float)
        for term, weight in vectors[ptt_index].items():
            for event_index, event_weight in event_postings[term]:
                dots[event_index] += weight * event_weight
        ranked = sorted(
            (
                (dot / (norms[ptt_index] * event_norms[event_index]), event_index)
                for event_index, dot in dots.items()
                if event_norms[event_index]
            ),
            reverse=True,
        )
        for similarity, event_index in ranked:
            if similarity < PTT_SIMILARITY_THRESHOLD:
                break
            event = events[event_index]
            if any(
                (created_at := _parse_created_at(news_row.get("created_at"))) is not None
                and abs(ptt_time - created_at) <= MAX_EVENT_GAP
                for news_row in event["rows"]
            ):
                event.setdefault("ptt_rows", []).append(row)
                break


def _event_evidence(
    event: dict[str, Any],
    *,
    reference_time: datetime,
    news_volume: dict[str, int],
    ptt_is_fresh: bool,
) -> dict[str, int | float | bool]:
    rows = event["rows"]
    news_6h = _count_in_window(rows, reference_time, timedelta(hours=6))
    news_24h = _count_in_window(rows, reference_time, timedelta(hours=24))
    news_previous_24h = _count_in_window(
        rows, reference_time, timedelta(hours=48), timedelta(hours=24)
    )
    publishers_6h = {str(row.get("source") or "").strip().casefold() for row in news_6h}
    publishers_24h = {str(row.get("source") or "").strip().casefold() for row in news_24h}
    ptt_rows = event.get("ptt_rows", []) if ptt_is_fresh else []
    ptt_6h = _count_in_window(ptt_rows, reference_time, timedelta(hours=6))
    ptt_24h = _count_in_window(ptt_rows, reference_time, timedelta(hours=24))
    ptt_previous_24h = _count_in_window(
        ptt_rows, reference_time, timedelta(hours=48), timedelta(hours=24)
    )
    ptt_boards = {str(row.get("board") or "").strip().casefold() for row in ptt_24h}
    comments = sum(max(0, int(row.get("comment_count") or 0)) for row in ptt_24h)
    reactions = sum(
        max(0, int(row.get(field) or 0))
        for row in ptt_24h
        for field in ("like_count", "neutral_count", "dislike_count")
    )
    previous_comments = sum(
        max(0, int(row.get("comment_count") or 0)) for row in ptt_previous_24h
    )
    previous_reactions = sum(
        max(0, int(row.get(field) or 0))
        for row in ptt_previous_24h
        for field in ("like_count", "neutral_count", "dislike_count")
    )
    total_24h = max(1, int(news_volume.get("24h", 0)))
    total_previous_24h = max(1, int(news_volume.get("previous_24h", 0)))
    total_6h = max(1, int(news_volume.get("6h", 0)))
    event_share_6h = len(news_6h) / total_6h
    event_share_24h = len(news_24h) / total_24h
    event_share_previous_24h = len(news_previous_24h) / total_previous_24h
    share_lift = log(
        (event_share_24h + 1 / total_24h)
        / (event_share_previous_24h + 1 / total_previous_24h)
    )
    ptt_post_lift = log((len(ptt_24h) + 1) / (len(ptt_previous_24h) + 1))
    ptt_interactions_per_post = (comments + reactions) / max(1, len(ptt_24h))
    ptt_previous_interactions_per_post = (
        (previous_comments + previous_reactions) / max(1, len(ptt_previous_24h))
    )
    ptt_interaction_lift = log(
        (ptt_interactions_per_post + 1) / (ptt_previous_interactions_per_post + 1)
    )
    news_support = (
        len(news_6h) >= 2 and len(publishers_6h) >= 2
    ) or (
        len(news_24h) >= 3 and len(publishers_24h) >= 2
    )
    ptt_support = (
        len(news_24h) >= 1 and len(ptt_24h) >= 3 and len(ptt_boards) >= 2
    )
    support = news_support or ptt_support
    rising = (
        (len(news_6h) >= 2 and len(publishers_6h) >= 2)
        or share_lift >= log(1.25)
        or (ptt_is_fresh and ptt_post_lift >= log(1.5))
        or (len(news_24h) >= 4 and event_share_24h >= 0.005)
    )
    score = (
        1.4 * log(1 + len(news_6h))
        + 1.0 * log(1 + len(news_24h))
        + 1.3 * log(1 + len(publishers_24h))
        + 0.6 * max(-1.0, min(2.0, share_lift))
        + (0.5 * log(1 + len(ptt_6h)) + 0.35 * log(1 + len(ptt_24h))
           + 0.25 * log(1 + len(ptt_boards)) + 0.15 * log(1 + comments + reactions)
           + 0.35 * max(-1.0, min(2.0, ptt_post_lift))
           + 0.2 * max(-1.0, min(2.0, ptt_interaction_lift))
           if ptt_is_fresh else 0.0)
    )
    return {
        "news_6h": len(news_6h),
        "news_24h": len(news_24h),
        "news_previous_24h": len(news_previous_24h),
        "publishers_6h": len(publishers_6h),
        "publishers_24h": len(publishers_24h),
        "news_volume_6h": int(news_volume.get("6h", 0)),
        "news_volume_24h": total_24h,
        "news_volume_previous_24h": total_previous_24h,
        "news_share_6h": round(event_share_6h, 5),
        "news_share_24h": round(event_share_24h, 5),
        "news_share_previous_24h": round(event_share_previous_24h, 5),
        "event_share_lift": round(share_lift, 3),
        "ptt_6h": len(ptt_6h),
        "ptt_24h": len(ptt_24h),
        "ptt_previous_24h": len(ptt_previous_24h),
        "ptt_boards_24h": len(ptt_boards),
        "ptt_comments_24h": comments,
        "ptt_reactions_24h": reactions,
        "ptt_interactions_previous_24h": previous_comments + previous_reactions,
        "ptt_post_lift": round(ptt_post_lift, 3),
        "ptt_interaction_lift": round(ptt_interaction_lift, 3),
        "ptt_fresh": ptt_is_fresh,
        "news_supported": news_support,
        "ptt_supported": ptt_support,
        "supported": support,
        "rising": rising,
        "score": round(score, 3),
    }


def _ptt_is_fresh(ptt_rows: list[dict[str, Any]], reference_time: datetime) -> bool:
    timestamps = [
        ingested_at
        for row in ptt_rows
        if (ingested_at := _parse_created_at(row.get("create_time"))) is not None
    ]
    local_offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    reference_utc = reference_time - local_offset
    current_ingestions = [timestamp for timestamp in timestamps if timestamp <= reference_utc]
    return bool(
        current_ingestions
        and reference_utc - max(current_ingestions) <= PTT_FRESHNESS_WINDOW
    )


def select_news_candidate(
    rows: list[dict[str, Any]],
    *,
    ptt_rows: list[dict[str, Any]] | None = None,
    news_volume: dict[str, int] | None = None,
    rng: random.Random | None = None,
    reference_time: datetime | None = None,
) -> NewsCandidateDecision:
    """Select from evidence-backed event candidates, or explicitly fall back."""
    traceable_rows = [row for row in rows if _is_traceable(row)]
    if not traceable_rows:
        raise ValueError("news candidates need a publisher and a traceable URL")

    chooser = rng or random
    current_time = _parse_created_at(reference_time) if reference_time else None
    current_time = current_time or datetime.now()
    community_rows = list(ptt_rows or [])
    ptt_fresh = _ptt_is_fresh(community_rows, current_time)
    if ptt_fresh:
        community_rows = [
            row for row in community_rows
            if (created_at := _parse_created_at(row.get("created_at"))) is not None
            and timedelta(0) <= current_time - created_at <= timedelta(hours=48)
            and is_brand_safe_news_selection(str(row.get("title") or ""), "")
        ]
    else:
        community_rows = []
    volume = dict(news_volume or {})
    if not volume:
        volume = {
            "6h": len(_count_in_window(traceable_rows, current_time, timedelta(hours=6))),
            "24h": len(_count_in_window(traceable_rows, current_time, timedelta(hours=24))),
            "previous_24h": len(_count_in_window(
                traceable_rows, current_time, timedelta(hours=48), timedelta(hours=24)
            )),
        }

    news_ordered = sorted(
        traceable_rows,
        key=lambda row: _parse_created_at(row.get("created_at")) or datetime.min,
        reverse=True,
    )
    ptt_ordered = sorted(
        community_rows,
        key=lambda row: _parse_created_at(row.get("created_at")) or datetime.min,
        reverse=True,
    )
    all_rows = [*news_ordered, *ptt_ordered]
    vectors, norms, postings = _build_vectors(all_rows)
    events = _make_events(
        news_ordered, vectors, norms, postings, current_time
    )
    _match_ptt_to_events(
        events, ptt_ordered, vectors, norms, len(news_ordered)
    )

    eligible_events: list[tuple[dict[str, Any], dict[str, int | float | bool]]] = []
    for event in events:
        evidence = _event_evidence(
            event,
            reference_time=current_time,
            news_volume=volume,
            ptt_is_fresh=ptt_fresh,
        )
        if event["fit_rows"] and evidence["supported"] and evidence["rising"]:
            eligible_events.append((event, evidence))

    if not eligible_events:
        return NewsCandidateDecision(
            selected=chooser.choice(news_ordered),
            reason="regular_fallback",
            evidence={"eligible_event_count": 0, "ptt_fresh": ptt_fresh},
        )

    event, evidence = chooser.choices(
        eligible_events,
        weights=[max(0.1, float(item[1]["score"])) ** 2 for item in eligible_events],
        k=1,
    )[0]
    recent_fit_rows = [
        row
        for row in event["fit_rows"]
        if _count_in_window([row], current_time, timedelta(hours=24))
    ]
    selected = max(
        recent_fit_rows or event["fit_rows"],
        key=lambda row: (
            sum(term in " ".join(str(row.get(field) or "") for field in ("title", "keyword", "category"))
                for term in TAIWAN_RELEVANCE_TERMS),
            sum(term in " ".join(str(row.get(field) or "") for field in ("title", "keyword", "category"))
                for term in CONCRETE_ACTION_TERMS),
            _parse_created_at(row.get("created_at")) or datetime.min,
        ),
    )
    return NewsCandidateDecision(
        selected=selected,
        reason="hot_event",
        event_title=str(event.get("title") or "").strip(),
        evidence={**evidence, "eligible_event_count": len(eligible_events)},
    )
