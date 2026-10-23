"""Read-only experiment for finding cross-publisher event clusters in recent news."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agentic" / "src"))

from agentic.runtime.news_grounding import (
    is_brand_safe_news_selection,
    is_usable_news_selection,
)
from agentic.tools.context_services import NewsContextService


LOOKBACK_DAYS = 7
CANDIDATE_LIMIT = 5000
MAX_ARTICLE_GAP_HOURS = 72
NEIGHBOR_COUNT = 30
HOT_PUBLISHER_THRESHOLD = 3
SIMILARITY_THRESHOLDS = (0.20, 0.24, 0.28, 0.32, 0.36)
REVIEW_THRESHOLD = 0.24
TOP_CLUSTER_LIMIT = 25
SAMPLE_HEADLINE_LIMIT = 4


def _load_recent_news() -> tuple[list[dict[str, Any]], int]:
    for env_file in ("media_overload.env", ".env"):
        if os.path.exists(env_file):
            load_dotenv(env_file)
            break

    service = NewsContextService()
    if not service.is_configured():
        raise RuntimeError("Project MySQL settings are not configured")

    import pymysql

    categories = list(service.DEFAULT_EXCLUDE_CATEGORIES)
    date_filter = (
        datetime.now() - timedelta(days=LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d %H:%M:%S")
    placeholders = ", ".join(["%s"] * len(categories))
    query = f"""
        SELECT id, article_id, title, keyword, category, created_at,
               article_url AS url, source
        FROM news_ch.news
        WHERE category NOT IN ({placeholders})
          AND COALESCE(keyword, '') != ''
          AND created_at >= %s
        ORDER BY id DESC
        LIMIT %s
    """
    connection = pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
        read_timeout=25,
        write_timeout=25,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, [*categories, date_filter, CANDIDATE_LIMIT])
            queried_rows = list(cursor.fetchall() or [])
    finally:
        connection.close()

    eligible_rows = [
        row
        for row in queried_rows
        if is_usable_news_selection(row.get("title", ""), row.get("keyword", ""))
        and is_brand_safe_news_selection(
            row.get("title", ""), row.get("keyword", "")
        )
    ]
    return eligible_rows, len(queried_rows)


def _normalize_title(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold(), flags=re.UNICODE)


def _build_title_neighbors(
    rows: list[dict[str, Any]],
) -> tuple[list[datetime | pd.NaT], list[str], Any, Any, Any, int]:
    data = pd.DataFrame(rows)
    times = [pd.to_datetime(value, errors="coerce") for value in data["created_at"]]
    publishers = [
        str(value or "").strip().casefold() for value in data["source"]
    ]
    documents = [_normalize_title(value) for value in data["title"]]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.10,
        sublinear_tf=True,
    )
    text_matrix = vectorizer.fit_transform(documents)
    neighbor_count = min(NEIGHBOR_COUNT + 1, len(rows))
    neighbors = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=neighbor_count,
        n_jobs=-1,
    ).fit(text_matrix)
    distances, indices = neighbors.kneighbors(text_matrix)
    return times, publishers, text_matrix, distances, indices, text_matrix.shape[1]


def _cluster_from_anchor(
    rows: list[dict[str, Any]],
    times: list[datetime | pd.NaT],
    publishers: list[str],
    distances: Any,
    neighbor_indices: Any,
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    claimed: set[int] = set()
    clusters: list[dict[str, Any]] = []
    edge_count = 0
    seed_order = sorted(
        range(len(rows)),
        key=lambda index: (
            pd.Timestamp.min if pd.isna(times[index]) else times[index]
        ),
        reverse=True,
    )

    for seed in seed_order:
        if seed in claimed or not publishers[seed] or pd.isna(times[seed]):
            continue
        members = [seed]
        member_sources = {publishers[seed]}
        member_similarities = {seed: 1.0}
        matches: list[tuple[float, int]] = []
        for distance, neighbor in zip(
            distances[seed], neighbor_indices[seed], strict=True
        ):
            neighbor = int(neighbor)
            if (
                neighbor == seed
                or neighbor in claimed
                or not publishers[neighbor]
                or publishers[neighbor] == publishers[seed]
                or pd.isna(times[neighbor])
            ):
                continue
            if publishers[neighbor] in member_sources:
                continue
            if (
                abs((times[seed] - times[neighbor]).total_seconds())
                > MAX_ARTICLE_GAP_HOURS * 3600
            ):
                continue
            similarity = 1.0 - float(distance)
            if similarity >= threshold:
                matches.append((similarity, neighbor))

        for similarity, neighbor in sorted(matches, reverse=True):
            if publishers[neighbor] in member_sources:
                continue
            members.append(neighbor)
            member_sources.add(publishers[neighbor])
            member_similarities[neighbor] = similarity
            edge_count += 1

        if len(member_sources) < HOT_PUBLISHER_THRESHOLD:
            continue

        claimed.update(members)
        newest_time = times[seed]
        recent_member_indices = {
            hours: [
                index
                for index in members
                if 0
                <= (newest_time - times[index]).total_seconds()
                <= hours * 3600
            ]
            for hours in (6, 24)
        }
        clusters.append(
            {
                "publisher_count": len(member_sources),
                "article_count": len(members),
                "newest_article": str(times[seed]),
                "publishers_last_6h": len(
                    {publishers[index] for index in recent_member_indices[6]}
                ),
                "publishers_last_24h": len(
                    {publishers[index] for index in recent_member_indices[24]}
                ),
                "sample_articles": [
                    {
                        "title": str(rows[index].get("title") or ""),
                        "source": str(rows[index].get("source") or ""),
                        "category": str(rows[index].get("category") or ""),
                        "created_at": str(rows[index].get("created_at") or ""),
                        "url": str(rows[index].get("url") or ""),
                        "similarity_to_anchor": round(
                            member_similarities[index], 3
                        ),
                    }
                    for index in members[:SAMPLE_HEADLINE_LIMIT]
                ],
            }
        )

    clusters.sort(
        key=lambda cluster: (
            cluster["publisher_count"],
            cluster["article_count"],
            cluster["newest_article"],
        ),
        reverse=True,
    )
    return clusters, edge_count


def _profile(rows: list[dict[str, Any]], queried_count: int) -> dict[str, Any]:
    data = pd.DataFrame(rows)
    source_values = data["source"].fillna("").astype(str).str.strip().str.casefold()
    title_values = data["title"].fillna("").astype(str).str.strip().str.casefold()
    times = pd.to_datetime(data["created_at"], errors="coerce")
    return {
        "queried_rows": queried_count,
        "eligible_rows": len(data),
        "distinct_article_ids": int(data["article_id"].nunique(dropna=True)),
        "distinct_publishers": int(source_values[source_values != ""].nunique()),
        "missing_publisher_rows": int((source_values == "").sum()),
        "duplicate_source_title_rows": int(
            data.assign(_source=source_values, _title=title_values).duplicated(
                ["_source", "_title"]
            ).sum()
        ),
        "oldest_created_at": str(times.min()),
        "newest_created_at": str(times.max()),
    }


def run_experiment() -> dict[str, Any]:
    rows, queried_count = _load_recent_news()
    if not rows:
        raise RuntimeError("No eligible news was returned for the current window")

    times, publishers, _matrix, distances, indices, feature_count = (
        _build_title_neighbors(rows)
    )
    threshold_results = []
    review_clusters = []
    for threshold in SIMILARITY_THRESHOLDS:
        clusters, edges = _cluster_from_anchor(
            rows, times, publishers, distances, indices, threshold
        )
        threshold_results.append(
            {
                "cosine_threshold": threshold,
                "candidate_events_3plus_publishers": len(clusters),
                "largest_event_publisher_count": max(
                    (cluster["publisher_count"] for cluster in clusters), default=0
                ),
                "largest_event_article_count": max(
                    (cluster["article_count"] for cluster in clusters), default=0
                ),
                "candidate_links": edges,
            }
        )
        if threshold == REVIEW_THRESHOLD:
            review_clusters = clusters

    return {
        "profile": _profile(rows, queried_count),
        "method": {
            "text_fields": ["title"],
            "similarity": "character 2-4 gram TF-IDF cosine",
            "max_document_frequency": 0.10,
            "nearest_neighbors_per_article": NEIGHBOR_COUNT,
            "max_article_gap_hours": MAX_ARTICLE_GAP_HOURS,
            "hot_publisher_threshold": HOT_PUBLISHER_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "feature_count": feature_count,
            "clustering": "newest-seeded, one representative per publisher, no transitive chaining",
        },
        "threshold_sensitivity": threshold_results,
        "top_event_candidates": review_clusters[:TOP_CLUSTER_LIMIT],
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), ensure_ascii=True, indent=2))
