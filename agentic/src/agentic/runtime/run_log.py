"""MySQL run log: one row per run, its Discord review stages and its publications.

This is the operator-facing ledger ("what ran, was it accepted, where did it
go"). It is separate from `RunRecorder` (per-run JSON files for debugging) and
from `SerialVideoStore` (the serial series ledger, which keeps its own reviews
and publications).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

# Review decisions recorded per stage (`run_reviews.decision`).
DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"
DECISION_REGENERATE = "regenerate"
# No human verdict: the stage ended without a usable human answer.
NON_VERDICT_DECISIONS = frozenset({"timed_out", "invalid_selection", "unavailable"})
DECISION_AUTOMATIC = "automatic"

# Run level verdicts (`runs.review_outcome`).
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_PARTIAL = "partial"
OUTCOME_REGENERATED = "regenerated"
OUTCOME_UNREVIEWED = "unreviewed"

STATUS_RUNNING = "running"
STATUS_FAILED = "failed"


def utc_now() -> datetime:
    """Naive UTC, which is what the DATETIME(6) columns store."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def derive_review_outcome(decisions: Iterable[str]) -> str:
    """Collapse the ordered review decisions of one run into its verdict.

    Only human decisions count. A stage that timed out, had an unusable
    selection or could not start still means "not accepted": after an approved
    stage it makes the run `partial`, on its own it leaves the run `unreviewed`.
    """
    seen = set(decisions)
    if DECISION_REGENERATE in seen:
        return OUTCOME_REGENERATED
    approved = DECISION_APPROVED in seen
    declined = bool(seen & ({DECISION_REJECTED} | NON_VERDICT_DECISIONS))
    if approved and declined:
        return OUTCOME_PARTIAL
    if approved:
        return OUTCOME_ACCEPTED
    if DECISION_REJECTED in seen:
        return OUTCOME_REJECTED
    return OUTCOME_UNREVIEWED


@dataclass(frozen=True, slots=True)
class ReviewEntry:
    ordinal: int
    plan: str
    stage: str
    node_id: str
    decision: str
    reviewer: str = ""
    review_session_id: str = ""
    selected_count: int = 0
    candidate_count: int = 0


@dataclass(frozen=True, slots=True)
class PublicationEntry:
    platform: str
    outcome: str
    external_id: str = ""
    receipt_status: str = ""
    visibility: str = ""
    verified: bool | None = None
    error: str = ""
    caption: str = ""
    media_paths: tuple[str, ...] = ()
    receipt_details: dict[str, Any] = field(default_factory=dict)
    # When the platform dispatch finished, whatever its outcome.
    dispatched_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RunEntry:
    run_id: str
    status: str
    strategy: str | None = None
    media_type: str | None = None
    workflow_name: str | None = None
    character_name: str | None = None
    prompt: str | None = None
    news_title: str | None = None
    news_url: str | None = None
    # None means the verdict lives elsewhere (serial runs keep it in their own ledger).
    review_outcome: str | None = OUTCOME_UNREVIEWED
    dry_run: bool = False
    failure_reason: str | None = None
    # Final media on disk, project-relative so Windows and Docker record the same value.
    output_paths: tuple[str, ...] = ()
    reviews: Sequence[ReviewEntry] = ()
    publications: Sequence[PublicationEntry] = ()
    finished_at: datetime | None = None


_SCHEMA: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR(32) NOT NULL PRIMARY KEY,
        strategy VARCHAR(64) NULL,
        media_type VARCHAR(64) NULL,
        workflow_name VARCHAR(128) NULL,
        character_name VARCHAR(255) NULL,
        prompt TEXT NULL,
        news_title TEXT NULL,
        news_url TEXT NULL,
        status VARCHAR(32) NOT NULL,
        review_outcome VARCHAR(16) NULL,
        dry_run TINYINT(1) NOT NULL DEFAULT 0,
        failure_reason TEXT NULL,
        output_paths JSON NOT NULL DEFAULT (JSON_ARRAY()),
        started_at DATETIME(6) NOT NULL,
        finished_at DATETIME(6) NULL,
        INDEX ix_runs_started_at (started_at),
        INDEX ix_runs_strategy_outcome (strategy, review_outcome)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE IF NOT EXISTS run_reviews (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(32) NOT NULL,
        ordinal INT NOT NULL,
        plan VARCHAR(32) NOT NULL,
        stage VARCHAR(128) NOT NULL,
        node_id VARCHAR(128) NOT NULL,
        decision VARCHAR(24) NOT NULL,
        reviewer VARCHAR(255) NOT NULL DEFAULT '',
        review_session_id VARCHAR(64) NOT NULL DEFAULT '',
        selected_count INT NOT NULL DEFAULT 0,
        candidate_count INT NOT NULL DEFAULT 0,
        UNIQUE KEY uq_run_reviews_order (run_id, ordinal),
        CONSTRAINT fk_run_reviews_run FOREIGN KEY (run_id) REFERENCES runs(run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE IF NOT EXISTS run_publications (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(32) NOT NULL,
        platform VARCHAR(64) NOT NULL,
        outcome VARCHAR(24) NOT NULL,
        external_id TEXT NOT NULL,
        receipt_status VARCHAR(128) NOT NULL DEFAULT '',
        visibility VARCHAR(64) NOT NULL DEFAULT '',
        verified TINYINT(1) NULL,
        error TEXT NULL,
        caption TEXT NULL,
        media_paths JSON NOT NULL,
        receipt_details JSON NOT NULL,
        dispatched_at DATETIME(6) NULL,
        UNIQUE KEY uq_run_publications_platform (run_id, platform),
        INDEX ix_run_publications_dispatched_at (dispatched_at),
        CONSTRAINT fk_run_publications_run FOREIGN KEY (run_id) REFERENCES runs(run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
)


class RunLog:
    """Writes the run log. Opening it creates the schema, so a bad config fails at once."""

    DEFAULT_DATABASE = "mediaoverload_runs"

    def __init__(
        self,
        database_name: str | None = None,
        *,
        mysql_options: dict[str, Any] | None = None,
    ) -> None:
        self.database_name = str(
            database_name or os.getenv("run_log_mysql_database") or self.DEFAULT_DATABASE
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.database_name):
            raise ValueError("Run log MySQL database names may contain only letters, digits, and underscores.")
        configured = mysql_options or {
            "host": os.getenv("mysql_host", ""),
            "port": os.getenv("mysql_port", "3306"),
            "user": os.getenv("run_log_mysql_user", ""),
            "password": os.getenv("run_log_mysql_password", ""),
        }
        if not all(str(configured.get(key) or "").strip() for key in ("host", "user", "password")):
            raise ValueError(
                "The run log requires mysql_host, run_log_mysql_user, run_log_mysql_password, "
                "and run_log_mysql_database in media_overload.env."
            )
        self.mysql_options = {
            "host": str(configured["host"]),
            "port": int(configured.get("port") or 3306),
            "user": str(configured["user"]),
            "password": str(configured["password"]),
            "database": self.database_name,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
            "autocommit": False,
            "connect_timeout": 5,
            "read_timeout": 30,
            "write_timeout": 30,
        }
        with self._transaction() as cursor:
            for statement in _SCHEMA:
                cursor.execute(statement)

    @classmethod
    def from_environment(cls) -> RunLog | None:
        """For entry points: enabled by `run_log_mysql_database`; without it runs are not logged."""
        if not os.getenv("run_log_mysql_database", "").strip():
            return None
        return cls()

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = pymysql.connect(**self.mysql_options)
        try:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin(self, run_id: str, *, prompt: str, dry_run: bool) -> None:
        """Insert the in-flight row, so killed or orphaned runs stay visible as `running`."""
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO runs (run_id, prompt, dry_run, status, started_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (run_id, prompt, int(dry_run), STATUS_RUNNING, utc_now()),
            )

    def finish(self, entry: RunEntry) -> None:
        """Complete the run row and store its reviews and publications atomically."""
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE runs SET strategy=%s, media_type=%s, workflow_name=%s, character_name=%s, "
                "prompt=COALESCE(%s, prompt), news_title=%s, news_url=%s, status=%s, "
                "review_outcome=%s, dry_run=%s, failure_reason=%s, output_paths=%s, "
                "finished_at=%s WHERE run_id=%s",
                (
                    entry.strategy,
                    entry.media_type,
                    entry.workflow_name,
                    entry.character_name,
                    entry.prompt,
                    entry.news_title,
                    entry.news_url,
                    entry.status,
                    entry.review_outcome,
                    int(entry.dry_run),
                    entry.failure_reason,
                    json.dumps(list(entry.output_paths), ensure_ascii=False),
                    entry.finished_at or utc_now(),
                    entry.run_id,
                ),
            )
            for review in entry.reviews:
                cursor.execute(
                    "INSERT INTO run_reviews (run_id, ordinal, plan, stage, node_id, decision, "
                    "reviewer, review_session_id, selected_count, candidate_count) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        entry.run_id,
                        review.ordinal,
                        review.plan,
                        review.stage,
                        review.node_id,
                        review.decision,
                        review.reviewer,
                        review.review_session_id,
                        review.selected_count,
                        review.candidate_count,
                    ),
                )
            for publication in entry.publications:
                cursor.execute(
                    "INSERT INTO run_publications (run_id, platform, outcome, external_id, "
                    "receipt_status, visibility, verified, error, caption, media_paths, "
                    "receipt_details, dispatched_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        entry.run_id,
                        publication.platform,
                        publication.outcome,
                        publication.external_id,
                        publication.receipt_status,
                        publication.visibility,
                        None if publication.verified is None else int(publication.verified),
                        publication.error or None,
                        publication.caption,
                        json.dumps(list(publication.media_paths), ensure_ascii=False),
                        json.dumps(publication.receipt_details, ensure_ascii=False, default=str),
                        publication.dispatched_at,
                    ),
                )

    def fail(self, run_id: str, *, reason: str) -> None:
        """Close a run that raised before producing a result; a run already finished stays as written."""
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE runs SET status=%s, review_outcome=%s, failure_reason=%s, finished_at=%s "
                "WHERE run_id=%s AND status=%s",
                (STATUS_FAILED, OUTCOME_UNREVIEWED, reason, utc_now(), run_id, STATUS_RUNNING),
            )
