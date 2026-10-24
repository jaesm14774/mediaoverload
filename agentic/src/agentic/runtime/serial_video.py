"""Persistent state and fail-closed lineage for explicit serial video runs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator, Sequence
import uuid

import pymysql
from pymysql.cursors import DictCursor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _existing_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Serial reference or media is not a file: {path}")
    return path


class _QueryResult:
    def __init__(self, rows: list[dict[str, Any]], rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _MySQLSession:
    """Small DB-API session that keeps queries parameterized and returns dictionaries."""

    def __init__(self, connection: pymysql.connections.Connection) -> None:
        self._connection = connection

    def execute(self, statement: str, parameters: Sequence[Any] = ()) -> _QueryResult:
        with self._connection.cursor() as cursor:
            cursor.execute(statement.replace("?", "%s"), tuple(parameters))
            rows = list(cursor.fetchall()) if cursor.description else []
            return _QueryResult(rows, int(cursor.rowcount))


class SerialVideoStore:
    """MySQL ledger for series, episode versions, renders, reviews, and publications."""

    IDENTITY_REFERENCE_LIMIT = 3
    DEFAULT_DATABASE = "mediaoverload_serial"
    _SCHEMA = (
        """CREATE TABLE IF NOT EXISTS series (
            series_id VARCHAR(191) PRIMARY KEY,
            character_name VARCHAR(191) NOT NULL,
            lifecycle_status VARCHAR(16) NOT NULL DEFAULT 'active',
            lifecycle_note TEXT NOT NULL DEFAULT (''),
            lifecycle_updated_at VARCHAR(40) NOT NULL DEFAULT (''),
            story_state_json LONGTEXT NOT NULL DEFAULT ('{}'),
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS episodes (
            episode_id CHAR(32) PRIMARY KEY,
            series_id VARCHAR(191) NOT NULL,
            episode_number INT NOT NULL,
            version INT NOT NULL,
            is_current TINYINT NOT NULL DEFAULT 1,
            prompt LONGTEXT NOT NULL,
            episode_topic VARCHAR(512) NOT NULL,
            target_duration_seconds INT NOT NULL,
            status VARCHAR(32) NOT NULL,
            previous_episode_id CHAR(32) NULL,
            story_context_json LONGTEXT NOT NULL DEFAULT ('{}'),
            story_json LONGTEXT NOT NULL DEFAULT ('[]'),
            plan_sha256 VARCHAR(64) NOT NULL DEFAULT '',
            final_video_path TEXT NOT NULL DEFAULT (''),
            final_video_sha256 VARCHAR(64) NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT (''),
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            UNIQUE KEY uq_episode_version(series_id, episode_number, version),
            KEY idx_episodes_series_number(series_id, episode_number, is_current),
            CONSTRAINT fk_episode_series FOREIGN KEY(series_id) REFERENCES series(series_id),
            CONSTRAINT fk_episode_previous FOREIGN KEY(previous_episode_id) REFERENCES episodes(episode_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS episode_plans (
            episode_id CHAR(32) PRIMARY KEY,
            plan_sha256 CHAR(64) NOT NULL,
            plan_json LONGTEXT NOT NULL,
            created_at VARCHAR(40) NOT NULL,
            CONSTRAINT fk_plan_episode FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS segments (
            segment_id CHAR(32) PRIMARY KEY,
            episode_id CHAR(32) NOT NULL,
            ordinal INT NOT NULL,
            target_duration_seconds DECIMAL(8,3) NOT NULL,
            story_json LONGTEXT NOT NULL DEFAULT ('{}'),
            status VARCHAR(32) NOT NULL,
            current_render_id CHAR(32) NULL,
            approved_render_id CHAR(32) NULL,
            tail_asset_id CHAR(32) NULL,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            UNIQUE KEY uq_segment_ordinal(episode_id, ordinal),
            KEY idx_segments_episode_ordinal(episode_id, ordinal),
            CONSTRAINT fk_segment_episode FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS render_attempts (
            render_id CHAR(32) PRIMARY KEY,
            segment_id CHAR(32) NOT NULL,
            attempt_number INT NOT NULL,
            status VARCHAR(32) NOT NULL,
            workflow_name VARCHAR(191) NOT NULL,
            h3_mode VARCHAR(32) NOT NULL,
            conditioning_json LONGTEXT NOT NULL,
            input_sha256 CHAR(64) NOT NULL,
            generated_output_path TEXT NOT NULL DEFAULT (''),
            generated_output_sha256 VARCHAR(64) NOT NULL DEFAULT '',
            output_path TEXT NOT NULL DEFAULT (''),
            output_sha256 VARCHAR(64) NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT (''),
            started_at VARCHAR(40) NOT NULL,
            finished_at VARCHAR(40) NOT NULL DEFAULT (''),
            UNIQUE KEY uq_render_attempt(segment_id, attempt_number),
            KEY idx_render_segment_status(segment_id, status),
            CONSTRAINT fk_render_segment FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS reviews (
            review_id CHAR(32) PRIMARY KEY,
            segment_id CHAR(32) NOT NULL,
            render_id CHAR(32) NOT NULL UNIQUE,
            video_sha256 CHAR(64) NOT NULL,
            technical_status VARCHAR(32) NOT NULL,
            technical_json LONGTEXT NOT NULL,
            human_decision VARCHAR(32) NOT NULL DEFAULT 'pending',
            human_json LONGTEXT NOT NULL DEFAULT ('{}'),
            stale TINYINT NOT NULL DEFAULT 0,
            created_at VARCHAR(40) NOT NULL,
            updated_at VARCHAR(40) NOT NULL,
            CONSTRAINT fk_review_segment FOREIGN KEY(segment_id) REFERENCES segments(segment_id),
            CONSTRAINT fk_review_render FOREIGN KEY(render_id) REFERENCES render_attempts(render_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS reference_assets (
            asset_id CHAR(32) PRIMARY KEY,
            series_id VARCHAR(191) NOT NULL,
            episode_id CHAR(32) NULL,
            segment_number INT NOT NULL DEFAULT 0,
            role VARCHAR(32) NOT NULL,
            source_render_id CHAR(32) NULL,
            path TEXT NOT NULL,
            sha256 CHAR(64) NOT NULL,
            status VARCHAR(32) NOT NULL,
            created_at VARCHAR(40) NOT NULL,
            KEY idx_reference_assets_episode_role(episode_id, role, status),
            CONSTRAINT fk_reference_series FOREIGN KEY(series_id) REFERENCES series(series_id),
            CONSTRAINT fk_reference_episode FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
            CONSTRAINT fk_reference_render FOREIGN KEY(source_render_id) REFERENCES render_attempts(render_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
        """CREATE TABLE IF NOT EXISTS episode_publications (
            publication_id CHAR(32) PRIMARY KEY,
            episode_id CHAR(32) NOT NULL,
            attempt_number INT NOT NULL,
            platform VARCHAR(64) NOT NULL,
            episode_topic VARCHAR(512) NOT NULL,
            caption LONGTEXT NOT NULL,
            hashtags TEXT NOT NULL,
            media_path TEXT NOT NULL,
            media_sha256 CHAR(64) NOT NULL,
            status VARCHAR(32) NOT NULL,
            review_decision VARCHAR(32) NOT NULL DEFAULT 'pending',
            review_session_id VARCHAR(191) NOT NULL DEFAULT (''),
            receipt_json LONGTEXT NOT NULL DEFAULT ('{}'),
            error TEXT NOT NULL DEFAULT (''),
            started_at VARCHAR(40) NOT NULL,
            finished_at VARCHAR(40) NOT NULL DEFAULT (''),
            UNIQUE KEY uq_publication_attempt(episode_id, platform, attempt_number),
            KEY idx_publication_episode_status(episode_id, status),
            CONSTRAINT fk_publication_episode FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    )

    def __init__(
        self,
        database_name: str | None = None,
        *,
        mysql_options: dict[str, Any] | None = None,
    ) -> None:
        self.database_name = str(
            database_name or os.getenv("serial_video_mysql_database") or self.DEFAULT_DATABASE
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.database_name):
            raise ValueError("Serial video MySQL database names may contain only letters, digits, and underscores.")
        configured = mysql_options or {
            "host": os.getenv("mysql_host", ""),
            "port": os.getenv("mysql_port", "3306"),
            "user": os.getenv("serial_video_mysql_user", ""),
            "password": os.getenv("serial_video_mysql_password", ""),
        }
        if not all(str(configured.get(key) or "").strip() for key in ("host", "user", "password")):
            raise ValueError(
                "Serial video requires mysql_host, serial_video_mysql_user, "
                "serial_video_mysql_password, and serial_video_mysql_database in media_overload.env."
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
        self._initialize_schema()

    def _connect(self) -> pymysql.connections.Connection:
        return pymysql.connect(**self.mysql_options)

    def _initialize_schema(self) -> None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                for statement in self._SCHEMA:
                    cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[_MySQLSession]:
        connection = self._connect()
        try:
            connection.begin()
            yield _MySQLSession(connection)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _identity_files(paths: Sequence[str | Path]) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        for value in paths:
            path = _existing_file(value)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                raise ValueError(f"Character identity reference must be an image: {path}")
            files.append({"path": str(path), "sha256": _sha256(path)})
        if len(files) > SerialVideoStore.IDENTITY_REFERENCE_LIMIT:
            raise ValueError(
                f"A series can use at most {SerialVideoStore.IDENTITY_REFERENCE_LIMIT} fixed character images; "
                "Ref2VA also needs room for the prior episode tail."
            )
        if len({item["path"] for item in files}) != len(files):
            raise ValueError("A character identity image can only be registered once per series.")
        return sorted(files, key=lambda item: item["path"])

    @staticmethod
    def _row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def _active_identity_files(
        self, connection: _MySQLSession, series_id: str
    ) -> list[dict[str, str]]:
        rows = connection.execute(
            """SELECT path, sha256 FROM reference_assets
               WHERE series_id = ? AND episode_id IS NULL
                 AND role = 'character_identity' AND status = 'active'
               ORDER BY path""",
            (series_id,),
        ).fetchall()
        return [{"path": row["path"], "sha256": row["sha256"]} for row in rows]

    @staticmethod
    def choose_auto_next_mode(roll: float, *, has_continuation_candidates: bool) -> str:
        """Use the lower half of the random interval for continuation when one is eligible."""
        value = float(roll)
        if not 0 <= value < 1:
            raise ValueError("The automatic series choice roll must be in [0, 1).")
        return "continue" if has_continuation_candidates and value < 0.5 else "new"

    def list_auto_continuation_candidates(
        self, character: str | None = None
    ) -> list[dict[str, Any]]:
        query = """SELECT s.series_id, s.character_name, s.lifecycle_status,
                          e.episode_id AS latest_episode_id,
                          e.episode_number AS latest_episode_number
                   FROM series s JOIN episodes e ON e.series_id = s.series_id AND e.is_current = 1
                   WHERE s.lifecycle_status = 'active'
                     AND e.episode_number = (
                         SELECT MAX(e2.episode_number) FROM episodes e2
                         WHERE e2.series_id = s.series_id AND e2.is_current = 1
                     )
                     AND e.status = 'ready'"""
        parameters: tuple[Any, ...] = ()
        normalized_character = str(character or "").strip()
        if normalized_character:
            query += " AND s.character_name = ?"
            parameters = (normalized_character,)
        query += " ORDER BY s.series_id"
        with self._connection() as connection:
            return connection.execute(query, parameters).fetchall()

    def list_auto_resume_candidates(
        self, character: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the newest current blocked episode that has enough persisted state to retry."""
        query = """SELECT s.series_id, s.character_name, s.lifecycle_status,
                          e.episode_id AS blocked_episode_id,
                          e.episode_number AS blocked_episode_number,
                          e.prompt AS episode_prompt,
                          e.episode_topic,
                          e.target_duration_seconds,
                          e.updated_at
                   FROM series s JOIN episodes e ON e.series_id = s.series_id AND e.is_current = 1
                   WHERE s.lifecycle_status IN ('active', 'completed')
                     AND e.status = 'blocked'
                     AND e.episode_number = (
                         SELECT MAX(e2.episode_number) FROM episodes e2
                         WHERE e2.series_id = s.series_id AND e2.is_current = 1
                     )
                     AND e.plan_sha256 IS NOT NULL
                     AND EXISTS (
                         SELECT 1 FROM episode_plans p
                         WHERE p.episode_id = e.episode_id AND p.plan_sha256 = e.plan_sha256
                     )
                     AND EXISTS (SELECT 1 FROM segments seg WHERE seg.episode_id = e.episode_id)"""
        parameters: tuple[Any, ...] = ()
        normalized_character = str(character or "").strip()
        if normalized_character:
            query += " AND s.character_name = ?"
            parameters = (normalized_character,)
        query += " ORDER BY e.updated_at DESC, e.created_at DESC, e.episode_id"
        with self._connection() as connection:
            return connection.execute(query, parameters).fetchall()

    def prepare_auto_next_context(
        self,
        character: str,
        identity_reference_paths: Sequence[str | Path] = (),
        *,
        random_source: Any | None = None,
    ) -> dict[str, Any]:
        """Resume the latest failed episode first, otherwise choose a new or ready series at 50/50."""
        normalized_character = str(character or "").strip()
        if not normalized_character:
            raise ValueError("Automatic serial generation requires a fixed character.")
        source = random_source or random.SystemRandom()
        retry_candidates = self.list_auto_resume_candidates(normalized_character)
        if retry_candidates:
            selected = retry_candidates[0]
            context = self.prepare_series_context(
                str(selected["series_id"]),
                normalized_character,
                episode_number=int(selected["blocked_episode_number"]),
            )
            context["automation_mode"] = "resume"
            context["automation_source_series_id"] = str(selected["series_id"])
            context["automation_source_episode_id"] = str(selected["blocked_episode_id"])
            context["resume_episode_id"] = str(selected["blocked_episode_id"])
            context["episode_prompt"] = str(selected["episode_prompt"] or "")
            context["episode_topic"] = str(selected["episode_topic"] or "")
            context["target_duration_seconds"] = int(selected["target_duration_seconds"])
            return context

        candidates = self.list_auto_continuation_candidates(normalized_character)
        mode = self.choose_auto_next_mode(
            source.random(), has_continuation_candidates=bool(candidates)
        )
        if mode == "continue":
            selected = source.choice(candidates)
            context = self.prepare_series_context(
                str(selected["series_id"]),
                normalized_character,
                episode_number=int(selected["latest_episode_number"]) + 1,
            )
            context["automation_mode"] = "continue"
            context["automation_source_series_id"] = str(selected["series_id"])
            context["automation_source_episode_id"] = str(selected["latest_episode_id"])
            return context

        identity_references = list(identity_reference_paths)
        templates: list[list[dict[str, str]]] = []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT series_id FROM series WHERE character_name = ? ORDER BY series_id",
                (normalized_character,),
            ).fetchall()
            for row in rows:
                references = self._active_identity_files(connection, str(row["series_id"]))
                if references:
                    templates.append(references)
        if not identity_references and templates:
            identity_references = [item["path"] for item in source.choice(templates)]
        if not identity_references:
            raise ValueError(
                "Starting a new automatic series requires a registered character identity image."
            )
        slug = re.sub(r"[^a-z0-9]+", "-", normalized_character.lower()).strip("-") or "series"
        series_id = f"auto-{slug}-{uuid.uuid4().hex[:12]}"
        context = self.prepare_series_context(
            series_id, normalized_character, identity_references
        )
        context["automation_mode"] = "new"
        context["automation_source_series_id"] = ""
        context["automation_source_episode_id"] = ""
        return context

    def set_series_lifecycle_status(
        self, series_id: str, status: str, *, note: str = ""
    ) -> None:
        """Close an active series as completed or discarded; terminal series cannot re-enter auto selection."""
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"completed", "discarded"}:
            raise ValueError("A serial lifecycle outcome must be completed or discarded.")
        now = _now()
        with self._connection(write=True) as connection:
            result = connection.execute(
                """UPDATE series SET lifecycle_status = ?, lifecycle_note = ?,
                       lifecycle_updated_at = ?, updated_at = ?
                   WHERE series_id = ? AND lifecycle_status = 'active'""",
                (normalized_status, str(note or "").strip(), now, now, str(series_id).strip()),
            )
            if result.rowcount != 1:
                raise ValueError("Only an active existing series can be completed or discarded.")

    def _validate_asset(self, path: str, expected_sha: str) -> Path:
        resolved = _existing_file(path)
        actual_sha = _sha256(resolved)
        if actual_sha != expected_sha:
            raise ValueError(f"Serial asset changed after registration or review: {resolved}")
        return resolved

    def prepare_series_context(
        self,
        series_id: str,
        character: str,
        identity_reference_paths: Sequence[str | Path] = (),
        episode_number: int | None = None,
    ) -> dict[str, Any]:
        """Resolve read-only episode context and current reference assets."""
        normalized_series_id = str(series_id or "").strip()
        normalized_character = str(character or "").strip()
        if not normalized_series_id:
            raise ValueError("serial mode requires a non-empty series ID")
        if not normalized_character:
            raise ValueError("serial mode requires an explicit fixed character")
        supplied_identity = self._identity_files(identity_reference_paths)
        with self._connection() as connection:
            series = connection.execute(
                "SELECT * FROM series WHERE series_id = ?", (normalized_series_id,)
            ).fetchone()
            if series is None:
                if not supplied_identity:
                    raise ValueError("A new series requires at least one fixed character identity image.")
                identity_files = supplied_identity
            else:
                if series["character_name"] != normalized_character:
                    raise ValueError("The requested character differs from this series' fixed character.")
                identity_files = self._active_identity_files(connection, normalized_series_id)
                if not identity_files:
                    raise ValueError("This series has no active fixed character references.")
                if supplied_identity and supplied_identity != identity_files:
                    raise ValueError("Character reference paths or bytes changed; serial generation is blocked.")
                for item in identity_files:
                    self._validate_asset(item["path"], item["sha256"])

            rows = connection.execute(
                """SELECT * FROM episodes
                   WHERE series_id = ? AND is_current = 1
                   ORDER BY episode_number DESC, version DESC""",
                (normalized_series_id,),
            ).fetchall()
            current_by_number = {int(row["episode_number"]): row for row in rows}
            if episode_number is None:
                if rows and rows[0]["status"] != "ready":
                    raise ValueError(
                        f"Episode {rows[0]['episode_number']} is not ready; pass --serial-resume-episode-number to resume it."
                    )
                target_number = int(rows[0]["episode_number"]) + 1 if rows else 1
            else:
                target_number = int(episode_number)
                if target_number < 1:
                    raise ValueError("serial episode number must be positive")

            previous = None
            if target_number > 1:
                previous = current_by_number.get(target_number - 1)
                if previous is None or previous["status"] != "ready":
                    raise ValueError(
                        f"Episode {target_number - 1} must be ready before episode {target_number} can start."
                    )
            reference_manifest: list[dict[str, str]] = []
            previous_tail: dict[str, Any] | None = None
            story_context = ""
            if previous is not None:
                tail = connection.execute(
                    """SELECT * FROM reference_assets
                       WHERE episode_id = ? AND role = 'episode_end' AND status = 'active'
                       ORDER BY created_at DESC LIMIT 1""",
                    (previous["episode_id"],),
                ).fetchone()
                if tail is None:
                    raise ValueError("The prior episode has no current approved ending frame.")
                self._validate_asset(tail["path"], tail["sha256"])
                terminal = connection.execute(
                    """SELECT s.approved_render_id, r.output_path, r.output_sha256
                       FROM segments s JOIN render_attempts r ON r.render_id = s.approved_render_id
                       WHERE s.episode_id = ? ORDER BY s.ordinal DESC LIMIT 1""",
                    (previous["episode_id"],),
                ).fetchone()
                if (
                    terminal is None
                    or terminal["approved_render_id"] != tail["source_render_id"]
                    or not terminal["output_path"]
                ):
                    raise ValueError("The prior episode ending frame no longer belongs to its approved final video.")
                self._validate_asset(terminal["output_path"], terminal["output_sha256"])
                previous_tail = dict(tail)
                reference_manifest.append(
                    {
                        "path": tail["path"],
                        "type": "image",
                        "role": "continuation",
                        "retention": "Use only for story state and character continuity; create a new shot composition and camera framing.",
                        "notes": "Current approved ending frame from the previous episode.",
                    }
                )
                story_context = self._story_context(previous)

            for item in identity_files:
                reference_manifest.append(
                    {
                        "path": item["path"],
                        "type": "image",
                        "role": "identity",
                        "retention": "Preserve the fixed character identity while choosing a new pose, action, and shot composition.",
                        "notes": "Fixed character identity reference for this series.",
                    }
                )

            return {
                "series_id": normalized_series_id,
                "character": normalized_character,
                "episode_number": target_number,
                "previous_episode_id": str(previous["episode_id"]) if previous else "",
                "story_context": story_context,
                "identity_references": identity_files,
                "previous_tail": previous_tail,
                "reference_manifest": reference_manifest,
            }

    @staticmethod
    def _story_context(previous: dict[str, Any]) -> str:
        story = json.loads(previous["story_json"] or "[]")
        if not isinstance(story, list) or not story:
            return ""
        beats: list[str] = []
        for item in story:
            if not isinstance(item, dict):
                continue
            visual = str(item.get("visual") or item.get("narration") or "").strip()
            ending = str(item.get("end_state") or item.get("effect") or "").strip()
            beat = "；".join(value for value in (visual, ending) if value)
            if beat:
                beats.append(beat)
        if not beats:
            return ""
        return f"Previous episode {previous['episode_number']} story state: " + " → ".join(beats)

    def start_episode(
        self,
        context: dict[str, Any],
        prompt: str,
        *,
        topic: str = "",
        target_duration_seconds: int = 30,
    ) -> dict[str, Any]:
        """Persist a new episode version and bind its fixed/reference assets."""
        series_id = str(context["series_id"])
        character = str(context["character"])
        episode_number = int(context["episode_number"])
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("A serial episode requires a story premise.")
        topic = str(topic or context.get("episode_topic") or prompt).strip()
        duration = int(target_duration_seconds or context.get("target_duration_seconds") or 30)
        if not topic:
            raise ValueError("A serial episode requires an explicit topic.")
        if duration < 1:
            raise ValueError("A serial episode target duration must be positive.")
        episode_id = uuid.uuid4().hex
        now = _now()
        with self._connection(write=True) as connection:
            series = connection.execute(
                "SELECT * FROM series WHERE series_id = ?", (series_id,)
            ).fetchone()
            if series is not None and series["lifecycle_status"] != "active":
                raise ValueError("A completed or discarded series cannot start or rerender an episode.")
            identity = list(context["identity_references"])
            if series is None:
                connection.execute(
                    "INSERT INTO series(series_id, character_name, story_state_json, created_at, updated_at) VALUES (?, ?, '{}', ?, ?)",
                    (series_id, character, now, now),
                )
                for item in identity:
                    self._insert_reference(
                        connection,
                        series_id=series_id,
                        episode_id=None,
                        segment_number=0,
                        role="character_identity",
                        source_render_id=None,
                        path=item["path"],
                        sha256=item["sha256"],
                        now=now,
                    )
            else:
                if series["character_name"] != character:
                    raise ValueError("The requested character differs from this series' fixed character.")
                if self._active_identity_files(connection, series_id) != identity:
                    raise ValueError("Character reference assets changed while the episode was being prepared.")

            previous_episode_id = str(context.get("previous_episode_id") or "") or None
            if previous_episode_id:
                previous = connection.execute(
                    "SELECT * FROM episodes WHERE episode_id = ? AND is_current = 1",
                    (previous_episode_id,),
                ).fetchone()
                if previous is None or previous["status"] != "ready":
                    raise ValueError("The prior episode changed before the new episode could be started.")

            current_same_number = connection.execute(
                """SELECT * FROM episodes WHERE series_id = ? AND episode_number = ? AND is_current = 1""",
                (series_id, episode_number),
            ).fetchone()
            version_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM episodes WHERE series_id = ? AND episode_number = ?",
                (series_id, episode_number),
            ).fetchone()
            version = int(version_row["version"]) + 1
            if current_same_number is not None:
                self._stale_episode_tree(connection, series_id, episode_number)

            connection.execute(
                """INSERT INTO episodes(
                       episode_id, series_id, episode_number, version, is_current, prompt,
                       episode_topic, target_duration_seconds, status, previous_episode_id,
                       story_context_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, 'planned', ?, ?, ?, ?)""",
                (
                    episode_id,
                    series_id,
                    episode_number,
                    version,
                    prompt,
                    topic,
                    duration,
                    previous_episode_id,
                    _json(
                        {
                            "story_context": context.get("story_context", ""),
                            "automation_mode": context.get("automation_mode", "manual"),
                            "automation_source_series_id": context.get("automation_source_series_id", ""),
                            "automation_source_episode_id": context.get("automation_source_episode_id", ""),
                        }
                    ),
                    now,
                    now,
                ),
            )
            previous_tail = context.get("previous_tail")
            if previous_tail:
                # Recheck the frame bytes at the moment the episode binds them.
                self._validate_asset(previous_tail["path"], previous_tail["sha256"])
                self._insert_reference(
                    connection,
                    series_id=series_id,
                    episode_id=episode_id,
                    segment_number=0,
                    role="episode_opening",
                    source_render_id=previous_tail["source_render_id"],
                    path=previous_tail["path"],
                    sha256=previous_tail["sha256"],
                    now=now,
                )
            connection.execute(
                "UPDATE series SET updated_at = ? WHERE series_id = ?", (now, series_id)
            )
        return {
            "episode_id": episode_id,
            "series_id": series_id,
            "episode_number": episode_number,
            "version": version,
            "episode_topic": topic,
            "target_duration_seconds": duration,
        }

    def _insert_reference(
        self,
        connection: _MySQLSession,
        *,
        series_id: str,
        episode_id: str | None,
        segment_number: int,
        role: str,
        source_render_id: str | None,
        path: str,
        sha256: str,
        now: str,
    ) -> str:
        asset_id = uuid.uuid4().hex
        connection.execute(
            """INSERT INTO reference_assets(
                   asset_id, series_id, episode_id, segment_number, role,
                   source_render_id, path, sha256, status, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
            (asset_id, series_id, episode_id, segment_number, role, source_render_id, path, sha256, now),
        )
        return asset_id

    def record_plan(self, episode_id: str, *, segment_count: int, plan_payload: dict[str, Any]) -> str:
        count = int(segment_count)
        if count < 1:
            raise ValueError("A serial episode must plan at least one video segment.")
        plan_sha = hashlib.sha256(_json(plan_payload).encode("utf-8")).hexdigest()
        now = _now()
        with self._connection(write=True) as connection:
            episode = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ? AND is_current = 1", (episode_id,)
            ).fetchone()
            if episode is None:
                raise ValueError("Cannot plan an episode version that is no longer current.")
            existing_segments = connection.execute(
                "SELECT COUNT(*) AS row_count FROM segments WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if existing_segments and int(existing_segments["row_count"]):
                raise ValueError("Episode plan is already persisted; start a new episode version to replan it.")
            goal_payload = plan_payload.get("goal", {})
            duration = (
                float(goal_payload.get("duration_seconds") or episode["target_duration_seconds"])
                if isinstance(goal_payload, dict)
                else float(episode["target_duration_seconds"])
            )
            segment_duration = duration / count
            connection.execute(
                """UPDATE episodes SET plan_sha256 = ?, target_duration_seconds = ?,
                       status = 'planned', updated_at = ? WHERE episode_id = ?""",
                (plan_sha, round(duration), now, episode_id),
            )
            connection.execute(
                "INSERT INTO episode_plans(episode_id, plan_sha256, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (episode_id, plan_sha, _json(plan_payload), now),
            )
            for ordinal in range(1, count + 1):
                connection.execute(
                    """INSERT INTO segments(
                           segment_id, episode_id, ordinal, target_duration_seconds, status,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, 'planned', ?, ?)""",
                    (uuid.uuid4().hex, episode_id, ordinal, segment_duration, now, now),
                )
        return plan_sha

    def get_current_episode(self, series_id: str, episode_number: int) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM episodes WHERE series_id = ? AND episode_number = ? AND is_current = 1""",
                (str(series_id).strip(), int(episode_number)),
            ).fetchone()
            if row is None:
                raise ValueError("The requested current serial episode does not exist.")
            return dict(row)

    def get_plan_payload(self, episode_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            episode = connection.execute(
                "SELECT plan_sha256 FROM episodes WHERE episode_id = ? AND is_current = 1",
                (episode_id,),
            ).fetchone()
            if episode is None:
                raise ValueError("Cannot load a plan for a stale episode version.")
            row = connection.execute(
                "SELECT plan_sha256, plan_json FROM episode_plans WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None:
                return None
            if row["plan_sha256"] != episode["plan_sha256"]:
                raise ValueError("The persisted serial plan no longer matches its episode checksum.")
            payload = json.loads(row["plan_json"])
            if not isinstance(payload, dict):
                raise ValueError("The persisted serial plan is not an object.")
            return payload

    def store_reconstructed_plan(self, episode_id: str, plan_payload: dict[str, Any]) -> str:
        """Bind a rebuilt plan only when it exactly matches the episode's original checksum."""
        plan_sha = hashlib.sha256(_json(plan_payload).encode("utf-8")).hexdigest()
        now = _now()
        with self._connection(write=True) as connection:
            episode = connection.execute(
                "SELECT plan_sha256 FROM episodes WHERE episode_id = ? AND is_current = 1",
                (episode_id,),
            ).fetchone()
            if episode is None or episode["plan_sha256"] != plan_sha:
                raise ValueError("The rebuilt plan differs from the blocked episode's persisted plan checksum.")
            existing = connection.execute(
                "SELECT plan_sha256, plan_json FROM episode_plans WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if existing is not None:
                if existing["plan_sha256"] != plan_sha or existing["plan_json"] != _json(plan_payload):
                    raise ValueError("A different plan payload is already bound to this episode.")
                return plan_sha
            connection.execute(
                "INSERT INTO episode_plans(episode_id, plan_sha256, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (episode_id, plan_sha, _json(plan_payload), now),
            )
        return plan_sha

    def begin_publication_attempt(
        self,
        episode_id: str,
        *,
        platforms: Sequence[str],
        episode_topic: str,
        caption: str,
        hashtags: str,
        media_path: str | Path,
    ) -> dict[str, Any]:
        """Persist a pending Discord approval and one delivery row per target platform."""
        normalized_platforms = list(dict.fromkeys(str(value).strip() for value in platforms if str(value).strip()))
        if not normalized_platforms:
            raise ValueError("A serial publication attempt requires at least one platform.")
        topic = str(episode_topic or "").strip()
        if not topic:
            raise ValueError("A serial publication attempt requires an explicit episode topic.")
        media = _existing_file(media_path)
        if media.suffix.lower() != ".mp4":
            raise ValueError("Serial final publication requires the assembled MP4.")
        media_sha = _sha256(media)
        now = _now()
        with self._connection(write=True) as connection:
            episode = connection.execute(
                """SELECT e.*, s.lifecycle_status FROM episodes e
                   JOIN series s ON s.series_id = e.series_id
                   WHERE e.episode_id = ? AND e.is_current = 1""",
                (episode_id,),
            ).fetchone()
            if episode is None:
                raise ValueError("Cannot publish a stale episode version.")
            if episode["status"] != "ready" or not episode["final_video_path"]:
                raise ValueError("Only a final-QA-passed ready episode can enter publication review.")
            if episode["lifecycle_status"] == "discarded":
                raise ValueError("A discarded series cannot start a publication attempt.")
            if episode["episode_topic"] != topic:
                raise ValueError("Publication topic differs from the topic bound to this episode.")
            if (
                episode["final_video_path"] != str(media)
                or episode["final_video_sha256"] != media_sha
            ):
                raise ValueError("Publication media differs from the current assembled episode video.")
            successful_rows = connection.execute(
                """SELECT DISTINCT platform FROM episode_publications
                   WHERE episode_id = ? AND status = 'reported_success'""",
                (episode_id,),
            ).fetchall()
            already_delivered = {str(row["platform"]) for row in successful_rows}
            duplicate_targets = [
                platform for platform in normalized_platforms if platform in already_delivered
            ]
            if duplicate_targets:
                raise ValueError(
                    "Platforms with an existing reported-success receipt cannot be dispatched again: "
                    + ", ".join(duplicate_targets)
                )
            attempt_row = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) AS attempt_number FROM episode_publications WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            attempt_number = int(attempt_row["attempt_number"]) + 1
            for platform in normalized_platforms:
                connection.execute(
                    """INSERT INTO episode_publications(
                           publication_id, episode_id, attempt_number, platform, episode_topic,
                           caption, hashtags, media_path, media_sha256, status, started_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'review_pending', ?)""",
                    (
                        uuid.uuid4().hex,
                        episode_id,
                        attempt_number,
                        platform,
                        topic,
                        str(caption or ""),
                        str(hashtags or ""),
                        str(media),
                        media_sha,
                        now,
                    ),
                )
        return {
            "episode_id": episode_id,
            "attempt_number": attempt_number,
            "platforms": normalized_platforms,
            "media_path": str(media),
            "media_sha256": media_sha,
        }

    def record_publication_review(
        self,
        episode_id: str,
        attempt_number: int,
        *,
        decision: str,
        review_session_id: str = "",
        caption: str | None = None,
        hashtags: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        normalized = str(decision or "").strip().lower()
        status_by_decision = {
            "accepted": "approved",
            "rejected": "review_rejected",
            "timeout": "review_timeout",
            "failed": "review_failed",
            "unavailable": "review_unavailable",
        }
        if normalized not in status_by_decision:
            raise ValueError(f"Unsupported publication review decision: {decision}")
        with self._connection(write=True) as connection:
            rows = connection.execute(
                "SELECT publication_id FROM episode_publications WHERE episode_id = ? AND attempt_number = ?",
                (episode_id, int(attempt_number)),
            ).fetchall()
            if not rows:
                raise ValueError("Serial publication attempt does not exist.")
            for row in rows:
                connection.execute(
                    """UPDATE episode_publications SET status = ?, review_decision = ?,
                           review_session_id = ?, caption = COALESCE(?, caption),
                           hashtags = COALESCE(?, hashtags), receipt_json = ?
                       WHERE publication_id = ?""",
                    (
                        status_by_decision[normalized],
                        normalized,
                        str(review_session_id or ""),
                        caption,
                        hashtags,
                        _json(details or {}),
                        row["publication_id"],
                    ),
                )

    def record_publication_dispatch(
        self,
        episode_id: str,
        attempt_number: int,
        *,
        results: dict[str, Any],
        receipts: dict[str, Any],
        errors: dict[str, Any],
        dispatch_status: str,
    ) -> None:
        now = _now()
        with self._connection(write=True) as connection:
            rows = connection.execute(
                "SELECT * FROM episode_publications WHERE episode_id = ? AND attempt_number = ?",
                (episode_id, int(attempt_number)),
            ).fetchall()
            if not rows:
                raise ValueError("Serial publication attempt does not exist.")
            for row in rows:
                if row["review_decision"] != "accepted":
                    raise ValueError("Social dispatch requires an accepted Discord publication review.")
                platform = str(row["platform"])
                succeeded = bool(results.get(platform, False))
                receipt = receipts.get(platform)
                error = errors.get(platform)
                if str(dispatch_status).lower() == "dry_run":
                    status = "dry_run"
                elif succeeded and isinstance(receipt, dict) and receipt:
                    status = "reported_success"
                elif succeeded:
                    status = "receipt_missing"
                    error = error or "provider reported success without a delivery receipt"
                else:
                    status = "failed"
                connection.execute(
                    """UPDATE episode_publications SET status = ?, receipt_json = ?, error = ?,
                           finished_at = ? WHERE publication_id = ?""",
                    (
                        status,
                        _json({"dispatch_status": dispatch_status, "receipt": receipt}),
                        str(error or ""),
                        now,
                        row["publication_id"],
                    ),
                )

    def get_episode_publications(self, episode_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return connection.execute(
                "SELECT * FROM episode_publications WHERE episode_id = ? ORDER BY attempt_number, platform",
                (episode_id,),
            ).fetchall()

    def platforms_without_successful_receipt(
        self, episode_id: str, platforms: Sequence[str]
    ) -> list[str]:
        requested = list(dict.fromkeys(str(value).strip() for value in platforms if str(value).strip()))
        if not requested:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT DISTINCT platform FROM episode_publications
                   WHERE episode_id = ? AND status = 'reported_success'""",
                (episode_id,),
            ).fetchall()
        delivered = {str(row["platform"]) for row in rows}
        return [platform for platform in requested if platform not in delivered]

    def assert_publication_media_current(
        self, episode_id: str, attempt_number: int
    ) -> dict[str, str]:
        """Require the approved publication attempt to still reference the ready episode bytes."""
        with self._connection() as connection:
            episode = connection.execute(
                """SELECT e.*, s.lifecycle_status FROM episodes e
                   JOIN series s ON s.series_id = e.series_id
                   WHERE e.episode_id = ? AND e.is_current = 1""",
                (episode_id,),
            ).fetchone()
            if episode is None or episode["status"] != "ready":
                raise ValueError("Only the current ready episode can be published.")
            if episode["lifecycle_status"] == "discarded":
                raise ValueError("A discarded series cannot be published.")
            rows = connection.execute(
                "SELECT * FROM episode_publications WHERE episode_id = ? AND attempt_number = ?",
                (episode_id, int(attempt_number)),
            ).fetchall()
            if not rows or any(
                row["status"] != "approved" or row["review_decision"] != "accepted"
                for row in rows
            ):
                raise ValueError("Social dispatch requires an approved publication attempt.")
            media = self._validate_asset(
                episode["final_video_path"], episode["final_video_sha256"]
            )
            digest = _sha256(media)
            if any(
                row["media_path"] != str(media) or row["media_sha256"] != digest
                for row in rows
            ):
                raise ValueError("The reviewed publication media no longer matches the ready episode SHA.")
            return {"path": str(media), "sha256": digest}

    def resume_episode(self, episode_id: str) -> dict[str, Any]:
        """Reopen a blocked episode using only its contiguous, current accepted render prefix."""
        now = _now()
        initial_node_outputs: dict[str, Any] = {}
        with self._connection(write=True) as connection:
            episode = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ? AND is_current = 1",
                (episode_id,),
            ).fetchone()
            if episode is None or episode["status"] != "blocked":
                raise ValueError("Only the current blocked serial episode can be resumed.")
            plan = connection.execute(
                "SELECT plan_sha256 FROM episode_plans WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if plan is None or plan["plan_sha256"] != episode["plan_sha256"]:
                raise ValueError("The blocked episode has no verified persisted execution plan.")
            segments = connection.execute(
                """SELECT s.*, r.generated_output_path, r.generated_output_sha256,
                          r.output_path, r.output_sha256, r.status AS render_status,
                          r.h3_mode, r.workflow_name, rv.technical_status,
                          rv.technical_json, rv.human_decision, rv.human_json,
                          rv.video_sha256, rv.stale AS review_stale
                   FROM segments s
                   LEFT JOIN render_attempts r ON r.render_id = s.current_render_id
                   LEFT JOIN reviews rv ON rv.render_id = s.current_render_id
                   WHERE s.episode_id = ? ORDER BY s.ordinal""",
                (episode_id,),
            ).fetchall()
            if not segments:
                raise ValueError("The blocked episode has no persisted segments to resume.")
            story = json.loads(episode["story_json"] or "[]")
            story_is_complete = isinstance(story, list) and len(story) == len(segments)

            accepted_prefix = 0
            gap_found = False
            for segment in segments:
                accepted = (
                    segment["status"] == "accepted"
                    and segment["approved_render_id"]
                    and segment["approved_render_id"] == segment["current_render_id"]
                    and segment["render_status"] == "rendered"
                    and segment["technical_status"] == "passed"
                    and segment["human_decision"] == "accepted"
                    and not segment["review_stale"]
                    and segment["video_sha256"] == segment["output_sha256"]
                    and segment["output_path"]
                )
                if accepted and gap_found:
                    raise ValueError("Accepted segments cannot appear after an unapproved segment gap.")
                if not accepted:
                    gap_found = True
                    continue
                generated_path = self._validate_asset(
                    segment["generated_output_path"], segment["generated_output_sha256"]
                )
                path = self._validate_asset(segment["output_path"], segment["output_sha256"])
                ordinal = int(segment["ordinal"])
                review_payload = json.loads(segment["human_json"] or "{}")
                technical_payload = json.loads(segment["technical_json"] or "{}")
                initial_node_outputs[f"segment-video-{ordinal:02d}"] = {
                    "video_path": str(generated_path),
                    "final_video_path": str(generated_path),
                    "saved_files": [str(generated_path)],
                    "h3_mode": segment["h3_mode"],
                    "workflow_name": segment["workflow_name"],
                }
                if path != generated_path:
                    initial_node_outputs[f"segment-trim-{ordinal:02d}"] = {
                        "video_path": str(path),
                        "final_video_path": str(path),
                        "saved_files": [str(path)],
                    }
                initial_node_outputs[f"segment-qa-{ordinal:02d}"] = technical_payload
                initial_node_outputs[f"segment-review-{ordinal:02d}"] = {
                    **(review_payload if isinstance(review_payload, dict) else {}),
                    "human_decision": "accepted",
                    "video_path": str(path),
                    "media_paths": [str(path)],
                    "selected_assets": [str(path)],
                }
                tail = connection.execute(
                    """SELECT * FROM reference_assets WHERE episode_id = ? AND segment_number = ?
                       AND role IN ('segment_tail', 'episode_end') AND status = 'active'
                       ORDER BY created_at DESC LIMIT 1""",
                    (episode_id, ordinal),
                ).fetchone()
                if tail is not None:
                    tail_path = self._validate_asset(tail["path"], tail["sha256"])
                    if tail["source_render_id"] != segment["approved_render_id"]:
                        raise ValueError("An accepted segment's saved tail belongs to another render version.")
                    initial_node_outputs[f"segment-tail-{ordinal:02d}"] = {
                        "frame_path": str(tail_path),
                        "saved_files": [str(tail_path)],
                    }
                accepted_prefix += 1

            if accepted_prefix and not story_is_complete:
                raise ValueError("Accepted segments require their complete persisted story plan.")
            if story_is_complete:
                initial_node_outputs["script-plan"] = {
                    "segments": story,
                    "segment_count": len(story),
                }
            connection.execute(
                "UPDATE episodes SET status = 'planned', error = '', updated_at = ? WHERE episode_id = ?",
                (now, episode_id),
            )
        return {
            "episode_id": episode_id,
            "completed_segment_count": accepted_prefix,
            "next_segment_number": accepted_prefix + 1,
            "initial_node_outputs": initial_node_outputs,
        }

    def record_story(self, episode_id: str, segments: Sequence[dict[str, Any]]) -> None:
        now = _now()
        story = [dict(item) for item in segments]
        with self._connection(write=True) as connection:
            rows = connection.execute(
                "SELECT segment_id, ordinal FROM segments WHERE episode_id = ? ORDER BY ordinal",
                (episode_id,),
            ).fetchall()
            if len(rows) != len(story):
                raise ValueError("Story plan segment count differs from the persisted episode plan.")
            for row, item in zip(rows, story, strict=True):
                connection.execute(
                    "UPDATE segments SET story_json = ?, updated_at = ? WHERE segment_id = ?",
                    (_json(item), now, row["segment_id"]),
                )
            connection.execute(
                "UPDATE episodes SET story_json = ?, updated_at = ? WHERE episode_id = ?",
                (_json(story), now, episode_id),
            )

    def _episode_and_segment(
        self, connection: _MySQLSession, episode_id: str, segment_number: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        episode = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ? AND is_current = 1", (episode_id,)
        ).fetchone()
        segment = connection.execute(
            "SELECT * FROM segments WHERE episode_id = ? AND ordinal = ?",
            (episode_id, int(segment_number)),
        ).fetchone()
        if episode is None or segment is None:
            raise ValueError("Serial episode or segment is not current and persisted.")
        return episode, segment

    def _expected_episode_references(
        self, connection: _MySQLSession, episode: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return connection.execute(
            """SELECT * FROM reference_assets
               WHERE series_id = ? AND status = 'active' AND (
                   (episode_id IS NULL AND role = 'character_identity') OR
                   (episode_id = ? AND role = 'episode_opening')
               )
               ORDER BY CASE WHEN role = 'episode_opening' THEN 0 ELSE 1 END,
                        created_at, asset_id""",
            (episode["series_id"], episode["episode_id"]),
        ).fetchall()

    def _verify_reference_bundle(
        self,
        connection: _MySQLSession,
        episode: dict[str, Any],
        conditioning_paths: Sequence[str | Path],
    ) -> list[dict[str, str]]:
        expected = self._expected_episode_references(connection, episode)
        if not expected:
            raise ValueError("Ref2VA has no registered series identity references.")
        if len(expected) > 4:
            raise ValueError("Ref2VA reference bundle exceeds the configured four-image workflow limit.")
        bound: list[dict[str, str]] = []
        for item in expected:
            path = self._validate_asset(item["path"], item["sha256"])
            bound.append({"path": str(path), "sha256": item["sha256"], "role": item["role"]})
        supplied = [_existing_file(path) for path in conditioning_paths]
        if [str(path) for path in supplied] != [item["path"] for item in bound]:
            raise ValueError("Ref2VA inputs do not match the current episode's approved reference bundle.")
        return bound

    def _verify_prior_tail(
        self,
        connection: _MySQLSession,
        episode: dict[str, Any],
        segment_number: int,
        conditioning_paths: Sequence[str | Path],
    ) -> tuple[dict[str, str], str]:
        if segment_number <= 1:
            raise ValueError("Serial I2VA continuation is only valid after the first segment.")
        previous = connection.execute(
            "SELECT * FROM segments WHERE episode_id = ? AND ordinal = ?",
            (episode["episode_id"], segment_number - 1),
        ).fetchone()
        if previous is None or not previous["approved_render_id"]:
            raise ValueError("The immediately preceding segment has no accepted render version.")
        review = connection.execute(
            """SELECT * FROM reviews WHERE render_id = ? AND stale = 0
               AND technical_status = 'passed' AND human_decision = 'accepted'""",
            (previous["approved_render_id"],),
        ).fetchone()
        render = connection.execute(
            "SELECT * FROM render_attempts WHERE render_id = ? AND status = 'rendered'",
            (previous["approved_render_id"],),
        ).fetchone()
        tail = connection.execute(
            """SELECT * FROM reference_assets WHERE episode_id = ? AND segment_number = ?
               AND role = 'segment_tail' AND status = 'active'
               ORDER BY created_at DESC LIMIT 1""",
            (episode["episode_id"], segment_number - 1),
        ).fetchone()
        if review is None or render is None or tail is None:
            raise ValueError("The prior segment lacks current QA, acceptance, or an extracted tail frame.")
        self._validate_asset(render["output_path"], render["output_sha256"])
        if (
            previous["status"] != "accepted"
            or previous["current_render_id"] != previous["approved_render_id"]
        ):
            raise ValueError("The prior segment is no longer the current accepted render.")
        if review["video_sha256"] != render["output_sha256"]:
            raise ValueError("The prior segment review is bound to a different video version.")
        frame = self._validate_asset(tail["path"], tail["sha256"])
        if tail["source_render_id"] != render["render_id"]:
            raise ValueError("The prior segment tail was extracted from an obsolete render version.")
        supplied = [_existing_file(path) for path in conditioning_paths]
        if [str(path) for path in supplied] != [str(frame)]:
            raise ValueError("I2VA must use only the immediately preceding accepted segment's actual tail frame.")
        return {"path": str(frame), "sha256": tail["sha256"], "role": "segment_tail"}, str(render["render_id"])

    def _stale_episode_tree(
        self, connection: _MySQLSession, series_id: str, from_episode_number: int
    ) -> None:
        episode_rows = connection.execute(
            """SELECT episode_id FROM episodes WHERE series_id = ? AND episode_number >= ? AND is_current = 1""",
            (series_id, int(from_episode_number)),
        ).fetchall()
        episode_ids = [row["episode_id"] for row in episode_rows]
        now = _now()
        for episode_id in episode_ids:
            segment_rows = connection.execute(
                "SELECT segment_id, ordinal, current_render_id FROM segments WHERE episode_id = ?",
                (episode_id,),
            ).fetchall()
            render_ids = [row["current_render_id"] for row in segment_rows if row["current_render_id"]]
            connection.execute(
                "UPDATE reviews SET stale = 1, updated_at = ? WHERE segment_id IN (SELECT segment_id FROM segments WHERE episode_id = ?)",
                (now, episode_id),
            )
            connection.execute(
                "UPDATE segments SET status = 'stale', approved_render_id = NULL, tail_asset_id = NULL, updated_at = ? WHERE episode_id = ?",
                (now, episode_id),
            )
            connection.execute(
                "UPDATE reference_assets SET status = 'stale' WHERE episode_id = ?",
                (episode_id,),
            )
            connection.execute(
                "UPDATE episodes SET is_current = 0, status = 'stale', final_video_path = '', final_video_sha256 = '', updated_at = ? WHERE episode_id = ?",
                (now, episode_id),
            )
            for render_id in render_ids:
                connection.execute(
                    "UPDATE render_attempts SET status = 'superseded' WHERE render_id = ? AND status IN ('running', 'rendered')",
                    (render_id,),
                )

    def _invalidate_after_segment(
        self,
        connection: _MySQLSession,
        episode: dict[str, Any],
        segment: dict[str, Any],
    ) -> None:
        now = _now()
        same_episode_rows = connection.execute(
            """SELECT segment_id, ordinal, current_render_id FROM segments
               WHERE episode_id = ? AND ordinal >= ?""",
            (episode["episode_id"], segment["ordinal"]),
        ).fetchall()
        for row in same_episode_rows:
            connection.execute(
                "UPDATE reviews SET stale = 1, updated_at = ? WHERE segment_id = ?",
                (now, row["segment_id"]),
            )
            connection.execute(
                "UPDATE reference_assets SET status = 'stale' WHERE episode_id = ? AND segment_number = ?",
                (episode["episode_id"], row["ordinal"]),
            )
            if row["current_render_id"]:
                connection.execute(
                    "UPDATE render_attempts SET status = 'superseded' WHERE render_id = ? AND status IN ('running', 'rendered')",
                    (row["current_render_id"],),
                )
        connection.execute(
            """UPDATE segments SET status = 'stale', approved_render_id = NULL, tail_asset_id = NULL, updated_at = ?
               WHERE episode_id = ? AND ordinal > ?""",
            (now, episode["episode_id"], segment["ordinal"]),
        )
        connection.execute(
            "UPDATE segments SET status = 'rendering', approved_render_id = NULL, tail_asset_id = NULL, updated_at = ? WHERE segment_id = ?",
            (now, segment["segment_id"]),
        )
        connection.execute(
            "UPDATE episodes SET status = 'rendering', final_video_path = '', final_video_sha256 = '', error = '', updated_at = ? WHERE episode_id = ?",
            (now, episode["episode_id"]),
        )
        self._stale_episode_tree(connection, episode["series_id"], int(episode["episode_number"]) + 1)

    def begin_render(
        self,
        episode_id: str,
        segment_number: int,
        *,
        workflow_name: str,
        h3_mode: str,
        conditioning_paths: Sequence[str | Path],
        render_inputs: dict[str, Any] | None = None,
    ) -> str:
        mode = str(h3_mode or "").strip().lower()
        with self._connection(write=True) as connection:
            episode, segment = self._episode_and_segment(connection, episode_id, segment_number)
            if mode == "ref2va" and int(segment_number) == 1:
                conditioning = self._verify_reference_bundle(connection, episode, conditioning_paths)
                source_render_id = ""
            elif mode == "i2va" and int(segment_number) > 1:
                tail, source_render_id = self._verify_prior_tail(
                    connection, episode, int(segment_number), conditioning_paths
                )
                conditioning = [tail]
            else:
                raise ValueError("Serial H3 requires Ref2VA for the first segment and I2VA for every continuation.")

            self._invalidate_after_segment(connection, episode, segment)
            attempt_count = connection.execute(
                "SELECT COUNT(*) AS row_count FROM render_attempts WHERE segment_id = ?",
                (segment["segment_id"],),
            ).fetchone()
            attempt_number = int(attempt_count["row_count"] if attempt_count else 0) + 1
            attempt_id = uuid.uuid4().hex
            input_record = {
                "workflow_name": str(workflow_name),
                "h3_mode": mode,
                "conditioning": conditioning,
                "source_render_id": source_render_id,
                "render_inputs": dict(render_inputs or {}),
            }
            input_sha = hashlib.sha256(_json(input_record).encode("utf-8")).hexdigest()
            connection.execute(
                """INSERT INTO render_attempts(
                       render_id, segment_id, attempt_number, status, workflow_name,
                       h3_mode, conditioning_json, input_sha256, started_at
                   ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    segment["segment_id"],
                    attempt_number,
                    str(workflow_name),
                    mode,
                    _json(input_record),
                    input_sha,
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE segments SET current_render_id = ?, updated_at = ? WHERE segment_id = ?",
                (attempt_id, _now(), segment["segment_id"]),
            )
        return attempt_id

    def complete_render(self, render_id: str, video_path: str | Path) -> dict[str, str]:
        path = _existing_file(video_path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("Every serial segment render must be a generated MP4 video.")
        from agentic.tools.comfy_h3 import probe_video

        probe_video(path)
        digest = _sha256(path)
        now = _now()
        with self._connection(write=True) as connection:
            attempt = connection.execute(
                "SELECT * FROM render_attempts WHERE render_id = ?", (render_id,)
            ).fetchone()
            if attempt is None or attempt["status"] != "running":
                raise ValueError("Render attempt is not active.")
            segment = connection.execute(
                "SELECT * FROM segments WHERE segment_id = ?", (attempt["segment_id"],)
            ).fetchone()
            if segment is None or segment["current_render_id"] != render_id:
                raise ValueError("Render attempt was superseded before its video was recorded.")
            connection.execute(
                """UPDATE render_attempts SET status = 'rendered', output_path = ?, output_sha256 = ?,
                       finished_at = ?, generated_output_path = ?, generated_output_sha256 = ?
                   WHERE render_id = ?""",
                (str(path), digest, now, str(path), digest, render_id),
            )
            connection.execute(
                "UPDATE segments SET status = 'rendered', updated_at = ? WHERE segment_id = ?",
                (now, segment["segment_id"]),
            )
        return {"path": str(path), "sha256": digest}

    def bind_delivered_video(
        self, episode_id: str, segment_number: int, video_path: str | Path
    ) -> dict[str, str]:
        """Bind a duration-normalized MP4 while retaining the original H3 render."""
        path = _existing_file(video_path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("A delivered serial segment must be an MP4 video.")
        from agentic.tools.comfy_h3 import probe_video

        probe_video(path)
        digest = _sha256(path)
        now = _now()
        with self._connection(write=True) as connection:
            _, segment, render = self._current_render(connection, episode_id, segment_number)
            if render["status"] != "rendered" or not render["generated_output_path"]:
                raise ValueError("A delivered segment must descend from the current completed H3 render.")
            self._validate_asset(render["generated_output_path"], render["generated_output_sha256"])
            connection.execute(
                "UPDATE render_attempts SET output_path = ?, output_sha256 = ? WHERE render_id = ?",
                (str(path), digest, render["render_id"]),
            )
            connection.execute(
                "UPDATE segments SET updated_at = ? WHERE segment_id = ?",
                (now, segment["segment_id"]),
            )
        return {"path": str(path), "sha256": digest}

    def fail_render(self, render_id: str, error: str) -> None:
        now = _now()
        with self._connection(write=True) as connection:
            attempt = connection.execute(
                "SELECT * FROM render_attempts WHERE render_id = ?", (render_id,)
            ).fetchone()
            if attempt is None or attempt["status"] != "running":
                return
            connection.execute(
                "UPDATE render_attempts SET status = 'failed', error = ?, finished_at = ? WHERE render_id = ?",
                (str(error), now, render_id),
            )
            connection.execute(
                "UPDATE segments SET status = 'failed', updated_at = ? WHERE segment_id = ? AND current_render_id = ?",
                (now, attempt["segment_id"], render_id),
            )
            connection.execute(
                "UPDATE episodes SET status = 'failed', error = ?, updated_at = ? WHERE episode_id = (SELECT episode_id FROM segments WHERE segment_id = ?)",
                (str(error), now, attempt["segment_id"]),
            )

    def _current_render(
        self, connection: _MySQLSession, episode_id: str, segment_number: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        episode, segment = self._episode_and_segment(connection, episode_id, segment_number)
        if not segment["current_render_id"]:
            raise ValueError("Segment has no current render attempt.")
        render = connection.execute(
            "SELECT * FROM render_attempts WHERE render_id = ?", (segment["current_render_id"],)
        ).fetchone()
        if render is None:
            raise ValueError("Segment current render attempt is missing.")
        return episode, segment, render

    def record_technical_qa(
        self,
        episode_id: str,
        segment_number: int,
        video_path: str | Path,
        passed: bool,
        report: dict[str, Any],
    ) -> None:
        path = _existing_file(video_path)
        digest = _sha256(path)
        now = _now()
        with self._connection(write=True) as connection:
            episode, segment, render = self._current_render(connection, episode_id, segment_number)
            if render["status"] != "rendered" or render["output_path"] != str(path) or render["output_sha256"] != digest:
                raise ValueError("Technical QA does not match the segment's current rendered video SHA.")
            review_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO reviews(
                       review_id, segment_id, render_id, video_sha256, technical_status,
                       technical_json, human_decision, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                   ON DUPLICATE KEY UPDATE
                       video_sha256 = VALUES(video_sha256),
                       technical_status = VALUES(technical_status),
                       technical_json = VALUES(technical_json),
                       human_decision = 'pending', human_json = '{}', stale = 0,
                       updated_at = VALUES(updated_at)""",
                (
                    review_id,
                    segment["segment_id"],
                    render["render_id"],
                    digest,
                    "passed" if passed else "failed",
                    _json(report),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE segments SET status = ?, updated_at = ? WHERE segment_id = ?",
                ("review_pending" if passed else "qa_failed", now, segment["segment_id"]),
            )
            if not passed:
                connection.execute(
                    "UPDATE episodes SET status = 'qa_failed', error = 'segment technical QA failed', updated_at = ? WHERE episode_id = ?",
                    (now, episode["episode_id"]),
                )

    def record_review(
        self,
        episode_id: str,
        segment_number: int,
        decision: str,
        reviewed_path: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        normalized = str(decision or "").strip().lower()
        if normalized not in {"accepted", "rejected", "timeout", "unavailable", "regeneration_requested", "failed"}:
            raise ValueError(f"Unsupported serial review decision: {decision}")
        now = _now()
        with self._connection(write=True) as connection:
            episode, segment, render = self._current_render(connection, episode_id, segment_number)
            review = connection.execute(
                "SELECT * FROM reviews WHERE render_id = ? AND stale = 0", (render["render_id"],)
            ).fetchone()
            if review is None or review["technical_status"] != "passed":
                raise ValueError("Human review cannot accept or reject a video before technical QA passes.")
            actual_video = self._validate_asset(render["output_path"], render["output_sha256"])
            if review["video_sha256"] != render["output_sha256"]:
                raise ValueError("Human review is bound to a different video SHA.")
            if normalized == "accepted" and reviewed_path is None:
                raise ValueError("An accepted review must identify the reviewed MP4 path.")
            if reviewed_path is not None:
                candidate = _existing_file(reviewed_path)
                if candidate != actual_video or _sha256(candidate) != render["output_sha256"]:
                    raise ValueError("Human review identified a different or changed video version.")
            connection.execute(
                "UPDATE reviews SET human_decision = ?, human_json = ?, updated_at = ? WHERE review_id = ?",
                (normalized, _json(details or {}), now, review["review_id"]),
            )
            if normalized == "accepted":
                connection.execute(
                    "UPDATE segments SET status = 'accepted', approved_render_id = ?, updated_at = ? WHERE segment_id = ?",
                    (render["render_id"], now, segment["segment_id"]),
                )
            else:
                connection.execute(
                    "UPDATE segments SET status = ?, approved_render_id = NULL, updated_at = ? WHERE segment_id = ?",
                    ("rejected" if normalized == "rejected" else "blocked", now, segment["segment_id"]),
                )
                connection.execute(
                    "UPDATE episodes SET status = 'blocked', error = ?, updated_at = ? WHERE episode_id = ?",
                    (f"segment human review {normalized}", now, episode["episode_id"]),
                )
        return {
            "decision": normalized,
            "video_path": str(actual_video),
            "video_sha256": str(render["output_sha256"]),
        }

    def assert_can_continue(
        self, episode_id: str, segment_number: int, video_path: str | Path
    ) -> dict[str, str]:
        path = _existing_file(video_path)
        digest = _sha256(path)
        with self._connection() as connection:
            _, segment, render = self._current_render(connection, episode_id, segment_number)
            review = connection.execute(
                """SELECT * FROM reviews WHERE render_id = ? AND stale = 0
                   AND technical_status = 'passed' AND human_decision = 'accepted'""",
                (render["render_id"],),
            ).fetchone()
            if (
                segment["status"] != "accepted"
                or segment["approved_render_id"] != render["render_id"]
                or render["status"] != "rendered"
                or render["output_path"] != str(path)
                or render["output_sha256"] != digest
                or review is None
                or review["video_sha256"] != digest
            ):
                raise ValueError("Only the current QA-passed, human-accepted video SHA can continue.")
            return {"render_id": render["render_id"], "path": str(path), "sha256": digest}

    def record_tail(
        self,
        episode_id: str,
        segment_number: int,
        source_video_path: str | Path,
        tail_path: str | Path,
        *,
        episode_end: bool = False,
    ) -> dict[str, str]:
        approved = self.assert_can_continue(episode_id, segment_number, source_video_path)
        frame = _existing_file(tail_path)
        if frame.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("A serial continuation tail must be an extracted image frame.")
        digest = _sha256(frame)
        role = "episode_end" if episode_end else "segment_tail"
        now = _now()
        with self._connection(write=True) as connection:
            episode, segment, render = self._current_render(connection, episode_id, segment_number)
            if render["render_id"] != approved["render_id"] or render["output_sha256"] != approved["sha256"]:
                raise ValueError("The approved segment changed before its tail frame was registered.")
            connection.execute(
                """UPDATE reference_assets SET status = 'stale'
                   WHERE episode_id = ? AND segment_number = ? AND role IN ('segment_tail', 'episode_end')""",
                (episode_id, int(segment_number)),
            )
            asset_id = self._insert_reference(
                connection,
                series_id=episode["series_id"],
                episode_id=episode_id,
                segment_number=int(segment_number),
                role=role,
                source_render_id=render["render_id"],
                path=str(frame),
                sha256=digest,
                now=now,
            )
            connection.execute(
                "UPDATE segments SET tail_asset_id = ?, updated_at = ? WHERE segment_id = ?",
                (asset_id, now, segment["segment_id"]),
            )
        return {"asset_id": asset_id, "path": str(frame), "sha256": digest, "source_render_id": approved["render_id"]}

    def complete_episode(
        self, episode_id: str, run_status: str, final_video_path: str | Path | None
    ) -> dict[str, Any]:
        if str(run_status).lower() != "success":
            self.fail_episode(episode_id, f"workflow finished with status {run_status}")
            raise ValueError("Serial episode is not ready because its workflow did not succeed.")
        if final_video_path is None:
            self.fail_episode(episode_id, "workflow did not expose its assembled MP4")
            raise ValueError("Serial episode has no assembled video artifact.")
        final_path = _existing_file(final_video_path)
        if final_path.suffix.lower() != ".mp4":
            self.fail_episode(episode_id, "assembled output was not an MP4 video")
            raise ValueError("Serial episode delivery must be a video MP4.")
        from agentic.tools.comfy_h3 import probe_video

        probe_video(final_path)
        digest = _sha256(final_path)
        now = _now()
        with self._connection(write=True) as connection:
            episode = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ? AND is_current = 1", (episode_id,)
            ).fetchone()
            if episode is None:
                raise ValueError("Cannot finalize a stale episode version.")
            segments = connection.execute(
                """SELECT s.*, r.output_path, r.output_sha256
                   FROM segments s LEFT JOIN render_attempts r ON r.render_id = s.approved_render_id
                   WHERE s.episode_id = ? ORDER BY s.ordinal""",
                (episode_id,),
            ).fetchall()
            if not segments or any(
                row["status"] != "accepted"
                or not row["approved_render_id"]
                or row["approved_render_id"] != row["current_render_id"]
                or not row["output_path"]
                or Path(row["output_path"]).suffix.lower() != ".mp4"
                for row in segments
            ):
                raise ValueError("Episode assembly requires every current segment to be an accepted MP4.")
            for row in segments:
                self._validate_asset(row["output_path"], row["output_sha256"])
                review = connection.execute(
                    """SELECT 1 FROM reviews WHERE render_id = ? AND stale = 0
                       AND video_sha256 = ? AND technical_status = 'passed' AND human_decision = 'accepted'""",
                    (row["approved_render_id"], row["output_sha256"]),
                ).fetchone()
                if review is None:
                    raise ValueError("Episode assembly found a missing or stale segment review.")
            terminal = segments[-1]
            tail = connection.execute(
                """SELECT 1 FROM reference_assets WHERE episode_id = ? AND segment_number = ?
                   AND role = 'episode_end' AND source_render_id = ? AND status = 'active'""",
                (episode_id, terminal["ordinal"], terminal["approved_render_id"]),
            ).fetchone()
            if tail is None:
                raise ValueError("Episode assembly requires the accepted final segment's current extracted tail frame.")
            story = json.loads(episode["story_json"] or "[]")
            ending = {}
            if isinstance(story, list) and story and isinstance(story[-1], dict):
                ending = {key: story[-1].get(key) for key in ("end_state", "effect", "visual") if story[-1].get(key)}
            connection.execute(
                """UPDATE episodes SET status = 'ready', final_video_path = ?, final_video_sha256 = ?,
                       error = '', updated_at = ? WHERE episode_id = ?""",
                (str(final_path), digest, now, episode_id),
            )
            connection.execute(
                "UPDATE series SET story_state_json = ?, updated_at = ? WHERE series_id = ?",
                (_json({"episode_number": episode["episode_number"], "ending": ending}), now, episode["series_id"]),
            )
        return {"episode_id": episode_id, "status": "ready", "final_video_path": str(final_path), "final_video_sha256": digest}

    def fail_episode(self, episode_id: str, error: str) -> None:
        with self._connection(write=True) as connection:
            connection.execute(
                "UPDATE episodes SET status = 'blocked', error = ?, updated_at = ? WHERE episode_id = ? AND is_current = 1",
                (str(error), _now(), episode_id),
            )

    def get_segment(self, episode_id: str, segment_number: int) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM segments WHERE episode_id = ? AND ordinal = ?",
                (episode_id, int(segment_number)),
            ).fetchone()
            if row is None:
                raise ValueError("Serial segment does not exist.")
            return dict(row)

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Serial episode does not exist.")
            return dict(row)


def store_from_constraints(constraints: dict[str, Any]) -> SerialVideoStore | None:
    if not bool(constraints.get("serial_mode")):
        return None
    database_name = str(constraints.get("serial_database_name") or "").strip()
    if not database_name:
        raise ValueError("Serial mode is missing its persistent MySQL database name.")
    return SerialVideoStore(database_name)
