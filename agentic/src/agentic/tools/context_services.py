from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REVIEW_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
MAX_REVIEW_FILE_BYTES = 200 * 1024 * 1024


def _load_env_once() -> None:
    if getattr(_load_env_once, "_loaded", False):
        return
    for env_path in ("media_overload.env", ".env"):
        path = Path(env_path)
        if path.exists():
            load_dotenv(path)
            break
    setattr(_load_env_once, "_loaded", True)


def _logger() -> logging.Logger:
    return logging.getLogger(os.getenv("AGENTIC_RUN_LOGGER_NAME", "mediaoverload.agentic"))


@dataclass(slots=True)
class NewsSelection:
    title: str
    keyword: str
    category: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "keyword": self.keyword,
            "category": self.category,
            "created_at": self.created_at,
        }


class NewsContextService:
    # The Kirby publishing path is family-safe by default. Keep these terms
    # out of the random news pool before they reach the story/image/video
    # models; this is input hygiene, not a hard-coded creative prompt.
    DEFAULT_EXCLUDE_TITLE_TERMS = (
        "性愛",
        "色情",
        "情色",
        "裸",
        "性侵",
        "強姦",
        "自殺",
        "死亡",
        "猝逝",
        "身亡",
        "命案",
        "凶殺",
        "血腥",
        "屍體",
    )
    DEFAULT_EXCLUDE_CATEGORIES = (
        "政治",
        "兩岸",
        "美股雷達",
        "A股港股",
        "財經雲",
        "股市",
        "台股新聞",
        "理財",
        "股票",
        "大陸房市",
        "美股",
        "港股",
        "A股",
        "債券",
        "新股上市",
        "指數",
        "外資",
        "法人",
        "主力",
        "台股盤勢",
        "大陸政經",
        "幣圈",
        "美股預估",
        "ETF",
        "外匯",
    )

    def __init__(self) -> None:
        _load_env_once()

    @staticmethod
    def is_usable_selection(title: str, keyword: str) -> bool:
        title_text = str(title or "").strip()
        keyword_text = str(keyword or "").strip()
        if not title_text or not keyword_text:
            return False
        if title_text.casefold() in {"...", "…", "n/a", "na", "null", "unknown", "untitled"}:
            return False
        return any(character.isalnum() for character in title_text)

    @staticmethod
    def selection_key(title: str, keyword: str) -> str:
        return f"{str(title or '').strip().casefold()}\u001f{str(keyword or '').strip().casefold()}"

    @classmethod
    def is_brand_safe_selection(cls, title: str, keyword: str) -> bool:
        text = f"{title} {keyword}".casefold()
        return not any(term.casefold() in text for term in cls.DEFAULT_EXCLUDE_TITLE_TERMS)

    def is_configured(self) -> bool:
        return all(
            os.getenv(key)
            for key in ("mysql_host", "mysql_port", "mysql_user", "mysql_password", "mysql_db_name")
        )

    def get_random_news(
        self,
        *,
        lookback_days: int = 7,
        limit: int = 200,
        exclude_categories: list[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> NewsSelection | None:
        if not self.is_configured():
            _logger().info("news.fetch.skipped | reason=mysql_not_configured")
            return None

        import pymysql

        categories = list(exclude_categories or self.DEFAULT_EXCLUDE_CATEGORIES)
        date_filter = (datetime.now() - timedelta(days=max(1, int(lookback_days)))).strftime("%Y-%m-%d %H:%M:%S")
        placeholders = ", ".join(["%s"] * len(categories))
        query = f"""
            SELECT title, keyword, category, created_at
            FROM news_ch.news
            WHERE category NOT IN ({placeholders})
              AND COALESCE(keyword, '') != ''
              AND created_at >= %s
            ORDER BY id DESC
            LIMIT %s
        """
        params: list[Any] = [*categories, date_filter, max(1, int(limit))]

        _logger().info("news.fetch.start | lookback_days=%s | limit=%s", lookback_days, limit)
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
                cursor.execute(query, params)
                rows = list(cursor.fetchall() or [])
        finally:
            connection.close()

        rows = [
            row
            for row in rows
            if self.is_usable_selection(row.get("title", ""), row.get("keyword", ""))
            and self.is_brand_safe_selection(row.get("title", ""), row.get("keyword", ""))
        ]
        excluded = {str(key) for key in (exclude_keys or set()) if str(key).strip()}
        if excluded:
            rows = [
                row
                for row in rows
                if self.selection_key(row.get("title", ""), row.get("keyword", "")) not in excluded
            ]
        if not rows:
            reason = "no_unseen_news_rows" if excluded else "no_usable_news_rows"
            _logger().info("news.fetch.empty | reason=%s", reason)
            return None
        selected = random.choice(rows)
        _logger().info("news.fetch.title | %s", str(selected.get("title") or "").strip())
        _logger().info(
            "news.fetch.selected | title=%s | keyword=%s | category=%s",
            str(selected.get("title") or "").strip(),
            str(selected.get("keyword") or "").strip(),
            str(selected.get("category") or "").strip(),
        )
        return NewsSelection(
            title=str(selected.get("title") or "").strip(),
            keyword=str(selected.get("keyword") or "").strip(),
            category=str(selected.get("category") or "").strip(),
            created_at=str(selected.get("created_at") or "").strip(),
        )


class CharacterGroupSelectionError(RuntimeError):
    """Raised when a configured character group cannot produce a character."""


@dataclass(frozen=True, slots=True)
class CharacterGroupCandidate:
    name: str
    role_description: str = ""
    keywords: str = ""
    status: int = 1
    weight: float = 1.0
    group_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role_description": self.role_description,
            "keywords": self.keywords,
            "status": self.status,
            "weight": self.weight,
            **({"group_name": self.group_name} if self.group_name else {}),
        }


@dataclass(frozen=True, slots=True)
class CharacterGroupSelection:
    group_name: str
    selected_character: str
    candidates: tuple[CharacterGroupCandidate, ...]
    selection_source: str = "group_weighted_random"

    @property
    def selected_profile(self) -> dict[str, str]:
        for candidate in self.candidates:
            if candidate.name == self.selected_character:
                return {
                    "role_description": candidate.role_description,
                    "keywords": candidate.keywords,
                }
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "group",
            "subject_mode": "single",
            "group_name": self.group_name,
            "selected_character": self.selected_character,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_profile": self.selected_profile,
            "selection_source": self.selection_source,
        }


@dataclass(frozen=True, slots=True)
class CharacterPairSelection:
    """Select two subject slots from one configured candidate pool.

    Sampling is intentionally with replacement.  ``is_same_group`` controls
    only the candidate scope; it does not impose a distinct-name constraint.
    """

    group_name: str
    is_same_group: bool
    selected_subjects: tuple[CharacterGroupCandidate, CharacterGroupCandidate]
    candidates: tuple[CharacterGroupCandidate, ...]
    selection_source: str = "pair_weighted_random_with_replacement"

    @staticmethod
    def _subject_payload(candidate: CharacterGroupCandidate, role: str, group_name: str) -> dict[str, Any]:
        profile = {
            "role_description": candidate.role_description,
            "keywords": candidate.keywords,
        }
        return {
            "role": role,
            "name": candidate.name,
            "group_name": candidate.group_name or group_name,
            "status": candidate.status,
            "weight": candidate.weight,
            "profile": profile,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "two_character_interaction",
            "subject_mode": "two_character_interaction",
            "group_name": self.group_name,
            "is_same_group": self.is_same_group,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "subjects": [
                self._subject_payload(self.selected_subjects[0], "primary", self.group_name),
                self._subject_payload(self.selected_subjects[1], "secondary", self.group_name),
            ],
            "selection_source": self.selection_source,
        }


class CharacterGroupSelectionService:
    """Select active, positively weighted roles from ``anime.anime_roles``."""

    _QUERY = """
        SELECT role_name_en, role_description, keywords, status, weight
        FROM anime.anime_roles
        WHERE group_name = %s
          AND status = 1
          AND COALESCE(weight, 0) > 0
          AND COALESCE(role_name_en, '') != ''
        ORDER BY id
    """
    _GLOBAL_QUERY = """
        SELECT group_name, role_name_en, role_description, keywords, status, weight
        FROM anime.anime_roles
        WHERE status = 1
          AND COALESCE(weight, 0) > 0
          AND COALESCE(role_name_en, '') != ''
        ORDER BY id
    """

    def __init__(self) -> None:
        _load_env_once()

    @staticmethod
    def is_configured() -> bool:
        return all(
            os.getenv(key)
            for key in ("mysql_host", "mysql_port", "mysql_user", "mysql_password", "mysql_db_name")
        )

    def get_candidates(self, group_name: str) -> tuple[CharacterGroupCandidate, ...]:
        normalized_group = str(group_name or "").strip()
        if not normalized_group:
            raise CharacterGroupSelectionError("Character group_name cannot be empty")
        if not self.is_configured():
            raise CharacterGroupSelectionError(
                f"Character group '{normalized_group}' requires configured MySQL connection settings"
            )

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
                cursor.execute(self._QUERY, [normalized_group])
                rows = list(cursor.fetchall() or [])
        finally:
            connection.close()

        return self._parse_candidates(rows)

    @staticmethod
    def _parse_candidates(rows: list[dict[str, Any]]) -> tuple[CharacterGroupCandidate, ...]:
        candidates: list[CharacterGroupCandidate] = []
        for row in rows:
            name = str(row.get("role_name_en") or "").strip()
            try:
                weight = float(row.get("weight") or 0)
            except (TypeError, ValueError):
                weight = 0.0
            try:
                status = int(row.get("status") or 0)
            except (TypeError, ValueError):
                status = 0
            if not name or status != 1 or weight <= 0:
                continue
            candidates.append(
                CharacterGroupCandidate(
                    name=name,
                    role_description=str(row.get("role_description") or "").strip()[:1200],
                    keywords=str(row.get("keywords") or "").strip()[:500],
                    status=status,
                    weight=weight,
                    group_name=str(row.get("group_name") or "").strip(),
                )
            )
        return tuple(candidates)

    def get_all_candidates(self) -> tuple[CharacterGroupCandidate, ...]:
        """Return every active, positively weighted role across all groups."""

        if not self.is_configured():
            raise CharacterGroupSelectionError(
                "Global character selection requires configured MySQL connection settings"
            )

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
                cursor.execute(self._GLOBAL_QUERY, [])
                rows = list(cursor.fetchall() or [])
        finally:
            connection.close()
        return self._parse_candidates(rows)

    @staticmethod
    def _weighted_choice(
        candidates: tuple[CharacterGroupCandidate, ...],
        *,
        rng: random.Random | None = None,
    ) -> CharacterGroupCandidate:
        chooser = rng or random
        total_weight = sum(candidate.weight for candidate in candidates)
        threshold = chooser.uniform(0, total_weight)
        cumulative = 0.0
        selected = candidates[-1]
        for candidate in candidates:
            cumulative += candidate.weight
            if threshold <= cumulative:
                selected = candidate
                break
        return selected

    def select_random_character(
        self,
        group_name: str,
        *,
        rng: random.Random | None = None,
    ) -> CharacterGroupSelection:
        normalized_group = str(group_name or "").strip()
        candidates = self.get_candidates(normalized_group)
        if not candidates:
            raise CharacterGroupSelectionError(
                f"Character group '{normalized_group}' has no active role with weight > 0"
            )
        selected = self._weighted_choice(candidates, rng=rng)
        return CharacterGroupSelection(
            group_name=normalized_group,
            selected_character=selected.name,
            candidates=candidates,
        )

    def select_pair(
        self,
        group_name: str | None,
        *,
        is_same_group: bool,
        rng: random.Random | None = None,
    ) -> CharacterPairSelection:
        """Select two subject slots without imposing distinct character names."""

        normalized_group = str(group_name or "").strip() if is_same_group else ""
        if is_same_group:
            if not normalized_group:
                raise CharacterGroupSelectionError(
                    "two_character_interaction with is_same_group=true requires group_name"
                )
            candidates = self.get_candidates(normalized_group)
        else:
            candidates = self.get_all_candidates()
        if not candidates:
            scope = f"group '{normalized_group}'" if normalized_group else "the global active pool"
            raise CharacterGroupSelectionError(
                f"two_character_interaction {scope} has no active role with weight > 0"
            )
        selected = (
            self._weighted_choice(candidates, rng=rng),
            self._weighted_choice(candidates, rng=rng),
        )
        return CharacterPairSelection(
            group_name=normalized_group,
            is_same_group=is_same_group,
            selected_subjects=selected,
            candidates=candidates,
        )


@dataclass(slots=True)
class HumanReviewDecision:
    status: str
    selected_paths: list[str]
    edited_text: str
    reviewer: str = ""
    review_mode: str = "auto"
    session_id: str = ""
    session_path: str = ""
    fallback_reason: str = ""
    delivery: dict[str, Any] | None = None
    edit_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_paths": self.selected_paths,
            "edited_text": self.edited_text,
            "reviewer": self.reviewer,
            "review_mode": self.review_mode,
            "session_id": self.session_id,
            "session_path": self.session_path,
            "fallback_reason": self.fallback_reason,
            "delivery": dict(self.delivery or {}),
            "edit_requested": self.edit_requested,
        }


class DiscordHumanReviewService:
    def __init__(self, output_root: Path) -> None:
        _load_env_once()
        self.output_root = output_root
        self.review_root = output_root / "review_sessions"
        self.review_root.mkdir(parents=True, exist_ok=True)

    def is_configured(self) -> bool:
        token = os.getenv("discord_review_bot_token")
        channel_id = os.getenv("discord_review_channel_id")
        return bool(token and channel_id)

    def _filter_media_paths(self, media_paths: list[str]) -> list[str]:
        allowed_roots = [self.output_root.resolve()]
        for raw_root in str(os.getenv("discord_review_allowed_roots") or "").split(","):
            if raw_root.strip():
                allowed_roots.append(Path(raw_root.strip()).expanduser().resolve())
        filtered: list[str] = []
        for raw_path in media_paths:
            try:
                resolved = Path(raw_path).resolve(strict=True)
                if not any(
                    resolved == root or root in resolved.parents
                    for root in allowed_roots
                ):
                    raise ValueError("outside allowed media root")
                if not resolved.is_file() or resolved.suffix.lower() not in REVIEW_MEDIA_EXTENSIONS:
                    raise ValueError("unsupported review media type")
                if resolved.stat().st_size > MAX_REVIEW_FILE_BYTES:
                    raise ValueError("review media file is too large")
            except (OSError, RuntimeError, ValueError) as exc:
                _logger().warning("discord.review.file_rejected | path=%s | reason=%s", raw_path, exc)
                continue
            filtered.append(str(resolved))
        return list(dict.fromkeys(filtered))

    def review_candidates(
        self,
        *,
        text: str,
        media_paths: list[str],
        timeout_seconds: int = 3600,
        allow_asset_selection: bool = True,
        allow_text_edit: bool = True,
        selection_mode: str = "multi",
        selection_required: bool = False,
        selection_limit: int | None = None,
        review_scope: str = "",
    ) -> HumanReviewDecision:
        filtered_media_paths = self._filter_media_paths(media_paths)
        normalized_selection_limit = None
        if selection_limit is not None:
            normalized_selection_limit = max(1, int(selection_limit))
            if selection_mode == "single":
                normalized_selection_limit = 1
        if not filtered_media_paths or not self.is_configured():
            _logger().info(
                "discord.review.skipped | configured=%s | candidate_count=%s",
                self.is_configured(),
                len(filtered_media_paths),
            )
            return HumanReviewDecision(
                status="skipped",
                selected_paths=filtered_media_paths,
                edited_text=text,
                review_mode="auto",
                fallback_reason="discord review is not configured or no candidate files were found",
            )

        _logger().info("discord.review.start | candidate_count=%s | timeout_seconds=%s", len(filtered_media_paths), timeout_seconds)
        discord_text = _fit_discord_message(
            text,
            max_length=1900,
            allow_asset_selection=allow_asset_selection,
            review_scope=review_scope,
        )
        try:
            decision = asyncio.run(
                _run_discord_file_feedback_process(
                    token=str(os.getenv("discord_review_bot_token")),
                    channel_id=int(str(os.getenv("discord_review_channel_id"))),
                    text=discord_text,
                    filepaths=filtered_media_paths,
                    timeout=float(timeout_seconds),
                    allow_asset_selection=allow_asset_selection,
                    allow_text_edit=allow_text_edit,
                    selection_mode=selection_mode,
                    selection_required=selection_required,
                    selection_limit=normalized_selection_limit,
                )
            )
        except Exception as exc:
            _logger().exception("discord.review.error | error=%s", exc)
            decision = ("error", None, text, None, {"status": "error", "error": str(exc)})
        status, reviewer, edited_text, selected_indices, delivery = decision
        if not isinstance(delivery, dict):
            delivery = {}
        if edited_text is None:
            edited_text = text
        selected_paths = filtered_media_paths
        fallback_reason = ""
        review_mode = "discord"
        normalized_status = "approved"
        if status in {"accept", "edit"}:
            if isinstance(selected_indices, list) and selected_indices:
                selected_paths = [
                    filtered_media_paths[index]
                    for index in selected_indices
                    if isinstance(index, int) and 0 <= index < len(filtered_media_paths)
                ]
        elif status == "reject":
            normalized_status = "rejected"
            selected_paths = []
        else:
            fallback_reason = {
                "timeout": "discord review timed out before any decision was received",
                "channel_unavailable": "discord review channel could not be resolved",
                "error": "discord review failed before a decision was received",
            }.get(str(status), "discord review did not return a valid human decision")
            normalized_status = "failed"
        session_id = uuid.uuid4().hex
        session_path = self.review_root / f"{session_id}.json"
        payload = {
            "created_at": datetime.now().isoformat(),
            "text": text,
            "discord_text": discord_text,
            "edited_text": edited_text,
            "media_paths": filtered_media_paths,
            "selected_paths": selected_paths,
            "status": status,
            "reviewer": reviewer or "",
            "review_mode": review_mode,
            "normalized_status": normalized_status,
            "fallback_reason": fallback_reason,
            "delivery": delivery,
        }
        session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _logger().info(
            "discord.review.end | status=%s | raw_status=%s | selected_count=%s | reviewer=%s | delivery=%s | message_id=%s | session=%s",
            normalized_status,
            str(status),
            len(selected_paths),
            str(reviewer or ""),
            str(delivery.get("status") or "unknown"),
            str(delivery.get("message_id") or ""),
            session_id,
        )
        return HumanReviewDecision(
            status=normalized_status,
            selected_paths=selected_paths,
            edited_text=str(edited_text),
            reviewer=str(reviewer or ""),
            review_mode=review_mode,
            session_id=session_id,
            session_path=str(session_path),
            fallback_reason=fallback_reason,
            delivery=delivery,
            edit_requested=str(status) == "edit" or bool(delivery.get("edit_requested")),
        )


class DiscordRunNotificationService:
    """Send a concise operational result without opening a review interaction."""

    def __init__(self) -> None:
        _load_env_once()

    def is_configured(self) -> bool:
        return bool(os.getenv("discord_review_bot_token") and os.getenv("discord_review_channel_id"))

    def notify(self, text: str, media_paths: list[str] | None = None) -> dict[str, Any]:
        if not self.is_configured():
            return {"status": "skipped", "reason": "discord notification is not configured"}
        try:
            return asyncio.run(
                _run_discord_status_notification(
                    token=str(os.getenv("discord_review_bot_token")),
                    channel_id=int(str(os.getenv("discord_review_channel_id"))),
                    text=str(text or "").strip()[:1900],
                    media_paths=media_paths or [],
                )
            )
        except Exception as exc:
            _logger().exception("discord.notification.error | error=%s", exc)
            return {"status": "error", "error": str(exc)}


def _fit_discord_message(
    text: str,
    *,
    max_length: int = 1900,
    allow_asset_selection: bool = True,
    review_scope: str = "",
) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_length:
        return normalized

    lines = normalized.splitlines()
    compact_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        compact = " ".join(stripped.split())
        if compact.lower().startswith("goal:"):
            compact = "Goal: " + textwrap.shorten(compact[5:].strip(), width=280, placeholder="...")
        elif compact.lower().startswith("selection limit:"):
            compact = compact
        elif compact.lower().startswith("suggested ranking:"):
            compact = "Suggested picks:"
        elif compact[:2].isdigit() and ". " in compact:
            compact = textwrap.shorten(compact, width=220, placeholder="...")
        else:
            compact = textwrap.shorten(compact, width=180, placeholder="...")
        compact_lines.append(compact)

    compact_text = "\n".join(compact_lines)
    if len(compact_text) <= max_length:
        return compact_text
    return textwrap.shorten(compact_text, width=max_length, placeholder="...")


async def _close_discord_client(client: Any, *, label: str) -> None:
    """Close discord.py and its HTTP session, including login-failure paths."""
    try:
        if not client.is_closed():
            await client.close()
    finally:
        # discord.py normally closes this session from Client.close(). Keep an
        # explicit idempotent close for failures during static_login/start and
        # for library versions that leave a connector alive after cancellation.
        http = getattr(client, "http", None)
        close = getattr(http, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            _logger().exception("discord.%s.http_close.error", label)


async def _run_discord_file_feedback_process(
    *,
    token: str,
    channel_id: int,
    text: str,
    filepaths: list[str],
    timeout: float,
    allow_asset_selection: bool = True,
    allow_text_edit: bool = True,
    selection_mode: str = "multi",
    selection_required: bool = False,
    selection_limit: int | None = None,
) -> tuple[str | None, str | None, str | None, list[int] | None, dict[str, Any]]:
    import discord
    from discord.ext import commands
    from discord.ui import Button, Modal, TextInput, View

    def split_caption_hashtags(value: str) -> tuple[str, str]:
        content: list[str] = []
        hashtags: list[str] = []
        for line in str(value or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith(("strategy:", "workflow:", "stage:", "selection:", "assets:")):
                continue
            if lowered.startswith("caption:"):
                stripped = stripped.split(":", 1)[1].strip()
                if not stripped:
                    continue
            elif lowered.startswith("hashtags:"):
                stripped = stripped.split(":", 1)[1].strip()
                if not stripped:
                    continue
            if stripped.startswith("#"):
                hashtags.append(stripped)
            else:
                content.append(stripped)
        return "\n".join(content).strip(), "\n".join(hashtags).strip()

    class EditModal(Modal):
        def __init__(self, original_content: str):
            super().__init__(title="Edit Caption")
            caption, hashtags = split_caption_hashtags(original_content)
            self.caption = TextInput(
                label="Caption",
                style=discord.TextStyle.paragraph,
                default=caption[:1500],
                max_length=1500,
                required=True,
            )
            self.hashtags = TextInput(
                label="Hashtags",
                style=discord.TextStyle.short,
                default=hashtags[:300],
                max_length=300,
                required=False,
            )
            self.add_item(self.caption)
            self.add_item(self.hashtags)

        async def on_submit(self, interaction: discord.Interaction) -> None:
            await interaction.response.defer()

    class ResponseView(View):
        def __init__(self, files: list[discord.File], content: str, timeout: float) -> None:
            super().__init__(timeout=timeout)
            self.result: str | None = None
            self.user_name: str | None = None
            self.files = files
            self.content = content
            self.selected_files: list[int] = []
            self.message: discord.Message | None = None
            self.edit_done = asyncio.Event()
            self.is_editing = False
            self.edit_requested = False
            self.decision_finalized = False
            self.reviewer_id: str | None = None
            self.state_lock = asyncio.Lock()

            self.allowed_reviewer_ids = {
                item.strip()
                for item in str(os.getenv("discord_review_allowed_user_ids") or "").split(",")
                if item.strip().isdigit()
            }

            if not allow_text_edit:
                edit_button = next(
                    (item for item in self.children if getattr(item, "label", "") == "Edit"),
                    None,
                )
                if edit_button is not None:
                    self.remove_item(edit_button)

            if allow_asset_selection:
                selectable_files = self.files[:25]
                options = [
                    discord.SelectOption(label=f"Asset {index + 1}", value=str(index))
                    for index in range(len(selectable_files))
                ]
                select = discord.ui.Select(
                    placeholder=(
                        "Select exactly one opening frame"
                        if selection_mode == "single"
                        else "Select final/reference assets"
                    ),
                    min_values=1 if selection_required else 0,
                    max_values=(
                        1
                        if selection_mode == "single"
                        else min(len(selectable_files), selection_limit or len(selectable_files))
                    ),
                    options=options,
                )

                async def select_callback(select_interaction: discord.Interaction) -> None:
                    if not await self.authorize(select_interaction):
                        return
                    async with self.state_lock:
                        if self.decision_finalized:
                            await select_interaction.response.send_message(
                                "This review is already complete.", ephemeral=True
                            )
                            return
                        self.selected_files = [int(value) for value in select.values]
                    await select_interaction.response.defer()

                select.callback = select_callback
                self.add_item(select)

        async def authorize(self, interaction: discord.Interaction) -> bool:
            user = interaction.user
            user_id = str(getattr(user, "id", "")).strip()
            allowed = user_id in self.allowed_reviewer_ids
            if not allowed:
                await interaction.response.send_message("You are not authorized for this review.", ephemeral=True)
                return False
            if self.reviewer_id is not None and self.reviewer_id != user_id:
                await interaction.response.send_message(
                    "This review is already being handled by another reviewer.", ephemeral=True
                )
                return False
            self.reviewer_id = user_id
            return True

        @discord.ui.button(label="Accept", style=discord.ButtonStyle.green)
        async def accept(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                if selection_required and not self.selected_files:
                    await interaction.response.send_message(
                        "Select the required asset before Accept.",
                        ephemeral=True,
                    )
                    return
                if selection_mode == "single" and len(self.selected_files) > 1:
                    await interaction.response.send_message(
                        "Select exactly one asset before Accept.",
                        ephemeral=True,
                    )
                    return
                if selection_limit is not None and len(self.selected_files) > selection_limit:
                    await interaction.response.send_message(
                        f"Select no more than {selection_limit} assets before Accept.",
                        ephemeral=True,
                    )
                    return
                self.decision_finalized = True
                self.result = "accept"
                self.user_name = str(interaction.user)
                if not self.selected_files:
                    self.selected_files = list(range(len(self.files)))
                self.edit_done.set()
            await interaction.response.defer()
            self.stop()

        @discord.ui.button(label="Reject", style=discord.ButtonStyle.red)
        async def reject(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                self.decision_finalized = True
                self.result = "reject"
                self.user_name = str(interaction.user)
                self.content = ""
                self.selected_files = []
                self.edit_done.set()
            await interaction.response.defer()
            self.stop()

        @discord.ui.button(label="Edit", style=discord.ButtonStyle.blurple)
        async def edit(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                self.is_editing = True
            modal = EditModal(self.content)
            try:
                await interaction.response.send_modal(modal)
                await modal.wait()
                async with self.state_lock:
                    if self.decision_finalized:
                        return
                    caption = str(modal.caption.value).strip()
                    hashtags = str(modal.hashtags.value).strip()
                    self.content = "\n\n".join(part for part in (caption[:1500], hashtags[:300]) if part)[:1900]
                    self.edit_requested = True
                if self.message is not None:
                    await self.message.edit(
                        content=self.content,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )

                if not allow_asset_selection:
                    async with self.state_lock:
                        if not self.decision_finalized:
                            self.decision_finalized = True
                            self.result = "edit"
                            self.user_name = str(interaction.user)
                    self.stop()
                    return

                await self.edit_done.wait()
            finally:
                self.is_editing = False
            self.stop()

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    completed = asyncio.Event()
    result: tuple[str | None, str | None, str | None, list[int] | None, dict[str, Any]] = (
        None,
        None,
        None,
        None,
        {},
    )

    @bot.event
    async def on_ready() -> None:
        nonlocal result
        try:
            channel = bot.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await bot.fetch_channel(channel_id)
                except Exception:
                    _logger().exception("discord.review.fetch_channel.error | channel_id=%s", channel_id)
                    result = ("channel_unavailable", None, text, None, {"status": "channel_unavailable", "channel_id": channel_id})
                    completed.set()
                    return
            files = [discord.File(path) for path in filepaths]
            view = ResponseView(files, text or "Review request", timeout=timeout)
            message = await channel.send(
                content=text or "Review request",
                files=files,
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            view.message = message
            delivery = {
                "status": "sent",
                "message_id": str(getattr(message, "id", "") or ""),
                "channel_id": str(channel_id),
                "sent_at": datetime.now().isoformat(),
                "attachment_count": len(filepaths),
                "attachment_names": [Path(path).name for path in filepaths],
                "edit_requested": False,
            }
            _logger().info(
                "discord.review.delivered | channel_id=%s | message_id=%s | attachment_count=%s",
                channel_id,
                str(delivery.get("message_id") or ""),
                len(filepaths),
            )
            try:
                await asyncio.wait_for(view.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                await message.edit(
                    content=f"{text}\n\nTimed out.",
                    view=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                delivery["status"] = "sent_timeout"
                result = ("timeout", None, text, None, delivery)
            else:
                delivery["status"] = "decision_received" if view.result else "sent_no_decision"
                delivery["edit_requested"] = bool(view.edit_requested)
                result = (view.result, view.user_name, view.content, view.selected_files, delivery)
        except Exception:
            _logger().exception("discord.review.on_ready.error")
            result = ("error", None, text, None, {"status": "error", "channel_id": channel_id})
        finally:
            completed.set()

    task = asyncio.create_task(bot.start(token))
    completed_wait = asyncio.create_task(completed.wait())
    try:
        done, _pending = await asyncio.wait(
            {task, completed_wait},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if completed_wait in done:
            await completed_wait
        elif task in done:
            try:
                await task
            except Exception:
                _logger().exception("discord.review.bot.start.error")
            if not completed.is_set():
                result = ("error", None, text, None, {"status": "error", "reason": "bot_stopped_before_review"})
                completed.set()
        else:
            result = ("timeout", None, text, None, {"status": "timeout", "reason": "review_process_timeout"})
            completed.set()
    finally:
        if not completed_wait.done():
            completed_wait.cancel()
            try:
                await completed_wait
            except asyncio.CancelledError:
                pass
        await _close_discord_client(bot, label="review")
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        else:
            try:
                await task
            except Exception:
                _logger().exception("discord.review.bot.start.error")
    return result


async def _run_discord_status_notification(
    *,
    token: str,
    channel_id: int,
    text: str,
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    import discord

    intents = discord.Intents.none()
    client = discord.Client(intents=intents)
    result: dict[str, Any] = {"status": "error", "channel_id": str(channel_id)}

    @client.event
    async def on_ready() -> None:
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            files = [discord.File(path) for path in (media_paths or []) if Path(path).exists()]
            message = await channel.send(
                content=text or "MediaOverload run completed.",
                files=files,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            result.update(
                {
                    "status": "sent",
                    "message_id": str(getattr(message, "id", "") or ""),
                    "channel_id": str(channel_id),
                    "attachment_count": len(files),
                    "sent_at": datetime.now().isoformat(),
                }
            )
            _logger().info(
                "discord.notification.delivered | channel_id=%s | message_id=%s",
                channel_id,
                str(result.get("message_id") or ""),
            )
        except Exception as exc:
            result.update({"status": "error", "error": str(exc)})
            _logger().exception("discord.notification.on_ready.error")
        finally:
            await _close_discord_client(client, label="notification")

    try:
        await asyncio.wait_for(client.start(token), timeout=30.0)
    except asyncio.TimeoutError:
        result.update({"status": "timeout", "reason": "notification_timeout"})
    finally:
        await _close_discord_client(client, label="notification")
    return result
