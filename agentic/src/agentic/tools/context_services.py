from __future__ import annotations

import asyncio
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

        if not rows:
            _logger().info("news.fetch.empty")
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

    def review_candidates(
        self,
        *,
        text: str,
        media_paths: list[str],
        timeout_seconds: int = 3600,
    ) -> HumanReviewDecision:
        filtered_media_paths = [str(Path(path)) for path in media_paths if Path(path).exists()]
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
        discord_text = _fit_discord_message(text, max_length=1900)
        try:
            decision = asyncio.run(
                _run_discord_file_feedback_process(
                    token=str(os.getenv("discord_review_bot_token")),
                    channel_id=int(str(os.getenv("discord_review_channel_id"))),
                    text=discord_text,
                    filepaths=filtered_media_paths,
                    timeout=float(timeout_seconds),
                )
            )
        except Exception as exc:
            _logger().exception("discord.review.error | error=%s", exc)
            decision = ("error", None, text, None)
        status, reviewer, edited_text, selected_indices = decision
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
        }
        session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _logger().info(
            "discord.review.end | status=%s | raw_status=%s | selected_count=%s | reviewer=%s | session=%s",
            normalized_status,
            str(status),
            len(selected_paths),
            str(reviewer or ""),
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
        )


def _fit_discord_message(text: str, *, max_length: int = 1900) -> str:
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

    compact_lines.append("Use Accept to keep shown assets, Edit to choose assets, Reject to stop.")
    compact_text = "\n".join(compact_lines)
    if len(compact_text) <= max_length:
        return compact_text
    return textwrap.shorten(compact_text, width=max_length, placeholder="...")


async def _run_discord_file_feedback_process(
    *,
    token: str,
    channel_id: int,
    text: str,
    filepaths: list[str],
    timeout: float,
) -> tuple[str | None, str | None, str | None, list[int] | None]:
    import discord
    from discord.ext import commands
    from discord.ui import Button, Modal, TextInput, View

    class EditModal(Modal):
        def __init__(self, original_content: str):
            super().__init__(title="Edit Review Text")
            self.content = TextInput(
                label="Updated text",
                style=discord.TextStyle.paragraph,
                default=original_content,
                required=True,
            )
            self.add_item(self.content)

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

        @discord.ui.button(label="Accept", style=discord.ButtonStyle.green)
        async def accept(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if self.is_editing:
                self.edit_done.set()
            self.result = "accept"
            self.user_name = str(interaction.user)
            if not self.selected_files:
                self.selected_files = list(range(len(self.files)))
            await interaction.response.defer()
            self.stop()

        @discord.ui.button(label="Reject", style=discord.ButtonStyle.red)
        async def reject(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if self.is_editing:
                self.edit_done.set()
            self.result = "reject"
            self.user_name = str(interaction.user)
            self.content = ""
            self.selected_files = []
            await interaction.response.defer()
            self.stop()

        @discord.ui.button(label="Edit", style=discord.ButtonStyle.blurple)
        async def edit(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            self.is_editing = True
            modal = EditModal(self.content)
            await interaction.response.send_modal(modal)
            await modal.wait()
            self.content = str(modal.content.value)
            if self.message is not None:
                await self.message.edit(content=self.content)

            options = [
                discord.SelectOption(
                    label=f"Asset {index + 1}",
                    value=str(index),
                    default=index in self.selected_files,
                )
                for index in range(len(self.files))
            ]
            select = discord.ui.Select(
                placeholder="Select assets to keep",
                min_values=0,
                max_values=len(self.files),
                options=options,
            )

            async def select_callback(select_interaction: discord.Interaction) -> None:
                self.selected_files = [int(value) for value in select.values]
                await select_interaction.response.defer()

            select.callback = select_callback
            view = View()
            view.add_item(select)
            await interaction.followup.send(
                "Select the assets to keep, then click Accept.",
                view=view,
                ephemeral=True,
            )
            await self.edit_done.wait()
            self.result = "edit"
            self.user_name = str(interaction.user)
            self.stop()

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    completed = asyncio.Event()
    result: tuple[str | None, str | None, str | None, list[int] | None] = (None, None, None, None)

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
                    result = ("channel_unavailable", None, text, None)
                    completed.set()
                    return
            files = [discord.File(path) for path in filepaths]
            view = ResponseView(files, text or "Review request", timeout=timeout)
            message = await channel.send(content=text or "Review request", files=files, view=view)
            view.message = message
            try:
                await asyncio.wait_for(view.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                await message.edit(content=f"{text}\n\nTimed out.", view=None)
                result = ("timeout", None, text, None)
            else:
                result = (view.result, view.user_name, view.content, view.selected_files)
        except Exception:
            _logger().exception("discord.review.on_ready.error")
            result = ("error", None, text, None)
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
                result = ("error", None, text, None)
                completed.set()
        else:
            result = ("timeout", None, text, None)
            completed.set()
    finally:
        if not completed_wait.done():
            completed_wait.cancel()
            try:
                await completed_wait
            except asyncio.CancelledError:
                pass
        if not bot.is_closed():
            await bot.close()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    return result
