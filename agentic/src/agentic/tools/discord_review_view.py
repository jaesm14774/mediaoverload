"""Discord review controls, separate from bot transport and workflow execution."""
from __future__ import annotations

import asyncio
import os

import discord
from discord.ui import Button, Modal, TextInput, View


def create_discord_review_view(
    files: list[discord.File],
    content: str,
    timeout: float,
    *,
    allow_asset_selection: bool = True,
    allow_text_edit: bool = True,
    selection_mode: str = "multi",
    selection_required: bool = False,
    selection_limit: int | None = None,
    allow_regenerate: bool = False,
    regenerate_current_segment: bool = False,
    allow_series_lifecycle_choice: bool = False,
) -> View:
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
        def __init__(
            self,
            files: list[discord.File],
            content: str,
            timeout: float,
            *,
            allow_regenerate: bool,
            regenerate_current_segment: bool,
            allow_series_lifecycle_choice: bool,
        ) -> None:
            super().__init__(timeout=timeout)
            self.result: str | None = None
            self.user_name: str | None = None
            self.files = files
            self.content = content
            self.selected_files: list[int] = []
            self.message: discord.Message | None = None
            self.edit_requested = False
            self.series_disposition = ""
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

            if not allow_regenerate:
                continue_button = next(
                    (item for item in self.children if getattr(item, "label", "") == "重新生成"),
                    None,
                )
                if continue_button is not None:
                    self.remove_item(continue_button)
            elif regenerate_current_segment:
                continue_button = next(
                    (item for item in self.children if getattr(item, "label", "") == "重新生成"),
                    None,
                )
                if continue_button is not None:
                    continue_button.label = "重新生成本段"

            if not allow_series_lifecycle_choice:
                for label in ("完成系列並發布本集", "廢棄系列"):
                    button = next(
                        (item for item in self.children if getattr(item, "label", "") == label),
                        None,
                    )
                    if button is not None:
                        self.remove_item(button)

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

        async def close_controls(self, *, content: str | None = None) -> None:
            try:
                if self.message is not None:
                    changes: dict[str, object] = {
                        "view": None,
                        "allowed_mentions": discord.AllowedMentions.none(),
                    }
                    if content is not None:
                        changes["content"] = content
                    await self.message.edit(**changes)
            finally:
                self.stop()

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
                await interaction.response.defer()
                self.decision_finalized = True
                self.result = "accept"
                self.user_name = str(interaction.user)
                if not self.selected_files:
                    self.selected_files = list(range(len(self.files)))
            await self.close_controls()

        @discord.ui.button(label="Reject", style=discord.ButtonStyle.red)
        async def reject(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                await interaction.response.defer()
                self.decision_finalized = True
                self.result = "reject"
                self.user_name = str(interaction.user)
                self.content = ""
                self.selected_files = []
            await self.close_controls()

        @discord.ui.button(label="Edit", style=discord.ButtonStyle.blurple)
        async def edit(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
            modal = EditModal(self.content)
            await interaction.response.send_modal(modal)
            timed_out = await modal.wait()
            if timed_out:
                return
            async with self.state_lock:
                if self.decision_finalized:
                    return
                caption = str(modal.caption.value).strip()
                hashtags = str(modal.hashtags.value).strip()
                previous_content = self.content
                previous_edit_requested = self.edit_requested
                self.content = "\n\n".join(part for part in (caption[:1500], hashtags[:300]) if part)[:1900]
                self.edit_requested = True
                try:
                    if self.message is not None:
                        await self.message.edit(
                            content=self.content,
                            allowed_mentions=discord.AllowedMentions.none(),
                        )
                except Exception:
                    self.content = previous_content
                    self.edit_requested = previous_edit_requested
                    raise

        @discord.ui.button(label="重新生成", style=discord.ButtonStyle.blurple)
        async def regenerate_run(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                self.decision_finalized = True
                self.result = "regenerate"
                self.user_name = str(interaction.user)
                self.selected_files = []
                await interaction.response.defer()
            await self.close_controls(
                content=(
                    f"{self.content[:1700]}\n\n已選擇重新生成本段；前面已接受的分段會保留，生成後會重新審核本段。"
                    if regenerate_current_segment
                    else f"{self.content[:1750]}\n\n已選擇重新生成，將從頭重新跑完整流程。"
                )
            )

        @discord.ui.button(label="完成系列並發布本集", style=discord.ButtonStyle.blurple)
        async def complete_series(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                await interaction.response.defer()
                self.decision_finalized = True
                self.result = "accept"
                self.series_disposition = "completed"
                self.user_name = str(interaction.user)
                self.selected_files = list(range(len(self.files)))
            await self.close_controls()

        @discord.ui.button(label="廢棄系列", style=discord.ButtonStyle.red)
        async def discard_series(self, interaction: discord.Interaction, button: Button) -> None:
            del button
            if not await self.authorize(interaction):
                return
            async with self.state_lock:
                if self.decision_finalized:
                    await interaction.response.send_message("This review is already complete.", ephemeral=True)
                    return
                await interaction.response.defer()
                self.decision_finalized = True
                self.result = "reject"
                self.series_disposition = "discarded"
                self.user_name = str(interaction.user)
                self.content = ""
                self.selected_files = []
            await self.close_controls()

    return ResponseView(
        files=files,
        content=content,
        timeout=timeout,
        allow_regenerate=allow_regenerate,
        regenerate_current_segment=regenerate_current_segment,
        allow_series_lifecycle_choice=allow_series_lifecycle_choice,
    )
