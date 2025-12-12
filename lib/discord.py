from typing import List, Optional, Tuple
import asyncio
import discord
from discord.ext import commands
from discord.ui import Modal, TextInput, View, Button

import json
import requests

#[Discord Package](https://github.com/Rapptz/discord.py)

class EditModal(Modal):
    def __init__(self, original_content: str):
        super().__init__(title="編輯內容")
        self.content = TextInput(
            label="修改內容",
            style=discord.TextStyle.paragraph,
            default=original_content,
            required=True
        )
        self.add_item(self.content)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()

class ResponseView(View):
    def __init__(self, files: List[discord.File], content: str, timeout=180):
        super().__init__(timeout=timeout)
        self.result = None
        self.user = None
        self.files = files
        self.content = content
        self.selected_files = []  # 預設為非勾選狀態，使用者檢核後才勾選
        self.message = None
        self.edit_done = asyncio.Event()
        self.is_editing = False

    @discord.ui.button(label="接受", style=discord.ButtonStyle.green)
    async def accept(self, interaction: discord.Interaction, button: Button):
        if self.is_editing:
            self.edit_done.set()
        self.result = "accept"
        self.user = interaction.user
        # 如果用戶直接點擊接受且沒有選擇任何項目，則接受所有項目
        if not self.selected_files:
            self.selected_files = list(range(len(self.files)))
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="拒絕", style=discord.ButtonStyle.red)
    async def reject(self, interaction: discord.Interaction, button: Button):
        if self.is_editing:
            self.edit_done.set()
        self.result = "reject"
        # 拒絕時清空編輯內容和選擇的圖片
        self.content = None
        self.selected_files = None
        self.user = interaction.user
        await interaction.response.defer()
        self.stop()

    @discord.ui.button(label="編輯", style=discord.ButtonStyle.blurple)
    async def edit(self, interaction: discord.Interaction, button: Button):
        self.is_editing = True
        
        modal = EditModal(self.content)
        await interaction.response.send_modal(modal)
        await modal.wait()
        
        self.content = modal.content.value
        await self.message.edit(content=self.content)
        
        options = [
            discord.SelectOption(
                label=f"圖片 {i+1}",
                value=str(i),
                default=i in self.selected_files
            )
            for i in range(len(self.files))
        ]
        
        select = discord.ui.Select(
            placeholder="選擇要保留的圖片",
            min_values=0,
            max_values=len(self.files),
            options=options
        )

        async def select_callback(interaction: discord.Interaction):
            self.selected_files = [int(value) for value in select.values]
            selected_files = [self.files[i] for i in self.selected_files]
            await self.message.edit(files=selected_files)
            await interaction.response.defer()

        select.callback = select_callback
        view = View()
        view.add_item(select)
        await interaction.followup.send("請選擇要保留的圖片，完成後請按接受按鈕：", view=view, ephemeral=True)

        await self.edit_done.wait()
        self.result = "edit"
        self.user = interaction.user
        self.stop()

class DiscordFeedbackBot:
    def __init__(self, token: str, channel_id: int):
        self.token = token
        self.channel_id = channel_id
        self.result_data = None
        self._process_completed = asyncio.Event()
        self._cleanup_done = False
        self._main_task = None
        
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.setup_bot()

    def setup_bot(self):
        @self.bot.event
        async def on_ready():
            try:
                print(f"Bot已登入為 {self.bot.user}")
                channel = self.bot.get_channel(self.channel_id)
                if not channel:
                    print("無法獲取指定的頻道，請檢查 CHANNEL_ID")
                    self.result_data = (None, None, None, None)
                    self._process_completed.set()
                    return

                files = [discord.File(file_path) for file_path in self.filepaths]
                
                result, user, edited_content, selected_indices = await self.get_user_response(
                    channel,
                    self.text,
                    files,
                    timeout=self.timeout
                )
                
                self.result_data = (result, user, edited_content, selected_indices)
                
            except Exception as e:
                print(f"發生錯誤: {e}")
                self.result_data = (None, None, None, None)
            finally:
                self._process_completed.set()

    async def safe_cleanup(self):
        """安全的清理所有資源"""
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        try:
            if not self.bot.is_closed():
                await self.bot.close()

            if self._main_task and not self._main_task.done():
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            print(f"清理錯誤: {str(e)}")

    async def get_user_response(
        self,
        channel: discord.TextChannel,
        content: str,
        files: List[discord.File],
        timeout: float = 60.0
    ) -> Tuple[Optional[str], Optional[discord.User], Optional[str], Optional[List[int]]]:
        view = ResponseView(files, content, timeout=timeout)
        message = await channel.send(content=content, files=files, view=view)
        view.message = message
        
        try:
            await asyncio.wait_for(view.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print("等待用戶回應逾時，強制中止。")
            await message.edit(content=f"{content}\n\n**操作已逾時。**", view=None)
            view.stop()
            return None, None, None, None
        
        return (
            view.result,
            view.user,
            view.content,
            view.selected_files
        )

    async def run_with_timeout(self, text: str, filepaths: List[str], timeout: float = 60.0):
        self.text = text if text else 'default message'
        self.filepaths = filepaths if isinstance(filepaths, list) else [filepaths]
        self.timeout = timeout

        async def main_task():
            await self.bot.start(self.token)
            
        try:
            self._main_task = asyncio.create_task(main_task())
            
            try:
                await asyncio.wait_for(self._process_completed.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"整體程序超時 ({timeout} 秒)")
            
            return self.result_data
            
        finally:
            await self.safe_cleanup()

async def run_discord_file_feedback_process(token: str, channel_id: int, text: str, filepaths, timeout: float = 60.0):
    """執行 Discord 檔案回饋程序"""
    bot = None
    try:
        bot = DiscordFeedbackBot(token, channel_id)
        return await bot.run_with_timeout(text, filepaths, timeout)
    except Exception as e:
        print(f"執行過程發生錯誤: {str(e)}")
        return None, None, None, None
    finally:
        if bot:
            await bot.safe_cleanup()

class DiscordNotify:
    webhook_url=''
    
    def notify(self, msg, file_path=None):
        if file_path:
            with open(file_path, 'rb') as file:
                files = {
                    'payload_json': (None, json.dumps({"content": msg})),
                    'image.png': file
                }
                response = requests.post(self.webhook_url, files=files, timeout=30)
        else:
            data = {"content": msg}
            response = requests.post(self.webhook_url, data=data, timeout=30)
        
        if response.status_code in [200, 201, 202, 203, 204]:
            print(f"{'File' if file_path else 'Message'} sent successfully.")
        else:
            print(f"Failed to send the {'file' if file_path else 'message'}.")
            print(response.text)
            