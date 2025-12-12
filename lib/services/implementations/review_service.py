"""審核服務實現"""
from typing import List, Tuple, Optional
from lib.services.interfaces.review_service import IReviewService
from lib.discord import run_discord_file_feedback_process
from utils.logger import setup_logger


class ReviewService(IReviewService):
    """審核服務實現
    
    透過 Discord 進行人工審核內容和媒體檔案。
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.discord_token = None
        self.discord_channel_id = None
    
    async def review_content(self,
                           text: str,
                           media_paths: List[str],
                           timeout: int = 4000) -> Tuple[bool, str, Optional[str], Optional[List[int]]]:
        """審核內容
        
        Args:
            text: 審核文字內容
            media_paths: 媒體檔案路徑列表
            timeout: 超時時間（秒）
            
        Returns:
            (審核結果, 用戶, 編輯後內容, 選擇的索引列表)
        """
        if not self.discord_token or not self.discord_channel_id:
            raise ValueError("Discord 設定未配置，請先調用 configure_discord")
        
        self.logger.info(f"開始 Discord 審核流程，待審核媒體數量: {len(media_paths)}")
        
        result = await run_discord_file_feedback_process(
            token=self.discord_token,
            channel_id=self.discord_channel_id,
            text=text,
            filepaths=media_paths,
            timeout=timeout
        )
        
        approval_result, user, edited_content, selected_indices = result
        
        if edited_content is None:
            edited_content = ''
        if selected_indices is None:
            selected_indices = []
        
        self.logger.info(f"Discord 審核結果: {approval_result}, 選擇數量: {len(selected_indices)}")
        
        return approval_result, user, edited_content, selected_indices
    
    def configure_discord(self, token: str, channel_id: int) -> None:
        """配置 Discord 設定
        
        Args:
            token: Discord bot token
            channel_id: 頻道 ID
        """
        self.discord_token = token
        self.discord_channel_id = channel_id
        self.logger.info(f"Discord 設定已配置，頻道 ID: {channel_id}") 