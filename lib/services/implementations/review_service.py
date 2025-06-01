"""審核服務實現"""
from typing import List, Tuple, Optional
from lib.services.interfaces.review_service import IReviewService
from lib.discord import run_discord_file_feedback_process
from utils.logger import setup_logger


class ReviewService(IReviewService):
    """審核服務實現"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.discord_token = None
        self.discord_channel_id = None
    
    async def review_content(self,
                           text: str,
                           image_paths: List[str],
                           timeout: int = 3600) -> Tuple[bool, str, Optional[str], Optional[List[int]]]:
        """審核內容"""
        if not self.discord_token or not self.discord_channel_id:
            raise ValueError("Discord 設定未配置，請先調用 configure_discord")
        
        self.logger.info("開始 Discord 審核流程")
        self.logger.info(f"待審核圖片數量: {len(image_paths)}")
        
        # 調用 Discord 審核流程
        result = await run_discord_file_feedback_process(
            token=self.discord_token,
            channel_id=self.discord_channel_id,
            text=text,
            filepaths=image_paths,
            timeout=timeout
        )
        
        # 解構結果
        approval_result, user, edited_content, selected_indices = result
        
        # 處理空值
        if edited_content is None:
            edited_content = ''
        if selected_indices is None:
            selected_indices = []
        
        self.logger.info(f"Discord 審核結果: {approval_result}")
        self.logger.info(f"審核用戶: {user}")
        self.logger.info(f"編輯後內容長度: {len(edited_content)}")
        self.logger.info(f"選擇的圖片數量: {len(selected_indices)}")
        
        return approval_result, user, edited_content, selected_indices
    
    def configure_discord(self, token: str, channel_id: int) -> None:
        """配置 Discord 設定"""
        self.discord_token = token
        self.discord_channel_id = channel_id
        self.logger.info(f"Discord 設定已配置，頻道 ID: {channel_id}") 