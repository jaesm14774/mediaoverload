import os
import re
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any
from lib.media_auto.process import BaseCharacter
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from utils.logger import setup_logger, log_execution_time
from lib.social_media import MediaPost
from lib.discord import run_discord_file_feedback_process, DiscordNotify
from utils.image import ImageProcessor
import time
from datetime import datetime

load_dotenv(f'media_overload.env')
load_dotenv(f'configs/social_media/discord/Discord.env')

class ContentProcessor:
    """內容處理器主類"""
    
    def __init__(self, character_class: BaseCharacter):
        self.character_class = character_class
        self.logger = setup_logger(__name__)
        self.start_time = time.time()
        
    @log_execution_time(logger=setup_logger(__name__))
    async def etl_process(self, prompt: str) -> Dict[str, Any]:
        """執行完整的 ETL 處理流程"""
        try:
            #特例處理
            if re.sub(string=prompt.lower(), pattern='\s', repl='').find('waddledee') != -1:
                prompt = re.sub(string=prompt, pattern='waddledee|Waddledee', repl='waddle dee')
            
            self.logger.info(f"開始處理提示詞: {prompt}")
            
            # 獲取角色特定的生成配置
            config_dict = self.character_class.get_generation_config(prompt)
            self.logger.info(f"獲取到生成配置: {config_dict['character']}")
            
            # 構建配置
            config = GenerationConfig(
                output_dir=os.path.join(config_dict['output_dir'], config_dict['character']),
                character=config_dict['character'],
                prompt=config_dict['prompt'],
                generation_type=config_dict['type'],
                workflow_path=config_dict['workflow_path'],
                default_hashtags=config_dict.get('default_hashtags', []),
                additional_params=config_dict['additional_params']
            )
            self.logger.info(f"配置構建完成，輸出目錄: {config.output_dir}")
            
            # 獲取對應的策略
            self.strategy = StrategyFactory.get_strategy(config.generation_type)
            self.logger.info(f"使用策略: {config.generation_type}")
            
            # 讀取config
            self.strategy.load_config(config)
            self.logger.info("策略配置載入完成")
            
            # 生成內容
            self.logger.info("開始生成內容流程")
            self.strategy.generate_description()
            self.logger.info("描述生成完成")
            
            self.strategy.generate_image()
            self.logger.info("圖片生成完成")
            
            self.strategy.analyze_image_text_match(config_dict.get('similarity_threshold', 0.9))
            self.logger.info("圖文匹配分析完成")
            
            self.strategy.generate_article_content()
            self.logger.info("文章內容生成完成")

            if len(self.strategy.filter_results) == 0:
                self.logger.warning('沒有任何生成的圖片被 LLM 分析選中！')
                self.logger.info('處理完成')
                self.cleanup(config)
                return 
            
            #超過6張只傳 6張即可，不然會過量
            self.selected_result = self.strategy.filter_results if len(self.strategy.filter_results) <= 6 else np.random.choice(self.strategy.filter_results,size=6, replace=False).tolist()
            self.logger.info(f"選擇了 {len(self.selected_result)} 張圖片進行審核")
            
            #審核內容 傳到discord
            self.logger.info("開始 Discord 審核流程")
            result = await run_discord_file_feedback_process(
                token=os.environ['discord_review_bot_token'],
                channel_id=int(os.environ['discord_review_channel_id']),
                text=self.strategy.article_content,
                filepaths=[row['image_path'] for row in self.selected_result],
                timeout=3600
            )
            
            # 解構結果
            result, user, edited_content, selected_indices = result
            if edited_content is None:
                edited_content=''
            if selected_indices is None:
                selected_indices=[]
            self.logger.info(f"Discord 審核結果: {result}")
            self.logger.info(f"審核用戶: {user}")
            self.logger.info(f"編輯後內容長度: {len(edited_content)}")
            self.logger.info(f"選擇的圖片數量: {len(selected_indices)}")    

            if not selected_indices:
                self.logger.warning('沒有任何生成的圖片被用戶選中！')
                self.logger.info('處理完成')
                self.cleanup(config)
                return 

            #處理IG 模組 只能上傳jpg format
            selected_image_paths = [self.selected_result[i]['image_path'] for i in selected_indices]
            self.logger.info("開始圖片處理")
            image_process=ImageProcessor(config.output_dir)
            image_process.main_process()
            selected_image_paths = [re.sub(string=image_path, pattern=r'\.png|\.jpeg', repl='.jpg') for image_path in selected_image_paths]
            self.logger.info("圖片處理完成")
            
            # 添加社群媒體上傳功能
            self.logger.info("開始社群媒體上傳")
            post = MediaPost(
                images=selected_image_paths,
                caption='',
                hashtags=edited_content,
                additional_params={'share_to_story': True}
            )
            is_successful_upload = self.character_class.upload_to_social_media(post, platforms=['instagram'])
            if is_successful_upload:
                self.logger.info('社群媒體上傳成功')
                self.logger.info('所有流程完成')
                
                # 計算總執行時間
                end_time = time.time()
                execution_time = end_time - self.start_time
                hours = int(execution_time // 3600)
                minutes = int((execution_time % 3600) // 60)
                seconds = int(execution_time % 60)
                
                # 發送 Discord 完成通知
                discord_notify = DiscordNotify()
                discord_notify.webhook_url = os.environ['程式執行狀態']
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                success_message = (
                    f"✨ 任務完成通知 ✨\n"
                    f"🎭 角色: {config_dict['character']}\n"
                    f"⏰ 完成時間: {current_time}\n"
                    f"⌛ 總執行時間: {hours}小時 {minutes}分鐘 {seconds}秒\n"
                    f"📸 成功上傳圖片數量: {len(selected_indices)}張"
                )
                
                discord_notify.notify(success_message)

            self.cleanup(config)
        except Exception as e:
            self.logger.error(f"ETL 流程執行失敗: {str(e)}")
            
            # 發送 Discord 錯誤通知
            discord_notify = DiscordNotify()
            discord_notify.webhook_url = os.environ['程式bug權杖']
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            error_message = (
                f"❌ 錯誤通知 ❌\n"
                f"🎭 角色: {config_dict['character']}\n"
                f"⏰ 錯誤時間: {current_time}\n"
                f"💥 錯誤訊息: {str(e)}\n"
            )
            
            discord_notify.notify(error_message)
            raise

    @log_execution_time(logger=setup_logger(__name__))
    def cleanup(self, config):
        import shutil
        if os.path.exists(config.output_dir):
            # 使用shutil.rmtree遞迴刪除目錄及其所有內容
            shutil.rmtree(config.output_dir)
            self.logger.info(f"已清理資源目錄: {config.output_dir}")
