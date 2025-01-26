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
from lib.discord import run_discord_file_feedback_process
from utils.image import ImageProcessor

load_dotenv(f'media_overload.env')

class ContentProcessor:
    """內容處理器主類"""
    
    def __init__(self, character_class: BaseCharacter):
        self.character_class = character_class
        self.logger = setup_logger(__name__)
        
    @log_execution_time(logger=setup_logger(__name__))
    async def etl_process(self, prompt: str) -> Dict[str, Any]:
        """執行完整的 ETL 處理流程"""
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

        self.cleanup(config)

    @log_execution_time(logger=setup_logger(__name__))
    def cleanup(self, config):
        import shutil
        if os.path.exists(config.output_dir):
            # 使用shutil.rmtree遞迴刪除目錄及其所有內容
            shutil.rmtree(config.output_dir)
            self.logger.info(f"已清理資源目錄: {config.output_dir}")
