import os
import re
import numpy as np
from dotenv import load_dotenv
from typing import Dict, Any
from lib.media_auto.process import BaseCharacter
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory

from lib.social_media import MediaPost
from lib.discord import run_discord_file_feedback_process
from utils.image import ImageProcessor

load_dotenv(f'media_overload.env')

class ContentProcessor:
    """內容處理器主類"""
    
    def __init__(self, character_class: BaseCharacter):
        self.character_class = character_class
        
    async def etl_process(self, prompt: str) -> Dict[str, Any]:
        """執行完整的 ETL 處理流程"""
        # 獲取角色特定的生成配置
        config_dict = self.character_class.get_generation_config(prompt)
        
        # 構建配置
        config = GenerationConfig(
            output_dir=os.path.join(config_dict['output_dir'], config_dict['character']),
            character=config_dict['character'],
            prompt=config_dict['prompt'],
            generation_type=config_dict['type'],
            workflow_path=config_dict['workflow_path'],
            additional_params=config_dict['additional_params']
        )
        
        # 獲取對應的策略
        self.strategy = StrategyFactory.get_strategy(config.generation_type)
        # 讀取config
        self.strategy.load_config(config)
        
        # 生成內容
        self.strategy.generate_description() \
            .generate_image() \
            .analyze_image_text_match(config_dict.get('similarity_threshold', 0.9)) \
            .generate_article_content()

        if len(self.strategy.filter_results) == 0:
            print('Any generated image is not selected by llm analysis!')
            print('All done')
            self.cleanup(config)
            return 
        
        #超過6張只傳 6張即可，不然會過量
        self.selected_result = self.strategy.filter_results if len(self.strategy.filter_results) <= 6 else np.random.choice(self.strategy.filter_results,size=6, replace=False).tolist()
        
        #審核內容 傳到discord
        result = await run_discord_file_feedback_process(
            token=os.environ['discord_review_bot_token'],
            channel_id=int(os.environ['discord_review_channel_id']),
            text=self.strategy.article_content,
            filepaths=[row['image_path'] for row in self.selected_result],
            timeout=3600
        )
        
        # 解構結果
        result, user, edited_content, selected_indices = result
        
        print(f"結果: {result}")
        print(f"操作用戶: {user}")
        print(f"編輯後內容: {edited_content}")
        print(f"選擇的圖片索引: {selected_indices}")    

        if not selected_indices:
            print('Any generated image is not selected by user!')
            print('All done')
            self.cleanup(config)
            return 

        #處理IG 模組 只能上傳jpg format
        selected_image_paths = [self.selected_result[i]['image_path'] for i in selected_indices]
        image_process=ImageProcessor(config.output_dir)
        image_process.main_process()
        selected_image_paths = [re.sub(string=image_path, pattern=r'\.png|\.jpeg', repl='.jpg') for image_path in selected_image_paths]
        
        # 添加社群媒體上傳功能
        post = MediaPost(
            images=selected_image_paths,
            caption='',
            hashtags=edited_content,
            additional_params={'share_to_story': True}
        )
        is_successful_upload = self.character_class.upload_to_social_media(post, platforms=['instagram'])
        if is_successful_upload:
            print('社群媒體任務上傳成功')
            print('All done')

        self.cleanup(config)

    def cleanup(self, config):
        import shutil
        if os.path.exists(config.output_dir):
            # 使用shutil.rmtree遞迴刪除目錄及其所有內容
            shutil.rmtree(config.output_dir)
            print(f"已清理資源目錄: {config.output_dir}")
