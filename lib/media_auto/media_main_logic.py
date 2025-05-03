import os
import re
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from lib.media_auto.process import BaseCharacter
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from utils.logger import setup_logger, log_execution_time
from lib.social_media import MediaPost
from lib.discord import run_discord_file_feedback_process, DiscordNotify
from utils.image import ImageProcessor
import time
from datetime import datetime
from lib.database import db_pool
from lib.content_generation.image_content_generator import VisionManagerBuilder

load_dotenv(f'media_overload.env')
load_dotenv(f'configs/social_media/discord/Discord.env')

class ContentProcessor:
    """內容處理器主類"""
    
    def __init__(self, character_class: BaseCharacter):
        self.character_class = character_class
        self.logger = setup_logger(__name__)
        self.start_time = time.time()
        self.now_date = datetime.now().strftime('%Y-%m-%d')

        db_pool.initialize('mysql',
                        host=os.environ['mysql_host'],
                        port=int(os.environ['mysql_port']),
                        user=os.environ['mysql_user'],
                        password=os.environ['mysql_password'],
                        db_name=os.environ['mysql_db_name'])
        
        self.mysql_conn = db_pool.get_connection('mysql')

    def get_random_character_from_group(self) -> str:
        """從群組中隨機獲取一個角色"""
        cursor = self.mysql_conn.cursor
        query = """
            SELECT role_name_en 
            FROM anime.anime_roles
            WHERE group_name = %s AND status = 1 AND workflow_name = %s AND weight > 0
        """.strip()
        filename = os.path.basename(self.character_class.workflow_path)
        name_without_ext, _ = os.path.splitext(filename)
        cursor.execute(query, (self.character_class.group_name, name_without_ext))
        results = cursor.fetchall()  # 獲取所有結果
        characters = [row[0] for row in results]  # 提取角色名稱
        if not characters:
            return self.character_class.character
        return random.choice(characters)

    @log_execution_time(logger=setup_logger(__name__))
    def generate_prompt(self, character: str, temperature: float = 1.0) -> str:
        """生成提示詞
        
        Args:
            character (str): 角色名稱
            temperature (float, optional): LLM 溫度參數. Defaults to 1.0.
            
        Returns:
            str: 生成的提示詞
        """
        self.logger.info(f'開始為角色生成提示詞: {character}')
        # 初始化 VisionManager
        ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llama3.2-vision') \
            .with_text_model('ollama', model_name='qwen3:8b', temperature=temperature) \
            .build()        
        
        self.logger.info(f'角色生成方式採用: {self.character_class.config.generate_prompt_method}')

        if self.character_class.config.generate_prompt_method == 'default':
            prompt = self.generate_prompt_by_default(ollama_vision_manager, character)
        elif self.character_class.config.generate_prompt_method == 'news':
            prompt = self.generate_prompt_by_news(ollama_vision_manager, character)

        self.logger.info(f'生成的提示詞: {prompt}')
        return prompt.lower()

    def generate_prompt_by_default(self, ollama_vision_manager, character):
       
        # 生成第一層提示詞
        prompt = ollama_vision_manager.generate_arbitrary_input(
            character=character,
            extra=f"current time : {datetime.now().strftime('%Y-%m-%d')}",
            prompt_type=self.character_class.config.generate_prompt_method
        )
        prompt = prompt.replace("'", '').replace('"', '')
        # 生成第二層提示詞
        prompt = ollama_vision_manager.generate_arbitrary_input(
            character=character,
            extra=f'Act as a Chaos Narrative Forge: roll an imaginary 12-sided die to pick a genre, flip a coin for narrative voice, and draw a tarot card for emotional color. Retain at most **one** micro-detail (≤ 5 words) from "{prompt}" as a hidden Easter egg—do NOT mention which. Everything else must be explosively original: collide clashing eras, graft surreal physics, twist clichés inside-out, and end on an unforeseen pivot. Output 1-3 lush, sensory sentences rich in metaphor and motion, with no meta commentary or game mechanics exposed.'
        )
        return prompt

    def generate_prompt_by_news(self, ollama_vision_manager, character):
        engine = self.mysql_conn.engine
        df = pd.read_sql_query(f"select title, keyword, created_at, category from news_ch.news order by id desc limit 10000", engine)
        df = df[df.keyword.map(str.strip) != '']
        df = df[~df.category.isin(['政治', '兩岸', '美股雷達', 'A股港股', '財經雲', '股市', '台股新聞'])]
        df = df[df['created_at'] >= self.now_date].reset_index(drop=True)

        choose_index = random.choice(range(0, len(df)))
        keyword = df.loc[choose_index, 'keyword']
        title = df.loc[choose_index, 'title']
        info = f"""extra_info : {title} ; {keyword}""".strip()
        
        # 生成第一層提示詞
        prompt = ollama_vision_manager.generate_arbitrary_input(
            character=character,
            extra=info,
            prompt_type=self.character_class.config.image_system_prompt
        )
        prompt = prompt.replace("'", '').replace('"', '')

        return prompt
            
    @log_execution_time(logger=setup_logger(__name__))
    async def etl_process(self, prompt: Optional[str] = None, temperature: float = 1.0) -> Dict[str, Any]:
        """執行完整的 ETL 處理流程
        
        Args:
            prompt (Optional[str], optional): 提示詞. Defaults to None.
            temperature (float, optional): LLM 溫度參數. Defaults to 1.0.
            
        Returns:
            Dict[str, Any]: 處理結果
        """
        try:
            # 如果角色有群組設定，從群組中隨機選擇一個角色
            if self.character_class.group_name:
                dynamic_character = self.get_random_character_from_group().lower()
                # 將character 強轉新角色
                self.character_class.character = dynamic_character
                self.character_class.config.character = dynamic_character
                self.logger.info(f"從群組 {self.character_class.group_name} 中選擇角色: {dynamic_character}")
            
            # 如果沒有提供 prompt，則生成一個
            if not prompt:
                prompt = self.generate_prompt(
                    character=self.character_class.character,
                    temperature=temperature
                )
            
            #特例處理
            if re.sub(string=prompt.lower(), pattern='\s', repl='').find('waddledee') != -1:
                prompt = re.sub(string=prompt, pattern='waddledee|Waddledee', repl='waddle dee')
            
            self.logger.info(f"開始處理提示詞: {prompt}")
            
            # 獲取角色特定的生成配置
            config = self.character_class.get_generation_config(prompt)
            self.logger.info(f"獲取到生成配置: {config['character']}")
            
            # 直接使用解包運算符傳入所有配置
            config['output_dir'] = os.path.join(config['output_dir'], config['character'])

            self.logger.info(f"配置構建完成，輸出目錄: {config['output_dir']}")
            
            # 獲取對應的策略
            self.strategy = StrategyFactory.get_strategy(config['generation_type'])
            self.logger.info(f"使用策略: {config['generation_type']}")
            
            # 讀取config
            self.strategy.load_config(GenerationConfig(**config))
            self.logger.info("策略配置載入完成")
            
            # 生成內容
            self.logger.info("開始生成內容流程")

            self.strategy.generate_description()
            self.logger.info(f"""描述: {self.strategy.descriptions}""".strip())
            self.logger.info("描述生成完成")
            
            self.strategy.generate_image()
            self.logger.info("圖片生成完成")
            
            self.strategy.analyze_image_text_match(config.get('similarity_threshold', 0.9))
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
            image_process=ImageProcessor(config['output_dir'])
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
                    f"🎭 角色: {config['character']}\n"
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
                f"🎭 角色: {config['character']}\n"
                f"⏰ 錯誤時間: {current_time}\n"
                f"💥 錯誤訊息: {str(e)}\n"
            )
            
            discord_notify.notify(error_message)
            raise

    @log_execution_time(logger=setup_logger(__name__))
    def cleanup(self, config):
        import shutil
        if os.path.exists(config['output_dir']):
            # 使用shutil.rmtree遞迴刪除目錄及其所有內容
            shutil.rmtree(config['output_dir'])
            self.logger.info(f"已清理資源目錄: {config['output_dir']}")
        
        db_pool.close_all()
