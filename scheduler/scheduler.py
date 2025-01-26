import yaml
import schedule
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Dict, Any
import logging
import os

sys.path.append(str(Path(__file__).parent.parent))

from lib.media_auto.media_main_logic import ContentProcessor
from lib.content_generation.image_content_generator import VisionManagerBuilder
import lib.media_auto.process as media_process

class MediaScheduler:
    def __init__(self, config_path: str = "scheduler/schedule_config.yaml"):
        # 設置日誌
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)  # 確保 logs 資料夾存在

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 設置日誌檔案名稱
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file_path = logs_dir / f"{today}.log"

        file_handler = logging.FileHandler(self.log_file_path)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.config_path = config_path
        self.load_config()
        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llama3.2-vision') \
            .with_text_model('ollama', model_name='phi4') \
            .build()
        
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def generate_prompt(self, prompt_config: Dict[str, Any]) -> str:
        # 檢查是否使用 LLM 生成
        if prompt_config.get('use_llm', True):
            character = prompt_config.get('character', '')
            
            # 使用 VisionContentManager 生成提示詞
            print(f'Start generating any prompt for character : {character}.')
            prompt = self.ollama_vision_manager.generate_arbitrary_input(
                character=character
            )
            print('Generate arbitrary input: ', prompt)
            return prompt
        
        # 使用模板生成
        template = prompt_config['template']
        variables = prompt_config.get('variables', {})
        
        # 替換變數
        for key, values in variables.items():
            template = template.replace(f"{{{key}}}", random.choice(values))
            
        return template
    
    def process_character(self, character_name: str, prompt_config: dict):
        self.logger.info(f"開始處理角色 {character_name}")
        try:
            # 獲取角色類別
            character_class = getattr(media_process, f'{character_name}Process')
            processor = ContentProcessor(character_class())
            
            # 生成提示詞
            prompt = self.generate_prompt(prompt_config)
            self.logger.info(f"生成的提示詞: {prompt}")
            
            # 執行處理
            import asyncio
            asyncio.run(processor.etl_process(prompt=prompt))
            
            self.logger.info(f"成功處理角色 {character_name}")
        except Exception as e:
            self.logger.error(f"處理角色 {character_name} 時發生錯誤: {str(e)}", exc_info=True)
    
    def setup_schedules(self):
        self.logger.info("開始設置排程...")
        for char_name, char_config in self.config['schedules'].items():
            try:
                # 檢查時間格式
                scheduled_time = datetime.strptime(char_config['time'], '%H:%M').time()
                
                # 計算下次執行時間
                now = datetime.now()
                next_run = datetime.combine(now.date(), scheduled_time)
                if next_run < now:
                    next_run += timedelta(days=1)
                
                schedule.every().day.at(char_config['time']).do(
                    self.process_character,
                    character_name=char_config['character'],
                    prompt_config=char_config['prompts'][0]
                )
                
                self.logger.info(
                    f"已排程角色 {char_config['character']} 在 {char_config['time']} 執行, "
                    f"下次執行時間: {next_run}"
                )
            except Exception as e:
                self.logger.error(f"設置排程 {char_name} 時發生錯誤: {str(e)}")
    
    def run(self):
        self.logger.info("啟動排程器...")
        self.setup_schedules()
        self.logger.info("所有排程已設置完成，進入主迴圈")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"執行排程時發生錯誤: {str(e)}", exc_info=True)

if __name__ == "__main__":
    scheduler = MediaScheduler()
    scheduler.run() 