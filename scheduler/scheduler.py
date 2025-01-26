import yaml
import schedule
import time
import random
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional, Dict, Any
sys.path.append(str(Path(__file__).parent.parent))

from lib.media_auto.media_main_logic import ContentProcessor
from lib.content_generation.image_content_generator import VisionManagerBuilder
import lib.media_auto.process as media_process

class MediaScheduler:
    def __init__(self, config_path: str = "scheduler/schedule_config.yaml"):
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
            prompt = self.ollama_vision_manager.generate_arbitrary_input(
                character=character
            )
            return prompt
        
        # 使用模板生成
        template = prompt_config['template']
        variables = prompt_config.get('variables', {})
        
        # 替換變數
        for key, values in variables.items():
            template = template.replace(f"{{{key}}}", random.choice(values))
            
        return template
    
    def process_character(self, character_name: str, prompt_config: dict):
        try:
            # 獲取角色類別
            character_class = getattr(media_process, f'{character_name}Process')
            processor = ContentProcessor(character_class())
            
            # 生成提示詞
            prompt = self.generate_prompt(prompt_config)
            
            # 執行處理
            import asyncio
            asyncio.run(processor.etl_process(prompt=prompt))
            
            print(f"[{datetime.now()}] Successfully processed {character_name} with prompt: {prompt}")
        except Exception as e:
            print(f"[{datetime.now()}] Error processing {character_name}: {str(e)}")
    
    def setup_schedules(self):
        for char_name, char_config in self.config['schedules'].items():
            # 設置定時任務，使用 time 而不是 cron
            schedule.every().day.at(char_config['time']).do(
                self.process_character,
                character_name=char_config['character'],
                prompt_config=char_config['prompts'][0]
            )
            print(f"[{datetime.now()}] Scheduled {char_config['character']} at {char_config['time']}")
    
    def run(self):
        print(f"[{datetime.now()}] Starting scheduler...")
        self.setup_schedules()
        print(f"[{datetime.now()}] All schedules set up, entering main loop")
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分鐘檢查一次

if __name__ == "__main__":
    scheduler = MediaScheduler()
    scheduler.run() 