import yaml
import schedule
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
import sys
import math
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from lib.media_auto.media_main_logic import ContentProcessor
from lib.content_generation.image_content_generator import VisionManagerBuilder
import lib.media_auto.process as media_process
from utils.logger import setup_logger, log_execution_time

class MediaScheduler:
    def __init__(self, config_path: str = "scheduler/schedule_config.yaml"):
        self.logger = setup_logger(__name__)
        self.config_path = config_path
        self.load_config()
        
        self.last_execution_time = None

        # 基礎設定調整
        self.base_probability = 2.5/15    # 基礎機率（每15小時2.5次）
        self.max_probability = 3.0/15    # 最大機率（每15小時3次）
        self.min_probability = 0.5/15     # 最小機率（每15小時0.5次）
        
        # 機率調整參數
        self.recovery_hours = 6.0         # 機率顯著回升時間點
        self.steep_decline_hours = 2.0    # 急遽下降期
        self.decline_steepness = 2.0      # 下降曲線的陡峭程度
        self.recovery_steepness = 0.5     # 回升曲線的陡峭程度
        
        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llama3.2-vision') \
            .with_text_model('ollama', model_name='phi4') \
            .build()

    @log_execution_time(logger=setup_logger(__name__))
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @log_execution_time(logger=setup_logger(__name__))
    def generate_prompt(self, prompt_config: Dict[str, Any]) -> str:
        # 檢查是否使用 LLM 生成
        if prompt_config.get('use_llm', True):
            character = prompt_config.get('character', '')
            
            # 使用 VisionContentManager 生成提示詞
            self.logger.info(f'開始為角色生成提示詞: {character}')
            prompt = self.ollama_vision_manager.generate_arbitrary_input(
                character=character
            )
            self.logger.info(f'生成的提示詞: {prompt}')
            return prompt
        
        # 使用模板生成
        template = prompt_config['template']
        variables = prompt_config.get('variables', {})
        
        # 替換變數
        for key, values in variables.items():
            template = template.replace(f"{{{key}}}", random.choice(values))
            
        return template

    def calculate_time_factor(self, hours_since_last: float) -> float:
        """
        計算基於時間的機率調整因子
        使用組合的 sigmoid 函數實現更精確的機率控制
        """
        if hours_since_last <= 0:
            return 0.1  # 最低機率因子
        
        # 分段式 sigmoid 函數，實現急遽下降後緩慢回升
        def steep_decline(x: float) -> float:
            return 1 / (1 + math.exp(self.decline_steepness * (x - self.steep_decline_hours)))
            
        def gradual_recovery(x: float) -> float:
            return 1 / (1 + math.exp(-self.recovery_steepness * (x - self.recovery_hours)))
        
        # 計算下降和回升的綜合因子
        decline_factor = steep_decline(hours_since_last)
        recovery_factor = gradual_recovery(hours_since_last)
        
        # 組合兩個因子，確保在發文後立即降到低點，然後緩慢回升
        combined_factor = 0.1 + (0.9 * (1 - decline_factor) * recovery_factor)
        
        # 確保因子在合理範圍內
        return max(0.1, min(combined_factor, 1.0))

    @log_execution_time(logger=setup_logger(__name__))
    def should_execute(self) -> bool:
        """決定是否要執行處理，使用改進的機率分布"""
        # 檢查睡眠時段 (22:00-07:00)
        current_hour = datetime.now().hour
        if current_hour >= 22 or current_hour < 7:
            self.logger.info("當前時間在睡眠時段（22:00-07:00），跳過執行")
            return False
        
        # 計算距離上次執行的時間
        if self.last_execution_time:
            hours_since_last = (datetime.now() - self.last_execution_time).total_seconds() / 3600
            time_factor = self.calculate_time_factor(hours_since_last)
            
            # 計算當前小時的基礎機率
            current_probability = self.base_probability * time_factor
            
            # 確保機率在合理範圍內
            current_probability = max(self.min_probability, 
                                   min(current_probability, self.max_probability))
            
            self.logger.info(f"距離上次執行 {hours_since_last:.1f} 小時, "
                           f"時間因子 {time_factor:.2f}, "
                           f"當前執行機率 {current_probability:.3%}/小時")
        else:
            # 首次執行使用基礎機率
            current_probability = self.base_probability
            self.logger.info(f"首次執行，使用基礎機率 {current_probability:.3%}/小時")
        
        # 每分鐘的執行機率
        minute_probability = current_probability / 60
        return random.random() < minute_probability
    
    @log_execution_time(logger=setup_logger(__name__))
    def process_character(self, character_name: str, prompt_config: dict):
        self.logger.info(f"觸發角色 {character_name} 的處理")
        
        # 根據機率決定是否執行
        if not self.should_execute():
            self.logger.info(f"根據機率決定跳過執行角色 {character_name}")
            return
            
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
            self.last_execution_time = datetime.now()  # 更新上次執行時間
        except Exception as e:
            self.logger.error(f"處理角色 {character_name} 時發生錯誤: {str(e)}", exc_info=True)
    
    def get_random_minute(self) -> str:
        """生成隨機分鐘數（0-59）"""
        return f"{random.randint(0, 59):02d}"
    
    @log_execution_time(logger=setup_logger(__name__))
    def setup_schedules(self):
        self.logger.info("開始設置排程...")
        
        # 為每個角色設置每小時執行一次
        for char_name, char_config in self.config['schedules'].items():
            try:
                # 每小時執行一次，每次執行時重新設置下一個小時的隨機時間
                def schedule_next_run(char_config):
                    # 取得當前時間
                    now = datetime.now()
                    # 計算下一個小時
                    next_hour = now + timedelta(hours=1)
                    next_hour = next_hour.replace(minute=random.randint(0, 59), second=0, microsecond=0)
                    
                    # 如果下一個時間在睡眠時段（22:00-07:00），則調整到早上7點
                    if next_hour.hour >= 22 or next_hour.hour < 7:
                        next_hour = next_hour.replace(hour=7, minute=random.randint(0, 59))
                        if next_hour < now:  # 如果調整後的時間比現在早，就加一天
                            next_hour += timedelta(days=1)
                    
                    # 設置下一次執行
                    schedule.every().day.at(f"{next_hour.strftime('%H:%M')}").do(
                        self.process_and_reschedule,
                        char_config=char_config,
                        schedule_func=schedule_next_run
                    ).tag(f"schedule_{char_config['character']}")
                    
                    self.logger.info(
                        f"已為角色 {char_config['character']} 設置下次執行時間: {next_hour.strftime('%H:%M')}, "
                    )
                
                # 初始設置
                schedule_next_run(char_config)
                
            except Exception as e:
                self.logger.error(f"設置排程 {char_name} 時發生錯誤: {str(e)}")
    
    def process_and_reschedule(self, char_config: dict, schedule_func):
        """處理角色並重新排程下一次執行"""
        try:
            # 清除當前角色的所有排程
            schedule.clear(f"schedule_{char_config['character']}")
            
            # 執行處理
            self.process_character(
                character_name=char_config['character'],
                prompt_config=char_config['prompts'][0]
            )
            
            # 設置下一次執行
            schedule_func(char_config)
            
        except Exception as e:
            self.logger.error(f"處理和重新排程時發生錯誤: {str(e)}", exc_info=True)
            # 發生錯誤時也要確保重新排程
            schedule_func(char_config)
    
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