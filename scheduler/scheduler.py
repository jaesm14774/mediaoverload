import yaml
import schedule
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
import sys
import math
from typing import Dict, Any, Optional
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

from lib.services.service_factory import ServiceFactory
from lib.media_auto.character_base import ConfigurableCharacterWithSocialMedia
from utils.logger import setup_logger, log_execution_time

class MediaScheduler:
    def __init__(self, config_path: str = "scheduler/schedule_config.yaml"):
        self.logger = setup_logger(__name__)
        self.config_path = config_path
        self.load_config()
        
        self.last_execution_times = {}  # 儲存每個角色的上次執行時間
        self.service_factory = ServiceFactory()
        self.orchestration_service = self.service_factory.get_orchestration_service()

        # 從配置文件讀取或使用默認值
        probability_settings = self.config.get('probability_settings', {})
        self.base_probability = float(probability_settings.get('base_probability', 4/15))    # 基礎機率（每15小時4次）
        self.max_probability = float(probability_settings.get('max_probability', 1/2))    # 最大機率（每15小時7.5次）
        self.min_probability = float(probability_settings.get('min_probability', 0.5/15))     # 最小機率（每15小時0.5次）

        # 設置定期清理閒置連接的任務
        schedule.every(30).minutes.do(self.cleanup_idle_connections)

        # 立即執行一次處理
        asyncio.run(self.instant_execution())

    @log_execution_time(logger=setup_logger(__name__))
    def load_config(self):
        """載入配置文件"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def calculate_time_factor(self, hours_since_last: float) -> float:
        """計算基於時間的機率調整因子"""
        if hours_since_last <= 0:
            return 0.1

        steepness1 = 4
        steepness2 = 1
        midpoint1 = 4
        midpoint2 = 7

        decline = 1 / (1 + math.exp(steepness1 * (hours_since_last - midpoint1)))
        recovery = 1 / (1 + math.exp(-steepness2 * (hours_since_last - midpoint2)))

        factor = 1 - decline + recovery
        return max(0.1, factor)

    def should_execute(self, character_name: str) -> bool:
        """決定是否要執行處理"""
        # 如果是角色的首次執行（通常在程式啟動時），則強制執行
        if character_name not in self.last_execution_times:
            self.logger.info(f"角色 {character_name} 首次執行，強制執行。")
            return True
            
        current_hour = datetime.now().hour
        if current_hour >= 22 or current_hour < 7:
            self.logger.info("當前時間在睡眠時段（22:00-07:00），跳過執行")
            return False

        char_config = next((cfg for name, cfg in self.config['schedules'].items() 
                          if cfg['character'] == character_name), None)
        if not char_config:
            return False

        prob_settings = char_config.get('probability_settings', {})
        base_prob = float(prob_settings.get('base_probability', self.base_probability))
        max_prob = float(prob_settings.get('max_probability', self.max_probability))
        min_prob = float(prob_settings.get('min_probability', self.min_probability))

        last_execution_time = self.last_execution_times.get(character_name)
        if last_execution_time:
            hours_since_last = (datetime.now() - last_execution_time).total_seconds() / 3600
            time_factor = self.calculate_time_factor(hours_since_last)
            current_probability = base_prob * time_factor
            current_probability = max(min_prob, min(current_probability, max_prob))
            
            self.logger.info(f"距離上次執行 {hours_since_last:.1f} 小時, "
                           f"時間因子 {time_factor:.2f}, "
                           f"當前執行機率 {current_probability:.3%}/小時")
        else:
            # 此分支在新的邏輯下理論上不會被走到，因為首次執行會在上一個if條件中返回True
            current_probability = base_prob
            self.logger.info(f"首次執行，使用基礎機率 {current_probability:.3%}/小時")

        return random.random() < current_probability

    async def process_character(self, character_name: str, prompt_config: dict) -> None:
        """處理特定角色的內容"""
        if not self.should_execute(character_name):
            self.logger.info(f"根據機率決定跳過執行角色 {character_name}")
            return

        self.logger.info(f"開始處理角色 {character_name}")
        try:
            # 載入角色配置
            character_config_path = f"configs/characters/{character_name.lower()}.yaml"
            character = ConfigurableCharacterWithSocialMedia(character_config_path)

            # 註冊社群媒體平台
            publishing_service = self.service_factory.get_publishing_service()
            
            # 如果使用配置檔案，從配置中註冊平台
            if hasattr(character, '_social_media_config'):
                for platform_name, platform_config in character._social_media_config.items():
                    publishing_service.register_platform(platform_name, platform_config)

            
            # 執行工作流程
            result = await self.orchestration_service.execute_workflow(
                character=character,
                prompt=prompt_config.get('prompt'),
                temperature=prompt_config.get('temperature', 1.0)
            )
            
            self.logger.info(f"成功處理角色 {character_name}")
            self.last_execution_times[character_name] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"處理角色 {character_name} 時發生錯誤: {str(e)}", exc_info=True)

    async def instant_execution(self) -> None:
        """立即執行所有角色的處理"""
        self.logger.info("開始執行首次啟動任務...")
        for char_config in self.config['schedules'].values():
            await self.process_character(
                character_name=char_config['character'],
                prompt_config=char_config['prompts'][0]
            )

    def setup_schedules(self) -> None:
        """設置排程"""
        self.logger.info("開始設置排程...")
        
        for char_config in self.config['schedules'].values():
            try:
                def schedule_next_run(char_config):
                    now = datetime.now()
                    next_hour = now + timedelta(hours=1)
                    next_hour = next_hour.replace(
                        minute=random.randint(0, 59),
                        second=0,
                        microsecond=0
                    )
                    
                    if next_hour.hour >= 22 or next_hour.hour < 7:
                        next_hour = next_hour.replace(hour=7, minute=random.randint(0, 59))
                        if next_hour < now:
                            next_hour += timedelta(days=1)
                    
                    schedule.every().day.at(f"{next_hour.strftime('%H:%M')}").do(
                        self.process_and_reschedule,
                        char_config=char_config,
                        schedule_func=schedule_next_run
                    ).tag(f"schedule_{char_config['character']}")
                    
                    self.logger.info(
                        f"已為角色 {char_config['character']} 設置下次執行時間: {next_hour.strftime('%H:%M')}"
                    )
                
                schedule_next_run(char_config)
                
            except Exception as e:
                self.logger.error(f"設置排程 {char_config['character']} 時發生錯誤: {str(e)}")

    def process_and_reschedule(self, char_config: dict, schedule_func) -> None:
        """處理角色並重新排程"""
        try:
            schedule.clear(f"schedule_{char_config['character']}")
            
            asyncio.run(self.process_character(
                character_name=char_config['character'],
                prompt_config=char_config['prompts'][0]
            ))
            
            schedule_func(char_config)
            
        except Exception as e:
            self.logger.error(f"處理和重新排程時發生錯誤: {str(e)}", exc_info=True)
            schedule_func(char_config)

    def cleanup_idle_connections(self):
        """清理閒置的資料庫連接"""
        try:
            from lib.database import db_pool
            db_pool.cleanup_idle_connections(max_idle_time=1800)  # 30分鐘未使用的連接會被清理
            self.logger.info("已清理閒置的資料庫連接")
        except Exception as e:
            self.logger.error(f"清理閒置連接時發生錯誤: {str(e)}")

    def run(self) -> None:
        """運行排程器"""
        self.logger.info("啟動排程器...")
        self.setup_schedules()
        self.logger.info("所有排程已設置完成，進入主迴圈")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"執行排程時發生錯誤: {str(e)}", exc_info=True)
                # 發生錯誤時，嘗試清理並重新建立連接
                self.cleanup_idle_connections()

    def cleanup(self) -> None:
        """清理資源"""
        self.service_factory.cleanup()

if __name__ == "__main__":
    scheduler = MediaScheduler()
    try:
        scheduler.run()
    finally:
        scheduler.cleanup()