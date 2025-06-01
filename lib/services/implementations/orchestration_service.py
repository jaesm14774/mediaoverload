"""協調服務實現 - 新的 ContentProcessor"""
import os
import re
import time
import shutil
import numpy as np
from typing import Dict, Any, Optional
from lib.services.interfaces.orchestration_service import IOrchestrationService
from lib.media_auto.character_config import BaseCharacter
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.repositories.character_repository import ICharacterRepository
from lib.social_media import MediaPost
from utils.logger import setup_logger, log_execution_time


class OrchestrationService(IOrchestrationService):
    """協調服務實現 - 負責協調各個服務完成整個工作流程"""
    
    def __init__(self, character_repository: ICharacterRepository = None):
        self.logger = setup_logger(__name__)
        self.character_repository = character_repository
        
        # 服務依賴
        self.prompt_service = None
        self.content_service = None
        self.review_service = None
        self.publishing_service = None
        self.notification_service = None
    
    def configure_services(self,
                         prompt_service,
                         content_service,
                         review_service,
                         publishing_service,
                         notification_service) -> None:
        """配置所需的服務"""
        self.prompt_service = prompt_service
        self.content_service = content_service
        self.review_service = review_service
        self.publishing_service = publishing_service
        self.notification_service = notification_service
        self.logger.info("所有服務已配置完成")
    
    @log_execution_time(logger=setup_logger(__name__))
    async def execute_workflow(self,
                             character: BaseCharacter,
                             prompt: Optional[str] = None,
                             temperature: float = 1.0) -> Dict[str, Any]:
        """執行完整的工作流程"""
        start_time = time.time()
        config = None
        
        try:
            # 步驟 1: 處理角色選擇
            if character.group_name and self.character_repository:
                workflow_name = os.path.splitext(os.path.basename(character.workflow_path))[0]
                dynamic_character = self.character_repository.get_random_character_from_group(
                    character.group_name, 
                    workflow_name
                )
                if dynamic_character:
                    character.character = dynamic_character.lower()
                    character.config.character = dynamic_character.lower()
                    self.logger.info(f"從群組 {character.group_name} 中選擇角色: {dynamic_character}")
            
            # 步驟 2: 生成提示詞
            if not prompt:
                prompt = self.prompt_service.generate_prompt(
                    character=character.character,
                    method=character.config.generate_prompt_method,
                    temperature=temperature
                )
            
            # 特例處理 waddledee
            if re.sub(string=prompt.lower(), pattern='\s', repl='').find('waddledee') != -1:
                prompt = re.sub(string=prompt, pattern='waddledee|Waddledee', repl='waddle dee')
            
            self.logger.info(f"開始處理提示詞: {prompt}")
            
            # 步驟 3: 準備生成配置
            config_dict = character.get_generation_config(prompt)
            config_dict['output_dir'] = os.path.join(config_dict['output_dir'], config_dict['character'])
            config = GenerationConfig(**config_dict)
            
            # 步驟 4: 生成內容
            content_result = self.content_service.generate_content(config)
            
            if len(content_result['filter_results']) == 0:
                self.logger.warning('沒有任何生成的圖片被 LLM 分析選中！')
                self.cleanup(config_dict['output_dir'])
                return {'status': 'no_images_selected'}
            
            # 步驟 5: 準備審核
            selected_result = (content_result['filter_results'] 
                             if len(content_result['filter_results']) <= 6 
                             else np.random.choice(content_result['filter_results'], size=6, replace=False).tolist())
            
            self.logger.info(f"選擇了 {len(selected_result)} 張圖片進行審核")
            
            # 步驟 6: 審核內容
            review_result = await self.review_service.review_content(
                text=content_result['article_content'],
                image_paths=[row['image_path'] for row in selected_result],
                timeout=3600
            )
            
            approval_result, user, edited_content, selected_indices = review_result
            
            if not selected_indices:
                self.logger.warning('沒有任何生成的圖片被用戶選中！')
                self.cleanup(config_dict['output_dir'])
                return {'status': 'no_images_approved'}
            
            # 步驟 7: 處理圖片
            selected_image_paths = [selected_result[i]['image_path'] for i in selected_indices]
            processed_image_paths = self.publishing_service.process_images(
                selected_image_paths, 
                config_dict['output_dir']
            )
            
            # 步驟 8: 發布到社群媒體
            post = MediaPost(
                images=processed_image_paths,
                caption='',
                hashtags=edited_content,
                additional_params={'share_to_story': True}
            )
            
            publish_results = self.publishing_service.publish_to_social_media(
                post, 
                platforms=['instagram']
            )
            
            # 步驟 9: 發送通知
            if any(publish_results.values()):
                execution_time = time.time() - start_time
                self.notification_service.notify_success(
                    character=config_dict['character'],
                    execution_time=execution_time,
                    image_count=len(selected_indices)
                )
            
            # 步驟 10: 清理資源
            self.cleanup(config_dict['output_dir'])
            
            return {
                'status': 'success',
                'publish_results': publish_results,
                'image_count': len(selected_indices)
            }
            
        except Exception as e:
            self.logger.error(f"工作流程執行失敗: {str(e)}")
            
            # 發送錯誤通知
            if self.notification_service and config:
                self.notification_service.notify_error(
                    character=config.get_all_attributes().get('character', 'unknown'),
                    error_message=str(e)
                )
            
            # 清理資源
            if config:
                self.cleanup(config.get_all_attributes().get('output_dir', ''))
            
            raise
    
    def cleanup(self, output_dir: str) -> None:
        """清理資源"""
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            self.logger.info(f"已清理資源目錄: {output_dir}") 