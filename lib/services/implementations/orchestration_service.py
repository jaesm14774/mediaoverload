"""協調服務實現"""
import os
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
    """協調服務實現
    
    負責協調各個服務完成整個工作流程：提示詞生成、內容生成、審核、發布。
    """
    
    def __init__(self, character_repository: ICharacterRepository = None):
        """初始化協調服務
        
        Args:
            character_repository: 角色資料庫存取層（可選）
        """
        self.logger = setup_logger(__name__)
        self.character_repository = character_repository
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
        """配置所需的服務
        
        Args:
            prompt_service: 提示詞服務
            content_service: 內容生成服務
            review_service: 審核服務
            publishing_service: 發布服務
            notification_service: 通知服務
        """
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
        """執行完整的工作流程
        
        Args:
            character: 角色配置
            prompt: 提示詞（可選，未提供則自動生成）
            temperature: 生成溫度
            
        Returns:
            包含 status 和相關結果的字典
        """
        start_time = time.time()
        config = None
        
        try:
            # 處理角色選擇
            if character.group_name and self.character_repository:
                generation_type = getattr(character.config, 'generation_type', '')
                is_kirby_group = character.group_name.lower() == 'kirby'
                is_longvideo = generation_type.lower() in ['text2longvideo', 'text2longvideo_firstframe']
                
                if is_kirby_group and is_longvideo:
                    character.character = 'kirby'
                    character.config.character = 'kirby'
                    self.logger.info(f"長影片模式：群組 {character.group_name} 直接使用角色 kirby")
                else:
                    workflow_name = os.path.splitext(os.path.basename(character.workflow_path))[0]
                    dynamic_character = self.character_repository.get_random_character_from_group(
                        character.group_name, 
                        workflow_name
                    )
                    if dynamic_character:
                        character.character = dynamic_character.lower()
                        character.config.character = dynamic_character.lower()
                        self.logger.info(f"從群組 {character.group_name} 中選擇角色: {dynamic_character}")
            
            # 生成提示詞
            if not prompt:
                prompt = self.prompt_service.generate_prompt(
                    character=character.character,
                    method=character.config.generate_prompt_method,
                    temperature=temperature,
                    character_config=character.config
                )
                self.logger.info(f"生成提示詞: {prompt[:100]}..." if len(prompt) > 100 else f"生成提示詞: {prompt}")
            
            # 準備生成配置
            config_dict = character.get_generation_config(prompt)
            config_dict['output_dir'] = os.path.join(config_dict['output_dir'], config_dict['character'])
            config = GenerationConfig(**config_dict)
            
            # 生成內容（第一階段）
            content_result = self.content_service.generate_content(config)
            
            strategy = self.content_service.strategy
            strategy_name = getattr(strategy, 'name', None) or strategy.__class__.__name__
            
            # Discord 人工審核流程
            if not strategy.needs_user_review():
                self.logger.error('策略未實現 needs_user_review()，所有策略都必須進行 Discord 審核')
                self.cleanup(config_dict['output_dir'])
                return {'status': 'strategy_must_review'}
            
            self.logger.info('開始 Discord 人工審核流程')
            review_items = strategy.get_review_items(max_items=10)
            
            if not review_items:
                self.logger.warning('沒有需要審核的項目')
                self.cleanup(config_dict['output_dir'])
                return {'status': 'no_media_selected'}
            
            default_review_text = f'[策略: {strategy_name}] 請選擇要使用的圖片'
            review_text = content_result.get('article_content') or default_review_text
            
            review_result = await self.review_service.review_content(
                text=review_text,
                media_paths=[item['media_path'] for item in review_items],
                timeout=4000
            )
            
            approval_result, user, edited_content, selected_indices = review_result
            
            if not selected_indices:
                self.logger.warning('沒有任何項目被用戶選中')
                self.cleanup(config_dict['output_dir'])
                return {'status': 'no_media_approved'}
            
            selected_result = [review_items[i] for i in selected_indices if i < len(review_items)]
            selected_paths = [item['media_path'] for item in selected_result]
            
            if not strategy.handle_review_result(selected_indices, config_dict['output_dir'], selected_paths=selected_paths):
                self.cleanup(config_dict['output_dir'])
                return {'status': 'failed_to_continue'}
            selected_result_already_filtered = True
            
            similarity_threshold = config_dict.get('similarity_threshold', 0.9)
            final_filter_results = self.content_service.analyze_media_text_match(
                images=[],
                descriptions=content_result['descriptions'],
                similarity_threshold=similarity_threshold
            )
            
            content_result['filter_results'] = final_filter_results
            
            if strategy.needs_user_review():
                if strategy.should_generate_article_now():
                    article_content = self.content_service.generate_article(config, final_filter_results)
                    content_result['article_content'] = article_content
                
                review_items = strategy.get_review_items(max_items=10)
                
                if not review_items:
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_selected'}
                
                default_review_text = f'[策略: {strategy_name}] 請選擇要使用的影片'
                review_text = content_result.get('article_content') or default_review_text
                
                review_result = await self.review_service.review_content(
                    text=review_text,
                    media_paths=[item['media_path'] for item in review_items],
                    timeout=4000
                )
                
                approval_result, user, edited_content, selected_indices = review_result
                
                if approval_result == "accept" and not selected_indices:
                    selected_indices = list(range(len(review_items)))
                
                if not selected_indices:
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_approved'}
                
                if hasattr(strategy, '_videos_reviewed'):
                    strategy._videos_reviewed = True
                
                selected_result = [review_items[i] for i in selected_indices if i < len(review_items)]
                final_filter_results = selected_result
                content_result['filter_results'] = final_filter_results
                selected_result_already_filtered = True
                
                if edited_content and edited_content.strip():
                    content_result['article_content'] = edited_content
                else:
                    if not content_result.get('article_content') or not content_result['article_content'].strip():
                        article_content = self.content_service.generate_article(config, final_filter_results)
                        content_result['article_content'] = article_content
            else:
                if strategy.should_generate_article_now():
                    current_article_content = content_result.get('article_content', '')
                    if not current_article_content or current_article_content.strip() == '':
                        article_content = self.content_service.generate_article(config, final_filter_results)
                        content_result['article_content'] = article_content
                    else:
                        article_content = self.content_service.generate_article(config, final_filter_results)
                        content_result['article_content'] = article_content
                
                if 'selected_result' not in locals():
                    selected_result = final_filter_results
                    selected_indices = list(range(len(final_filter_results)))
                    selected_result_already_filtered = True
            
            if selected_result_already_filtered:
                selected_media_paths = [row['media_path'] for row in selected_result]
            else:
                selected_media_paths = [selected_result[i]['media_path'] for i in selected_indices if i < len(selected_result)]
                if not selected_media_paths:
                    selected_media_paths = [row['media_path'] for row in selected_result]
            
            processed_media_paths = strategy.post_process_media(
                selected_media_paths,
                config_dict['output_dir']
            )
            
            processed_media_paths = self.publishing_service.process_media(
                processed_media_paths, 
                config_dict['output_dir']
            )
            
            final_article_content = content_result.get('article_content', '')
            if edited_content and edited_content.strip():
                final_article_content = edited_content
            elif not final_article_content or not final_article_content.strip():
                final_article_content = '#ai #video #unbelievable #world #humor #interesting #funny #creative'
            
            post = MediaPost(
                media_paths=processed_media_paths,
                caption='',
                hashtags=final_article_content,
                additional_params={'share_to_story': True}
            )
            
            enabled_platforms = None
            if hasattr(character, '_social_media_config') and character._social_media_config:
                enabled_platforms = list(character._social_media_config.keys())
            
            publish_results = self.publishing_service.publish_to_social_media(
                post, 
                platforms=enabled_platforms
            )
            
            if any(publish_results.values()):
                execution_time = time.time() - start_time
                self.notification_service.notify_success(
                    character=config_dict['character'],
                    execution_time=execution_time,
                    media_count=len(selected_indices)
                )
            
            self.cleanup(config_dict['output_dir'])
            
            return {
                'status': 'success',
                'publish_results': publish_results,
                'media_count': len(selected_indices)
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
        """清理資源
        
        Args:
            output_dir: 要清理的輸出目錄
        """
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            self.logger.info(f"已清理資源目錄: {output_dir}") 