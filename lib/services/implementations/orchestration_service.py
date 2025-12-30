"""協調服務實現"""
import os
import time
import shutil
import numpy as np
from typing import Dict, Any, Optional
from lib.services.interfaces.orchestration_service import IOrchestrationService
from lib.media_auto.character_config import BaseCharacter
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.services.interfaces.character_data_service import ICharacterDataService
from lib.social_media import MediaPost
from utils.logger import setup_logger, log_execution_time


class OrchestrationService(IOrchestrationService):
    """協調服務實現
    
    負責協調各個服務完成整個工作流程：提示詞生成、內容生成、審核、發布。
    """
    
    def __init__(self, character_data_service: ICharacterDataService = None):
        """初始化協調服務
        
        Args:
            character_data_service: 角色資料服務（可選）
        """
        self.logger = setup_logger(__name__)
        self.character_data_service = character_data_service
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
            # 保存原始群組代表角色（用於查詢同群組的其他角色）
            group_representative_character = character.character
            
            # 處理角色選擇
            if character.group_name and self.character_data_service:
                generation_type = getattr(character.config, 'generation_type', '')
                is_kirby_group = character.group_name.lower() == 'kirby'
                is_longvideo = generation_type.lower() == 'text2longvideo'
                
                if is_kirby_group and is_longvideo:
                    character.character = 'kirby'
                    character.config.character = 'kirby'
                    self.logger.info(f"長影片模式：群組 {character.group_name} 直接使用角色 kirby")
                else:
                    workflow_name = os.path.splitext(os.path.basename(character.workflow_path))[0]
                    dynamic_character = self.character_data_service.get_random_character_from_group(
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
                self.logger.info(f"生成提示詞: {prompt}")
            
            # 準備生成配置
            config_dict = character.get_generation_config(prompt)
            config_dict['output_dir'] = os.path.join(config_dict['output_dir'], config_dict['character'])
            # 傳遞群組代表角色，用於雙角色互動查詢
            config_dict['group_representative_character'] = group_representative_character
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
            
            # 準備 filter_results 用於生成 article_content
            if hasattr(strategy, 'filter_results') and strategy.filter_results:
                initial_filter_results = strategy.filter_results
            else:
                similarity_threshold = config_dict.get('similarity_threshold', 0.9)
                initial_filter_results = self.content_service.analyze_media_text_match(
                    images=[],
                    descriptions=content_result['descriptions'],
                    similarity_threshold=similarity_threshold
                )
            
            # 檢查策略是否應該在第一次審核時顯示 article_content
            should_show_article = strategy.should_show_article_in_first_review()
            
            # 如果策略要求在第一次審核時顯示 article_content，現在生成
            if should_show_article and strategy.should_generate_article_now():
                # 暫時設置 filter_results 以便生成 article_content
                original_filter_results = getattr(strategy, 'filter_results', None)
                strategy.filter_results = initial_filter_results
                
                article_content = self.content_service.generate_article(config, initial_filter_results)
                content_result['article_content'] = article_content
                
                # 恢復 filter_results
                if original_filter_results is not None:
                    strategy.filter_results = original_filter_results
                elif hasattr(strategy, 'filter_results'):
                    delattr(strategy, 'filter_results')
            
            # 第一次審核：根據策略決定顯示內容
            if should_show_article and content_result.get('article_content'):
                review_text = content_result['article_content']
            else:
                review_text = f'[策略: {strategy_name}] 請選擇要使用的媒體'
            
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
            
            # 使用策略中已設置的 filter_results（handle_review_result 已設置）
            if hasattr(strategy, 'filter_results') and strategy.filter_results:
                final_filter_results = strategy.filter_results
            else:
                final_filter_results = initial_filter_results
            
            content_result['filter_results'] = final_filter_results
            
            # 如果策略在第一次審核時顯示了 article_content，處理用戶編輯
            if should_show_article:
                if edited_content and edited_content.strip():
                    content_result['article_content'] = edited_content
                elif not content_result.get('article_content') or not content_result['article_content'].strip():
                    article_content = self.content_service.generate_article(config, final_filter_results)
                    content_result['article_content'] = article_content
            
            # 檢查是否需要第二次審核（多階段策略）
            if strategy.needs_user_review():
                # 第二次審核：在最後一次審核時生成最終 article content
                if strategy.should_generate_article_now():
                    article_content = self.content_service.generate_article(config, final_filter_results)
                    content_result['article_content'] = article_content
                
                review_items = strategy.get_review_items(max_items=10)
                
                if not review_items:
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_selected'}
                
                # 第二次審核：只顯示最終 article_content，不顯示策略名稱
                if content_result.get('article_content'):
                    review_text = content_result['article_content']
                else:
                    review_text = '請選擇要使用的媒體'
                
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
                
                # 如果用戶編輯了內容，使用編輯後的內容；否則確保有最終 article content
                if edited_content and edited_content.strip():
                    content_result['article_content'] = edited_content
                elif not content_result.get('article_content') or not content_result['article_content'].strip():
                    article_content = self.content_service.generate_article(config, final_filter_results)
                    content_result['article_content'] = article_content
            else:
                # 單階段策略：如果第一次 review 時沒有生成 article_content，現在生成
                if strategy.should_generate_article_now():
                    current_article_content = content_result.get('article_content', '')
                    if not current_article_content or current_article_content.strip() == '':
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