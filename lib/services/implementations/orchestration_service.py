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
                    temperature=temperature,
                    character_config=character.config
                )
            
            # 特例處理 waddledee
            if re.sub(string=prompt.lower(), pattern='\s', repl='').find('waddledee') != -1:
                prompt = re.sub(string=prompt, pattern='waddledee|Waddledee', repl='waddle dee')
            
            self.logger.info(f"開始處理提示詞: {prompt}")
            
            # 步驟 3: 準備生成配置
            config_dict = character.get_generation_config(prompt)
            config_dict['output_dir'] = os.path.join(config_dict['output_dir'], config_dict['character'])
            config = GenerationConfig(**config_dict)
            
            # 步驟 4: 生成內容（第一階段）
            content_result = self.content_service.generate_content(config)
            
            strategy = self.content_service.strategy
            
            # 步驟 5: 檢查是否需要使用者審核
            if strategy.needs_user_review():
                self.logger.info('策略需要使用者審核，準備審核項目')
                
                # 獲取需要審核的項目（策略會處理 Discord 10 張限制）
                review_items = strategy.get_review_items(max_items=10)
                
                if not review_items:
                    self.logger.warning('沒有需要審核的項目')
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_selected'}
                
                # 準備審核文字（如果還沒有生成文章內容，使用預設文字）
                review_text = content_result.get('article_content', '請選擇要使用的圖片')
                
                # 步驟 6: 審核內容
                review_result = await self.review_service.review_content(
                    text=review_text,
                    media_paths=[item['media_path'] for item in review_items],
                    timeout=4000
                )
                
                approval_result, user, edited_content, selected_indices = review_result
                
                if not selected_indices:
                    self.logger.warning('沒有任何項目被用戶選中！')
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_approved'}
                
                # 步驟 6.5: Upscale 圖片（如果啟用且是 text2imgtovideo）
                # 對於 text2imgtovideo，在生成影片前先放大圖片
                strategy_type = strategy.__class__.__name__
                if strategy_type == 'Text2Image2VideoStrategy':
                    # 獲取選擇的圖片路徑
                    selected_image_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
                    
                    # 檢查是否為圖片（而非影片）
                    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
                    selected_images = [p for p in selected_image_paths if any(p.lower().endswith(ext) for ext in image_extensions)]
                    
                    if selected_images:
                        self.logger.info(f'開始放大 {len(selected_images)} 張圖片（text2imgtovideo 流程）')
                        
                        # 執行 upscale（workflow 路徑會從 config 中讀取）
                        upscaled_paths = strategy.upscale_images(
                            image_paths=selected_images,
                            output_dir=config_dict['output_dir']
                        )
                        
                        # 更新策略中的 first_stage_images 為放大後的圖片
                        if hasattr(strategy, 'first_stage_images'):
                            # 建立映射：原始圖片 -> 放大後的圖片
                            image_mapping = dict(zip(selected_images, upscaled_paths))
                            
                            # 保存原始圖片路徑（如果還沒有保存）
                            if not hasattr(strategy, 'original_images') or not strategy.original_images:
                                strategy.original_images = strategy.first_stage_images.copy()
                            
                            # 更新 first_stage_images 中的路徑
                            updated_first_stage_images = []
                            for img_path in strategy.first_stage_images:
                                if img_path in image_mapping:
                                    updated_first_stage_images.append(image_mapping[img_path])
                                else:
                                    updated_first_stage_images.append(img_path)
                            strategy.first_stage_images = updated_first_stage_images
                            
                            self.logger.info(f'已更新策略中的圖片路徑為放大後的版本，原始圖片路徑已保存')
                
                # 步驟 6.6: 繼續執行後續階段（策略會處理）
                if not strategy.continue_after_review(selected_indices):
                    self.logger.warning('策略無法繼續執行後續階段')
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'failed_to_continue'}
                
                # 重新分析結果（獲取最終的媒體）
                similarity_threshold = config_dict.get('similarity_threshold', 0.9)
                final_filter_results = self.content_service.analyze_media_text_match(
                    images=[],
                    descriptions=content_result['descriptions'],
                    similarity_threshold=similarity_threshold
                )
                
                # 更新 content_result
                content_result['filter_results'] = final_filter_results
                
                # 檢查是否需要再次審核（例如：影片生成後）
                if strategy.needs_user_review():
                    self.logger.info('策略需要再次使用者審核（影片生成後），準備審核項目')
                    
                    # 在影片審核前，先重新生成基於影片的文章內容
                    # 如果策略允許現在生成文章內容，則重新生成
                    if strategy.should_generate_article_now():
                        self.logger.info('重新生成基於影片的文章內容')
                        article_content = self.content_service.generate_article(config, final_filter_results)
                        content_result['article_content'] = article_content
                        self.logger.info(f'已生成文章內容（基於影片）: {article_content[:100]}...' if len(article_content) > 100 else f'已生成文章內容（基於影片）: {article_content}')
                    
                    # 獲取需要審核的項目（策略會處理 Discord 10 張限制）
                    review_items = strategy.get_review_items(max_items=10)
                    
                    if not review_items:
                        self.logger.warning('沒有需要審核的項目')
                        self.cleanup(config_dict['output_dir'])
                        return {'status': 'no_media_selected'}
                    
                    # 準備審核文字（使用新生成的基於影片的文章內容）
                    review_text = content_result.get('article_content', '請選擇要使用的影片')
                    
                    # 步驟 6.6: 再次審核內容（影片）
                    review_result = await self.review_service.review_content(
                        text=review_text,
                        media_paths=[item['media_path'] for item in review_items],
                        timeout=4000
                    )
                    
                    approval_result, user, edited_content, selected_indices = review_result
                    
                    if not selected_indices:
                        self.logger.warning('沒有任何項目被用戶選中！')
                        self.cleanup(config_dict['output_dir'])
                        return {'status': 'no_media_approved'}
                    
                    # 標記影片已審核
                    if hasattr(strategy, '_videos_reviewed'):
                        strategy._videos_reviewed = True
                    
                    # 更新最終結果為使用者選擇的影片
                    selected_result = [review_items[i] for i in selected_indices if i < len(review_items)]
                    final_filter_results = selected_result
                    content_result['filter_results'] = final_filter_results
                    
                    # 如果使用者在審核時編輯了內容，使用編輯後的內容；否則使用新生成的文章內容
                    if edited_content and edited_content.strip():
                        content_result['article_content'] = edited_content
                        self.logger.info('使用使用者編輯後的文章內容')
                    else:
                        # 確保使用新生成的基於影片的文章內容
                        if not content_result.get('article_content') or not content_result['article_content'].strip():
                            # 如果還是沒有文章內容，重新生成
                            article_content = self.content_service.generate_article(config, final_filter_results)
                            content_result['article_content'] = article_content
                            self.logger.info(f'已重新生成文章內容（基於影片）: {article_content[:100]}...' if len(article_content) > 100 else f'已重新生成文章內容（基於影片）: {article_content}')
                else:
                    # 如果不需要再次審核，但策略允許現在生成文章內容，則生成
                    if strategy.should_generate_article_now():
                        current_article_content = content_result.get('article_content', '')
                        # 如果當前文章內容是基於圖片的，重新生成基於影片的
                        if not current_article_content or current_article_content.strip() == '':
                            article_content = self.content_service.generate_article(config, final_filter_results)
                            content_result['article_content'] = article_content
                            self.logger.info(f'已生成文章內容（基於影片）: {article_content[:100]}...' if len(article_content) > 100 else f'已生成文章內容（基於影片）: {article_content}')
                        else:
                            # 即使有內容，也重新生成基於影片的內容（覆蓋基於圖片的內容）
                            article_content = self.content_service.generate_article(config, final_filter_results)
                            content_result['article_content'] = article_content
                            self.logger.info(f'已重新生成文章內容（基於影片，覆蓋原有內容）: {article_content[:100]}...' if len(article_content) > 100 else f'已重新生成文章內容（基於影片，覆蓋原有內容）: {article_content}')
                
                # 使用所有最終結果（因為使用者已經選擇過）
                if 'selected_result' not in locals():
                    selected_result = final_filter_results
                    selected_indices = list(range(len(final_filter_results)))
                
            else:
                # 不需要使用者審核的流程（原有邏輯）
                if len(content_result['filter_results']) == 0:
                    self.logger.warning('沒有任何生成的圖片(影片)被 LLM 分析選中！')
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_selected'}
                
                # 準備審核（如果圖片超過 6 張，隨機選擇 6 張）
                selected_result = (content_result['filter_results'] 
                                 if len(content_result['filter_results']) <= 6 
                                 else np.random.choice(content_result['filter_results'], size=6, replace=False).tolist())
                self.logger.info(f"選擇了 {len(selected_result)} 張圖片(影片)進行審核")
                
                # 步驟 6: 審核內容
                review_result = await self.review_service.review_content(
                    text=content_result['article_content'],
                    media_paths=[row['media_path'] for row in selected_result],
                    timeout=4000
                )
                
                approval_result, user, edited_content, selected_indices = review_result
                
                if not selected_indices:
                    self.logger.warning('沒有任何生成的圖片(影片)被用戶選中！')
                    self.cleanup(config_dict['output_dir'])
                    return {'status': 'no_media_approved'}
            
            # 步驟 7: Upscale 圖片（如果啟用且是 text2img）
            # 對於 text2img，在處理圖片前先放大
            strategy_type = strategy.__class__.__name__
            if strategy_type == 'Text2ImageStrategy':
                # 獲取選擇的圖片路徑
                selected_image_paths = [selected_result[i]['media_path'] for i in selected_indices if i < len(selected_result)]
                if not selected_image_paths:
                    selected_image_paths = [row['media_path'] for row in selected_result]
                
                # 檢查是否為圖片（而非影片）
                image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
                selected_images = [p for p in selected_image_paths if any(p.lower().endswith(ext) for ext in image_extensions)]
                
                if selected_images:
                    self.logger.info(f'開始放大 {len(selected_images)} 張圖片（text2img 流程）')
                    
                    # 執行 upscale（workflow 路徑會從 config 中讀取）
                    upscaled_paths = strategy.upscale_images(
                        image_paths=selected_images,
                        output_dir=config_dict['output_dir']
                    )
                    
                    # 使用放大後的圖片路徑
                    selected_media_paths = upscaled_paths + [p for p in selected_image_paths if p not in selected_images]
                else:
                    selected_media_paths = selected_image_paths
            else:
                # 其他策略（如 text2imgtovideo 的影片）不需要 upscale
                selected_media_paths = [selected_result[i]['media_path'] for i in selected_indices if i < len(selected_result)]
                if not selected_media_paths:
                    # 如果索引超出範圍，直接使用所有結果
                    selected_media_paths = [row['media_path'] for row in selected_result]
            
            # 步驟 7.5: 處理圖片(影片)
            processed_media_paths = self.publishing_service.process_media(
                selected_media_paths, 
                config_dict['output_dir']
            )
            
            # 步驟 8: 準備發布內容
            # 優先使用新生成的文章內容（基於影片），如果使用者在審核時有編輯，則使用編輯後的內容
            final_article_content = content_result.get('article_content', '')
            if edited_content and edited_content.strip():
                # 如果使用者在審核時編輯了內容，優先使用編輯後的內容
                final_article_content = edited_content
                self.logger.info('使用使用者編輯後的文章內容進行發布')
            elif not final_article_content or not final_article_content.strip():
                # 如果沒有文章內容，使用預設內容
                final_article_content = '#ai #video #unbelievable #world #humor #interesting #funny #creative'
                self.logger.warning('沒有文章內容，使用預設內容')
            else:
                self.logger.info('使用新生成的文章內容進行發布')
            
            # 步驟 8: 發布到社群媒體
            post = MediaPost(
                media_paths=processed_media_paths,
                caption='',
                hashtags=final_article_content,
                additional_params={'share_to_story': True}
            )
            
            # 從角色配置中獲取啟用的平台列表
            enabled_platforms = None
            if hasattr(character, '_social_media_config') and character._social_media_config:
                enabled_platforms = list(character._social_media_config.keys())
            
            publish_results = self.publishing_service.publish_to_social_media(
                post, 
                platforms=enabled_platforms
            )
            
            # 步驟 9: 發送通知
            if any(publish_results.values()):
                execution_time = time.time() - start_time
                self.notification_service.notify_success(
                    character=config_dict['character'],
                    execution_time=execution_time,
                    media_count=len(selected_indices)
                )
            
            # 步驟 10: 清理資源
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
        """清理資源"""
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            self.logger.info(f"已清理資源目錄: {output_dir}") 