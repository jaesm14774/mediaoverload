from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import inspect
import random
import time
import re
import os
import numpy as np

@dataclass
class GenerationConfig:
    """基礎生成配置類"""
    def __init__(self, **kwargs):
        self._attributes = kwargs  # 儲存所有屬性
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        return None
    
    def get_all_attributes(self):
        """獲取所有設置的屬性"""
        return self._attributes

class ContentStrategy(ABC):
    """內容生成策略的抽象基類"""
    
    def load_config(self, config: GenerationConfig):
        self.config = config
    
    def _get_strategy_config(self, strategy_type: str, stage: str = None) -> Dict[str, Any]:
        """獲取策略專用配置，支援 general 參數覆蓋
        
        Args:
            strategy_type: 策略類型 (text2img, text2image2video, 等)
            stage: 階段名稱 (first_stage, second_stage, video, 等)，可選
        
        Returns:
            合併後的配置字典（策略專用參數覆蓋 general 參數）
        """
        additional_params = getattr(self.config, 'additional_params', {})
        # 確保 additional_params 是字典類型
        if not isinstance(additional_params, dict):
            print(f"⚠️ additional_params 不是字典類型: {type(additional_params)}, 使用空字典")
            additional_params = {}
        
        general_params = additional_params.get('general', {}) or {}
        strategies = additional_params.get('strategies', {}) or {}
        strategy_config = strategies.get(strategy_type, {}) or {}
        
        if stage:
            stage_config = strategy_config.get(stage, {}) or {}
            # 合併：general -> strategy -> stage（後者覆蓋前者）
            return {**general_params, **strategy_config, **stage_config}
        else:
            # 合併：general -> strategy（strategy 覆蓋 general）
            return {**general_params, **strategy_config}
    
    def _get_config_value(self, config_dict: Dict[str, Any], key: str, default: Any = None) -> Any:
        """從配置字典或 config 屬性中獲取值
        
        優先級順序：
        1. config_dict 中的鍵值（已包含 general 和 stage 的合併結果）
        2. self.config 的屬性值
        3. 默認值
        
        Args:
            config_dict: 配置字典（通常來自 _get_strategy_config）
            key: 要查找的鍵名
            default: 默認值（如果都找不到）
        
        Returns:
            找到的值或默認值
        """
        # 首先從配置字典中查找
        if key in config_dict and config_dict[key] is not None:
            return config_dict[key]
        
        # 然後從 self.config 的屬性中查找
        if hasattr(self.config, key):
            value = getattr(self.config, key)
            if value is not None:
                return value
        
        # 最後返回默認值
        return default
    
    def _merge_node_manager_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合併參數供 NodeManager.generate_updates 使用
        
        以 config 為主，general 為輔助（config 覆蓋 general），並自動排除會作為明確參數傳遞的鍵
        
        邏輯說明：
        1. 先合併 general_params 和 config（config 會覆蓋 general_params 中相同的鍵）
        2. 然後排除所有會作為明確參數傳遞的鍵（如 description, seed, use_noise_seed）
        3. 最終返回的字典只包含會通過 **additional_params 傳遞的參數
        
        Args:
            config: 策略配置字典（優先級高）
        
        Returns:
            合併後的參數字典，已排除明確參數，只保留會通過 **additional_params 傳遞的參數
        """
        from lib.comfyui.node_manager import NodeManager
        
        additional_params = getattr(self.config, 'additional_params', {})
        if not isinstance(additional_params, dict):
            additional_params = {}
        
        general_params = additional_params.get('general', {}) or {}
        
        sig = inspect.signature(NodeManager.generate_updates)

        exclude_keys = [
            param_name for param_name, param in sig.parameters.items()
            if param.kind != inspect.Parameter.VAR_KEYWORD  # 排除 **kwargs
        ]

        merged = {**general_params, **config}

        return {k: v for k, v in merged.items() if k not in exclude_keys}
    
    @abstractmethod
    def generate_description(self):
        """生成內容的抽象方法"""
        pass
    
    @abstractmethod
    def generate_media(self):
        """生成媒體的抽象方法（圖片或視頻）"""
        pass
    
    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體文本匹配的抽象方法"""
        pass
    
    def generate_article_content(self):
        """生成文章內容 - 通用實現
        
        生成包含 SEO hashtags 和預設標籤的社群媒體文章內容。
        如果策略沒有 filter_results，則生成簡單的預設內容。
        """
        start_time = time.time()
        
        # 確保 filter_results 屬性存在
        if not hasattr(self, 'filter_results') or not self.filter_results:
            character = getattr(self.config, 'character', '')
            strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
            article_content = f"#{character} #AI #{strategy_name}" if character else f"#AI #{strategy_name}"
            
            # 確保 article_content 屬性存在
            if not hasattr(self, 'article_content'):
                self.article_content = ""
            self.article_content = article_content
            return self
        
        # 整合角色名稱、描述和預設標籤
        # 限制最多使用3張圖片的描述來生成文章內容
        limited_results = self.filter_results[:3]
        content_parts = [
            getattr(self.config, 'character', ''),
            *list(set([row['description'] for row in limited_results if 'description' in row])),
            getattr(self.config, 'prompt', '')
        ]
        
        # 過濾掉空字串
        content_parts = [part for part in content_parts if part and part.strip()]
        
        article_content = self.vision_manager.generate_seo_hashtags('\n\n'.join(content_parts))
        
        # 加入預設標籤
        if hasattr(self.config, 'default_hashtags') and self.config.default_hashtags:
            default_tags = ' #' + ' #'.join([tag.lstrip('#') for tag in self.config.default_hashtags])
            article_content += default_tags
        
        # 處理 redacted_reasoning 標籤（deepseek r1 格式）
        if '</think>' in article_content:
            article_content = article_content.split('</think>')[-1].strip()
        
        # 清理和格式化
        article_content = article_content.replace('"', '').replace('*', '').lower()
        
        # 防止 hashtag 數量過多
        article_content = self.prevent_hashtag_count_too_more(article_content)
        
        # 確保 article_content 屬性存在
        if not hasattr(self, 'article_content'):
            self.article_content = ""
        self.article_content = article_content
        
        print(f'產生文章內容花費: {time.time() - start_time:.2f} 秒')
        return self
    
    def prevent_hashtag_count_too_more(self, hashtag_text: str) -> str:
        """防止 hashtag 數量過多"""
        hashtag_candidate_list=[part.lower() for part in re.split(pattern='\n|#', string=hashtag_text) if part != '']

        deduplicate_list = []
        for part in hashtag_candidate_list:
            if part not in deduplicate_list:
                deduplicate_list.append(part)

        if len(deduplicate_list) > 30:
            hashtag_text = deduplicate_list[0] + '\n#' + '#'.join(deduplicate_list[1:2] + np.random.choice(deduplicate_list[2:], size=27, replace=False).tolist())

        return hashtag_text.lower().strip()
    
    def _get_random_secondary_character(self, main_character: str) -> Optional[str]:
        """獲取隨機的 Secondary Role
        
        從角色資料庫中獲取與主角色同群組的其他角色。
        
        Args:
            main_character: 主角色名稱
        
        Returns:
            次要角色名稱，如果無法獲取則返回 None
        """
        try:
            # 檢查是否有 character_repository
            if not hasattr(self, 'character_repository') or self.character_repository is None:
                try:
                    from lib.services.service_factory import ServiceFactory
                    service_factory = ServiceFactory()
                    character_repository = service_factory.get_character_repository()
                except ImportError:
                    print("無法導入 ServiceFactory，使用預設角色")
                    return self._get_default_secondary_character(main_character)
            else:
                character_repository = self.character_repository
            
            # 從角色配置中獲取 group_name 和 workflow
            group_name = getattr(self.config, 'group_name', '')
            workflow_path = getattr(self.config, 'workflow_path', '')
            
            # 從 workflow_path 中提取 workflow 名稱（去掉路徑和副檔名）
            workflow_name = os.path.splitext(os.path.basename(workflow_path))[0] if workflow_path else ''
            
            print(f"嘗試從群組 '{group_name}' 和工作流 '{workflow_name}' 中獲取角色")
            
            if group_name and workflow_name:
                # 從資料庫中獲取同群組的角色
                characters = character_repository.get_characters_by_group(group_name, workflow_name)
                
                # 過濾掉主角色
                available_characters = [
                    char for char in characters 
                    if char.lower() != main_character.lower()
                ]
                
                if available_characters:
                    selected_character = random.choice(available_characters)
                    print(f"從資料庫獲取到 Secondary Role: {selected_character}")
                    return selected_character
                else:
                    print(f"群組 '{group_name}' 中沒有其他可用角色")
            else:
                print(f"角色配置中缺少 group_name 或 workflow_path")
            
            # 如果無法從資料庫獲取，使用預設角色
            return self._get_default_secondary_character(main_character)
            
        except Exception as e:
            print(f"獲取 Secondary Role 時發生錯誤: {e}")
            return 
    
    def _generate_two_character_interaction_description(self, prompt: str, style: str = '') -> str:
        """生成雙角色互動描述
        
        這個方法會從資料庫中獲取一個 Secondary Role，並使用雙角色互動系統提示詞
        包含用戶原始 prompt。
        
        Args:
            prompt: 原始提示詞
            style: 風格描述（可選）
        
        Returns:
            雙角色互動的描述文字
        """
        from utils.logger import setup_logger
        logger = setup_logger('mediaoverload')
        
        logger.info('=' * 60)
        logger.info('開始生成雙角色互動描述')
        logger.info('=' * 60)
        logger.debug(f'原始 prompt: {prompt}')
        logger.debug(f'Style: {style}')
        
        try:
            # 優先使用 config 中指定的 secondary_character
            secondary_character = getattr(self.config, 'secondary_character', None)
            logger.debug(f'Config 中的 secondary_character: {secondary_character}')
            
            if not secondary_character:
                # 如果 config 中沒有指定，才從資料庫隨機獲取
                main_char = getattr(self.config, 'character', '')
                logger.debug(f'主角色: {main_char}')
                
                if not main_char:
                    logger.warning("⚠️ 警告：配置中沒有主角色，無法生成雙角色互動描述，返回原始 prompt")
                    print("⚠️ 警告：配置中沒有主角色，無法生成雙角色互動描述")
                    return prompt
                
                logger.info(f'從資料庫隨機獲取 Secondary Role（主角色: {main_char}）...')
                secondary_character = self._get_random_secondary_character(main_char)
                logger.info(f'獲取到的 Secondary Role: {secondary_character}')
            else:
                logger.info(f'使用指定的 Secondary Role: {secondary_character}')
                print(f'使用指定的 Secondary Role: {secondary_character}')
            
            if secondary_character:
                main_character = getattr(self.config, 'character', '')
                logger.info(f'雙角色互動配置：')
                logger.info(f'  Main Role: {main_character}')
                logger.info(f'  Secondary Role: {secondary_character}')
                logger.info(f'  Original Prompt: {prompt}')
                logger.info(f'  Style: {style}')
                print(f'雙角色互動：Main Role: {main_character}, Secondary Role: {secondary_character}')
                
                # 傳遞原始prompt給雙角色互動生成
                logger.info('調用 vision_manager.generate_two_character_interaction_prompt...')
                descriptions = self.vision_manager.generate_two_character_interaction_prompt(
                    main_character=main_character,
                    secondary_character=secondary_character,
                    prompt=prompt,
                    style=style  # 直接傳遞 style，不強制預設值
                )
                
                if descriptions and descriptions.strip():
                    logger.info(f'雙角色互動描述生成成功（長度: {len(descriptions)} 字元）')
                    logger.debug(f'生成的描述: {descriptions[:200]}...' if len(descriptions) > 200 else f'生成的描述: {descriptions}')
                    return descriptions
                else:
                    logger.warning('⚠️ 雙角色互動描述生成返回空值，使用預設方法')
                    print('⚠️ 雙角色互動描述生成返回空值，使用預設方法')
                    return self.vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
            else:
                logger.warning('無法獲取 Secondary Role，使用預設方法')
                print('無法獲取 Secondary Role，使用預設方法')
                return self.vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
            
        except Exception as e:
            logger.error(f'雙角色互動生成時發生錯誤: {e}', exc_info=True)
            print(f'雙角色互動生成時發生錯誤: {e}，使用預設方法')
            # 回退到預設方法
            if hasattr(self, 'vision_manager') and self.vision_manager:
                return self.vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
            return prompt
    
    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        預設返回 False，子類可以覆寫此方法來指示需要使用者審核
        例如：Text2Image2Video 在圖片生成後需要使用者選擇圖片
        
        Returns:
            bool: 如果需要使用者審核返回 True，否則返回 False
        """
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目
        
        預設返回 filter_results，子類可以覆寫此方法來提供自定義的審核項目
        注意：Discord API 一次最多上傳 10 張圖片
        
        Args:
            max_items: 最大項目數量（預設 10，符合 Discord 限制）
            
        Returns:
            List[Dict[str, Any]]: 需要審核的項目列表，每個項目包含 media_path 等資訊
        """
        if hasattr(self, 'filter_results'):
            return self.filter_results[:max_items]
        return []
    
    def continue_after_review(self, selected_indices: List[int]) -> bool:
        """在使用者審核後繼續執行後續階段
        
        預設返回 False（不需要後續操作），子類可以覆寫此方法來執行後續階段
        例如：Text2Image2Video 在使用者選擇圖片後需要生成影片
        
        Args:
            selected_indices: 使用者選擇的項目索引列表（相對於 get_review_items 返回的列表）
            
        Returns:
            bool: 如果成功繼續執行返回 True，否則返回 False
        """
        return False
    
    def should_generate_article_now(self) -> bool:
        """判斷是否應該現在生成文章內容
        
        預設返回 True，子類可以覆寫此方法來延遲文章生成
        例如：Text2Image2Video 應該在影片生成後才生成文章內容
        
        Returns:
            bool: 如果應該現在生成文章內容返回 True，否則返回 False
        """
        return True
    
    def post_process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        """後處理媒體文件（例如：放大圖片）
        
        預設不做任何處理，直接返回原始路徑
        子類可以覆寫此方法來實現特定的後處理邏輯（例如：圖片放大）
        
        Args:
            media_paths: 媒體文件路徑列表
            output_dir: 輸出路徑
            
        Returns:
            處理後的媒體文件路徑列表
        """
        return media_paths
    
    def handle_review_result(self, selected_indices: List[int], output_dir: str) -> bool:
        """處理使用者審核結果
        
        預設返回 False（不需要後續操作），子類可以覆寫此方法來執行後續階段
        例如：Text2Image2Video 在使用者選擇圖片後需要生成影片
        
        Args:
            selected_indices: 使用者選擇的項目索引列表（相對於 get_review_items 返回的列表）
            output_dir: 輸出路徑
            
        Returns:
            bool: 如果成功處理返回 True，否則返回 False
        """
        return False