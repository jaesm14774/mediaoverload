import time
import random
from typing import Dict, Any, List
import numpy as np
import glob

from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.media_auto.models.vision.model_switcher import ModelSwitcher
from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy

class Text2ImageStrategy(BaseGenerationStrategy):
    """
    文生圖策略
    實驗顯示 ollama 在判斷 文圖匹配時，不論是llama3.2-vision 或是 llava 都沒有達到期望
    改成google 大幅 增加準確性
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)

        # Text2Image 專用的額外管理器
        if not self.external_vision_manager:
            self.ollama_vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='llava:13b') \
                .with_text_model('ollama', model_name='llama3.2:latest') \
                .build()
            self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)

    def generate_description(self):
        """描述生成"""
        start_time = time.time()

        # 獲取策略專用配置
        image_config = self._get_strategy_config('text2img')
        
        # 獲取 style 和 image_system_prompt
        style = image_config.get('style') or getattr(self.config, 'style', '')
        
        image_system_prompt_weights = image_config.get('image_system_prompt_weights')
        
        # 如果有權重配置，使用權重選擇
        if image_system_prompt_weights:
            image_system_prompt = self._process_weighted_choice(image_system_prompt_weights)
        else:
            image_system_prompt = image_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')

        prompt = self.config.prompt
        # 只有當 style 不為空時才加上 style: 前綴
        if style and style.strip():
            prompt = f"""{prompt}\nstyle: {style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            # 檢查 prompt 中是否已包含角色名稱或 "main character" 等關鍵字
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")

        # 檢查是否使用雙角色互動系統提示詞
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            # 使用雙角色互動生成邏輯
            descriptions = self._generate_two_character_interaction_description(prompt, style)
        else:
            # 使用原有的生成邏輯
            descriptions = self.current_vision_manager.generate_image_prompts(prompt, image_system_prompt)
        
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [descriptions] if descriptions.replace(' ', '').lower().find(character) != -1 or descriptions.lower().find(character) != -1 else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
        
    def generate_media(self):
        start_time = time.time()
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            # 建立 WebSocket 連接（僅一次）
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始批次生成圖片")
            
            # 獲取策略專用配置
            image_config = self._get_strategy_config('text2img')
            
            images_per_description = image_config.get('images_per_description', 4)
            
            total_images = len(self.descriptions) * images_per_description
            current_image = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                for i in range(images_per_description):
                    current_image += 1
                    print(f'\n[{current_image}/{total_images}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{images_per_description} 張圖片')
                    
                    # 獲取自定義節點更新配置
                    custom_updates = image_config.get('custom_node_updates', [])
                    
                    # 合併配置參數
                    merged_params = {**self.config.additional_params, **image_config}
                    
                    # 使用統一的更新方法（自動處理衝突檢查）
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    # 處理工作流，但不自動關閉 WebSocket（auto_close=False）
                    # 只有在最後一張圖片時才關閉（通過 finally 處理）
                    is_last_image = (idx == len(self.descriptions) - 1 and i == images_per_description - 1)
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_d{idx}_{i}",
                        auto_close=False  # 不自動關閉，保持連接以供下次使用
                    )
                    
        finally:
            # 確保在所有圖片生成完成後關閉 WebSocket
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
                
        print(f'\n✅ 生成圖片總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片 - 預設使用 Gemini，保留 OpenRouter 作為備選"""
        media_paths = glob.glob(f'{self.config.output_dir}/*')

        # 預設使用 Gemini（更穩定且便宜），但保留 OpenRouter 作為備選
        # 如果需要使用 OpenRouter，可以通過 set_vision_provider('openrouter') 切換
        selected_manager = self.gemini_vision_manager
        print("使用 Gemini 進行圖像相似度分析")

        self.filter_results = selected_manager.analyze_media_text_match(
            media_paths=media_paths,
            descriptions=self.descriptions,
            main_character=self.config.character,
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )
        return self
    
    def post_process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        """後處理媒體文件 - 放大圖片
        
        Args:
            media_paths: 媒體文件路徑列表
            output_dir: 輸出路徑
            
        Returns:
            處理後的媒體文件路徑列表（放大後的圖片路徑）
        """
        # 檢查是否為圖片（而非影片）
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        selected_images = [p for p in media_paths if any(p.lower().endswith(ext) for ext in image_extensions)]
        
        if not selected_images:
            return media_paths
        
        # 執行 upscale
        upscaled_paths = self.upscale_images(
            image_paths=selected_images,
            output_dir=output_dir
        )
        
        # 合併放大後的圖片和其他媒體文件（如影片）
        processed_paths = upscaled_paths + [p for p in media_paths if p not in selected_images]
        
        return processed_paths