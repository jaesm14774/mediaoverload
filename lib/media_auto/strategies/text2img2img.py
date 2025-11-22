import time
import random
import glob
import numpy as np
from typing import Dict, Any

from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy

class Text2Image2ImageStrategy(BaseGenerationStrategy):
    """文生圖 -> 圖生圖策略
    
    流程：
    1. text 2 image (使用 nova-anime-xl)
    2. image 2 image (使用 flux-dev-i2i)
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
        self.first_stage_images = []  # 存儲第一階段生成的圖片路徑
    
    def generate_description(self):
        """生成描述 - 使用策略專用配置（text2image2image 的 text2image 階段）"""
        start_time = time.time()

        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        # 獲取 style（優先使用策略專用配置）
        style = first_stage_config.get('style') or getattr(self.config, 'style', '')
        
        # 獲取 image_system_prompt（支援加權選擇或多個選項）
        # 優先檢查是否有 image_system_prompt_weights（加權選擇）
        image_system_prompt_weights = first_stage_config.get('image_system_prompt_weights')
        
        if image_system_prompt_weights:
            # 使用加權選擇
            image_system_prompt = self._process_weighted_choice(image_system_prompt_weights)
        else:
            # 使用單一的 image_system_prompt 配置
            image_system_prompt = first_stage_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')
        
        prompt = self.config.prompt
        # 只有當 style 不為空時才加上 style: 前綴
        if style and style.strip():
            prompt = f"""{prompt}\nstyle: {style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
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
        else:
            self.descriptions = [descriptions] if descriptions else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成媒體 - 分兩階段：text2image -> image2image"""
        start_time = time.time()
        
        # 第一階段：Text to Image
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        # 獲取 workflow 路徑
        t2i_workflow_path = first_stage_config.get('t2i_workflow_path') or self.config.workflow_path
        
        first_stage_workflow = self._load_workflow(t2i_workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始第一階段生成")
            
            # 第一階段參數
            images_per_description = first_stage_config.get('images_per_description', 4)
            
            total_first_stage = len(self.descriptions) * images_per_description
            current_image = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                for i in range(images_per_description):
                    current_image += 1
                    print(f'\n[第一階段 {current_image}/{total_first_stage}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{images_per_description} 張圖片')
                    
                    custom_updates = first_stage_config.get('custom_node_updates', [])
                    
                    merged_params = {**self.config.additional_params, **first_stage_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=first_stage_workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    is_last_image = (idx == len(self.descriptions) - 1 and i == images_per_description - 1)
                    self.communicator.process_workflow(
                        workflow=first_stage_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/first_stage",
                        file_name=f"{self.config.character}_d{idx}_{i}",
                        auto_close=False
                    )
            
            # 等待第一階段完成並收集圖片
            print("\n第一階段生成完成，開始第二階段 Image to Image 生成...")
            
            # 收集第一階段生成的圖片
            first_stage_output_dir = f"{self.config.output_dir}/first_stage"
            first_stage_images = glob.glob(f'{first_stage_output_dir}/*')
            image_paths = [p for p in first_stage_images if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            
            if not image_paths:
                print("警告：第一階段沒有生成任何圖片")
                return self
            
            # 使用視覺模型篩選最佳圖片
            print(f"正在從 {len(image_paths)} 張圖片中篩選最佳圖片...")
            
            # 預設使用 Gemini（更穩定且便宜），但保留 OpenRouter 作為備選
            # 如果需要使用 OpenRouter，可以通過 set_vision_provider('openrouter') 切換
            selected_manager = self.gemini_vision_manager
            print("使用 Gemini 進行圖像相似度分析")
            
            filter_results = selected_manager.analyze_media_text_match(
                media_paths=image_paths,
                descriptions=self.descriptions,
                main_character=self.config.character,
                similarity_threshold=0.0,  # 這裡不設定閾值，只為了獲取分數
                temperature=0.3
            )
            
            # 根據相似度排序，取前 N 張
            second_stage_config = self._get_strategy_config('text2image2image', 'second_stage')
            top_k = second_stage_config.get('top_k_selection', 2)
            
            sorted_results = sorted(filter_results, key=lambda x: x['similarity'], reverse=True)
            best_images = [res['media_path'] for res in sorted_results[:top_k]]
            
            # 保存第一階段的最佳圖片路徑
            self.first_stage_images = best_images
            print(f"已選出 {len(best_images)} 張最佳圖片進入第二階段")
            
            # 第二階段：Image to Image
            print("\n" + "=" * 60)
            print("第二階段：Image to Image 生成")
            print("=" * 60)
            
            i2i_workflow_path = second_stage_config.get('i2i_workflow_path') or 'configs/workflow/flux_dev_i2i.json'
            i2i_workflow = self._load_workflow(i2i_workflow_path)
            
            images_per_input = second_stage_config.get('images_per_input', 1)
            
            total_second_stage = len(best_images) * images_per_input
            current_i2i = 0
            
            for img_idx, input_image_path in enumerate(best_images):
                # 上傳圖片到 ComfyUI
                image_filename = self._upload_image_to_comfyui(input_image_path)
                
                # 找到這張圖片對應的描述（通過 filter_results）
                desc_index = 0
                for res in sorted_results:
                    if res['media_path'] == input_image_path:
                        # 嘗試找到描述在 self.descriptions 中的索引
                        try:
                            desc_index = self.descriptions.index(res['description'])
                        except ValueError:
                            desc_index = 0
                        break
                
                for i in range(images_per_input):
                    current_i2i += 1
                    print(f'\n[第二階段 {current_i2i}/{total_second_stage}] 使用圖片 {img_idx+1}/{len(best_images)}，生成第 {i+1}/{images_per_input} 張圖片')
                    
                    custom_updates = second_stage_config.get('custom_node_updates', []).copy()
                    
                    # 添加 LoadImage 節點更新
                    custom_updates.append({
                        "node_type": "LoadImage",
                        "node_index": 0,
                        "inputs": {"image": image_filename}
                    })
                    
                    merged_params = {**self.config.additional_params, **second_stage_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=i2i_workflow,
                        updates_config=custom_updates,
                        description=self.descriptions[desc_index],
                        seed=random.randint(1, 999999999999) + i,
                        **merged_params
                    )
                    
                    is_last_i2i = (img_idx == len(self.first_stage_images) - 1 and i == images_per_input - 1)
                    self.communicator.process_workflow(
                        workflow=i2i_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/second_stage",
                        file_name=f"{self.config.character}_i2i_{img_idx}_{i}",
                        auto_close=False
                    )
            
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Text2Image2Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        """分析第二階段生成的圖片與描述的匹配度"""
        second_stage_output_dir = f"{self.config.output_dir}/second_stage"
        media_paths = glob.glob(f'{second_stage_output_dir}/*')

        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]

        if not image_paths:
            self.filter_results = []
            return self

        self.filter_results = self.current_vision_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=self.config.character,
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )

        return self
