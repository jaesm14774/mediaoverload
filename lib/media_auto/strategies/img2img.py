import time
import random
import glob
import os
from typing import Dict, Any

from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy

class Image2ImageStrategy(BaseGenerationStrategy):
    """圖生圖策略
    
    直接對現有圖片進行 image to image 生成。
    需要提供 input_image_path 配置。
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
        self.input_images: list = []  # 存儲輸入圖片路徑
    
    def generate_description(self):
        """生成描述 - 可選：從圖片中提取或使用提供的 prompt"""
        start_time = time.time()

        # 獲取策略專用配置
        i2i_config = self._get_strategy_config('image2image')
        
        # 檢查是否需要從圖片中提取描述
        extract_description = getattr(self.config, 'extract_description', False)
        input_image_path = getattr(self.config, 'input_image_path', None)
        
        if extract_description and input_image_path:
            # 從圖片中提取描述
            print(f"正在從圖片中提取描述: {input_image_path}")
            image_content = self.current_vision_manager.extract_image_content(input_image_path)
            
            # 獲取 image_system_prompt
            image_system_prompt = i2i_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')
            
            # 使用提取的內容生成描述
            descriptions = self.current_vision_manager.generate_image_prompts(image_content, image_system_prompt)
            self.descriptions = [descriptions] if descriptions else []
        else:
            # 使用提供的 prompt
            prompt = self.config.prompt or ''
            
            # 獲取 style
            style = i2i_config.get('style') or getattr(self.config, 'style', '')
            if style and style.strip():
                prompt = f"""{prompt}\nstyle: {style}""".strip()
            
            # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
            if self.config.character:
                character_lower = self.config.character.lower()
                prompt_lower = prompt.lower()
                if character_lower not in prompt_lower and "main character" not in prompt_lower:
                    prompt = f"Main character: {self.config.character}\n{prompt}"
                    print(f"已自動加入主角色到 prompt: {self.config.character}")
            
            # 如果沒有 prompt，使用預設描述
            if not prompt:
                prompt = "Enhance and improve the image with more details"
            
            self.descriptions = [prompt]
        
        # 收集輸入圖片
        if input_image_path:
            if os.path.isfile(input_image_path):
                self.input_images = [input_image_path]
            elif os.path.isdir(input_image_path):
                # 如果是目錄，收集所有圖片
                image_paths = glob.glob(f'{input_image_path}/*')
                self.input_images = [p for p in image_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            else:
                print(f"警告：輸入圖片路徑不存在: {input_image_path}")
                self.input_images = []
        else:
            print("警告：未提供 input_image_path")
            self.input_images = []
        
        print(f'Image descriptions : {self.descriptions}')
        print(f'Input images : {len(self.input_images)} 張')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成媒體 - Image to Image"""
        start_time = time.time()
        
        if not self.input_images:
            print("警告：沒有輸入圖片可供生成")
            return self
        
        # 獲取策略專用配置
        i2i_config = self._get_strategy_config('image2image')
        
        # 載入 workflow
        workflow_path = i2i_config.get('workflow_path') or self.config.workflow_path
        workflow = self._load_workflow(workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始 Image to Image 生成")
            
            # 獲取配置參數
            images_per_input = i2i_config.get('images_per_input', 1)
            
            total_images = len(self.input_images) * images_per_input
            current_image = 0
            
            for img_idx, input_image_path in enumerate(self.input_images):
                # 上傳圖片到 ComfyUI
                image_filename = self._upload_image_to_comfyui(input_image_path)
                
                # 獲取對應的描述（如果有多個描述，循環使用）
                desc_index = img_idx % len(self.descriptions) if self.descriptions else 0
                description = self.descriptions[desc_index] if self.descriptions else ''
                
                for i in range(images_per_input):
                    current_image += 1
                    print(f'\n[{current_image}/{total_images}] 使用圖片 {img_idx+1}/{len(self.input_images)}，生成第 {i+1}/{images_per_input} 張圖片')
                    
                    custom_updates = i2i_config.get('custom_node_updates', []).copy()
                    
                    # 添加 LoadImage 節點更新
                    custom_updates.append({
                        "node_type": "LoadImage",
                        "node_index": 0,
                        "inputs": {"image": image_filename}
                    })
                    
                    merged_params = {**self.config.additional_params, **i2i_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=random.randint(1, 999999999999) + i,
                        **merged_params
                    )
                    
                    is_last_image = (img_idx == len(self.input_images) - 1 and i == images_per_input - 1)
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_i2i_{img_idx}_{i}",
                        auto_close=False
                    )
            
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Image to Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片與描述的匹配度"""
        media_paths = glob.glob(f'{self.config.output_dir}/*')
        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]

        if not image_paths:
            self.filter_results = []
            return self

        # 預設使用 Gemini（更穩定且便宜），但保留 OpenRouter 作為備選
        # 如果需要使用 OpenRouter，可以通過 set_vision_provider('openrouter') 切換
        selected_manager = self.gemini_vision_manager
        print("使用 Gemini 進行圖像相似度分析")

        self.filter_results = selected_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=self.config.character,
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )
        return self
