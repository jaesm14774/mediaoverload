import time
import random
import glob
from typing import Dict, Any

from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy

class Text2VideoStrategy(BaseGenerationStrategy):
    """文生影片策略"""

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)

        # Text2Video 專用的額外管理器
        if not self.external_vision_manager:
            self.ollama_vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='qwen2.5vl:7b') \
                .with_text_model('ollama', model_name='gemma3:4b') \
                .build()

    def set_vision_provider(self, provider: str = 'openrouter'):
        """設置視覺模型提供者

        Args:
            provider: 'ollama' 或 'openrouter'
        """
        if self.external_vision_manager:
            print(f"警告：正在使用外部傳入的 VisionManager，無法切換提供者到 {provider}")
            return

        if provider == 'ollama':
            self.current_vision_manager = self.ollama_vision_manager
        elif provider == 'openrouter':
            self.current_vision_manager = self.openrouter_vision_manager
        else:
            raise ValueError(f"不支援的視覺模型提供者: {provider}")
        print(f"已切換至 {provider} 視覺模型提供者")
    
    def generate_description(self):
        """生成視頻描述"""
        start_time = time.time()
        
        # 獲取策略專用配置
        video_config = self._get_strategy_config('text2video')
        
        # 獲取基本配置
        style = video_config.get('style') or getattr(self.config, 'style', '')
        
        image_system_prompt = video_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'unbelievable_world_system_prompt')
        
        prompt = self.config.prompt
        if style:
            prompt = f"""{prompt}\nstyle:{style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            # 檢查 prompt 中是否已包含角色名稱或 "main character" 等關鍵字
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")
        
        # 第一階段：使用角色名稱生成基礎描述
        character_description = self.current_vision_manager.generate_image_prompts(
            user_input=self.config.character,
            system_prompt_key=image_system_prompt
        )
        
        # 第二階段：使用視頻系統提示詞生成詳細的視頻描述
        video_description = self.current_vision_manager.generate_video_prompts(
            user_input=character_description,
            system_prompt_key='video_description_system_prompt'
        )
        
        print(f'Character description: {character_description}')
        print(f'Video description: {video_description}')
        
        # 檢查角色名稱是否存在於描述中
        if self.config.character:
            character = self.config.character.lower()
            if video_description.replace(' ', '').lower().find(character) != -1 or video_description.lower().find(character) != -1:
                self.descriptions = [video_description]
            else:
                self.descriptions = []
        else:
            self.descriptions = [video_description]
        
        print(f'Final video descriptions: {self.descriptions}')
        print(f'生成描述花費: {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成視頻或圖片"""
        start_time = time.time()
        
        if not self.descriptions:
            print("沒有描述可供生成視頻")
            return self
        
        # 獲取策略專用配置
        video_config = self._get_strategy_config('text2video')
        
        # 載入工作流
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            # 建立 WebSocket 連接（僅一次）
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始批次生成視頻")
            
            # 為每個描述生成視頻
            videos_per_description = video_config.get('videos_per_description', 2)
            
            total_videos = len(self.descriptions) * videos_per_description
            current_video = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                
                for i in range(videos_per_description):
                    current_video += 1
                    print(f'\n[{current_video}/{total_videos}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{videos_per_description} 個視頻')
                    
                    # 獲取自定義節點更新配置
                    default_video_updates = [
                        # 更新寬度和高度
                        {
                            "node_type": "PrimitiveInt",
                            "inputs": {"value": 512}
                        },
                        # 更新影片長度
                        {
                            "node_type": "EmptyHunyuanLatentVideo",
                            "inputs": {"length": 97}
                        }
                    ]
                    
                    custom_updates = video_config.get('custom_node_updates', default_video_updates)
                    
                    # 合併配置參數
                    merged_params = {**self.config.additional_params, **video_config}
                    
                    # 使用統一的更新方法（自動處理衝突檢查）
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    # 處理工作流，但不自動關閉 WebSocket（auto_close=False）
                    # 只有在最後一個視頻時才關閉（通過 finally 處理）
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_video_d{idx}_{i}",
                        auto_close=False  # 不自動關閉，保持連接以供下次使用
                    )
                    
        finally:
            # 確保在所有視頻生成完成後關閉 WebSocket
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有視頻生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ 生成視頻總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的視頻（暫時不做文本匹配分析）"""
        # 對於視頻，我們暫時不做文本匹配分析
        # 可以在未來擴展此功能
        video_paths = glob.glob(f'{self.config.output_dir}/*')

        # 簡單地返回所有生成的視頻
        self.filter_results = []
        for video_path in video_paths:
            if any(video_path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif']):
                self.filter_results.append({
                    'media_path': video_path,  # 保持與現有接口一致
                    'description': self.descriptions[0] if self.descriptions else '',
                    'similarity': 1.0  # 暫時設為1.0
                })

        return self
