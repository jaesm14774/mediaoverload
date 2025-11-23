import time
import random
import glob
import re
import os
from typing import List, Dict, Any

from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy

class Text2Image2VideoStrategy(BaseGenerationStrategy):
    """文生圖 -> 圖生影片策略
    
    流程：
    1. text 2 image (使用 nova-anime-xl，可選)
    2. 透過 Discord 讓使用者選擇圖片（不做 AI 篩選）
    3. 透過 image 產生影片描述
    4. image + 影片描述 產生 音頻描述
    5. 使用 wan2.2_gguf_i2v workflow 產生含音頻的影片
    6. 使用影片生成文章內容
    7. 上傳到社群媒體
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
        self.first_stage_images: List[str] = []  # 第一階段生成的圖片路徑（可能是 upscale 後的）
        self.original_images: List[str] = []  # 原始圖片路徑（用於 extract_image_content 和生成文章內容）
        self.video_descriptions: Dict[str, str] = {}  # 圖片路徑 -> 影片描述
        self.audio_descriptions: Dict[str, str] = {}  # 圖片路徑 -> 音頻描述
        self._videos_generated = False  # 標記影片是否已生成
        self._videos_reviewed = False  # 標記影片是否已審核
    
    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核 - 在圖片生成後或影片生成後需要"""
        # 如果第一階段圖片已生成但影片未生成，則需要使用者審核圖片
        if len(self.first_stage_images) > 0 and not self._videos_generated:
            return True
        # 如果影片已生成但尚未審核，則需要使用者審核影片
        if self._videos_generated and not self._videos_reviewed:
            return True
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目（限制最多 10 張，符合 Discord API 限制）"""
        # 如果影片已生成但尚未審核，返回影片供審核
        if self._videos_generated and not self._videos_reviewed:
            video_output_dir = f"{self.config.output_dir}/videos"
            if os.path.exists(video_output_dir):
                media_paths = glob.glob(f'{video_output_dir}/*')
                video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]
                
                if video_paths:
                    review_items = []
                    for video_path in video_paths[:max_items]:
                        # 嘗試從文件名匹配對應的圖片和描述
                        match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                        if match:
                            img_idx = int(match.group(1))
                            if img_idx < len(self.first_stage_images):
                                img_path = self.first_stage_images[img_idx]
                                video_desc = self.video_descriptions.get(img_path, '')
                                review_items.append({
                                    'media_path': video_path,
                                    'description': video_desc,
                                    'similarity': 1.0
                                })
                        else:
                            review_items.append({
                                'media_path': video_path,
                                'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                                'similarity': 1.0
                            })
                    
                    if len(video_paths) > max_items:
                        print(f"警告：共有 {len(video_paths)} 個影片，但 Discord 限制一次最多 {max_items} 個，只發送前 {max_items} 個供審核")
                    
                    return review_items
        
        # 如果第一階段圖片已生成但影片未生成，返回圖片供審核
        if not self.first_stage_images:
            return []
        
        # 如果圖片超過 10 張，只返回前 10 張
        review_images = self.first_stage_images[:max_items]
        
        review_items = []
        for img_path in review_images:
            review_items.append({
                'media_path': img_path,
                'description': self.descriptions[0] if self.descriptions else '',
                'similarity': 1.0
            })
        
        if len(self.first_stage_images) > max_items:
            print(f"警告：共有 {len(self.first_stage_images)} 張圖片，但 Discord 限制一次最多 {max_items} 張，只發送前 {max_items} 張供審核")
        
        return review_items
    
    def continue_after_review(self, selected_indices: List[int]) -> bool:
        """在使用者審核後繼續執行後續階段 - 生成影片"""
        if not selected_indices:
            print("警告：沒有選擇任何圖片")
            return False
        
        # 將 selected_indices 映射回原始 first_stage_images 的索引
        # selected_indices 是相對於 get_review_items 返回的列表（最多 10 張）
        # 獲取審核項目（與發送給 Discord 的一致）
        review_items = self.get_review_items(max_items=10)
        selected_image_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
        
        if not selected_image_paths:
            print("警告：無法獲取選擇的圖片路徑")
            return False
        
        # 找到這些圖片在 first_stage_images 中的索引
        # 注意：如果圖片已經被放大，first_stage_images 中可能包含放大後的圖片路徑
        # 我們需要嘗試匹配原始路徑或放大後的路徑
        original_indices = []
        for img_path in selected_image_paths:
            found = False
            # 先嘗試直接匹配
            try:
                idx = self.first_stage_images.index(img_path)
                original_indices.append(idx)
                found = True
            except ValueError:
                # 如果直接匹配失敗，可能是因為圖片已經被放大
                # 嘗試通過文件名匹配（不包含路徑）
                img_basename = os.path.basename(img_path)
                for idx, first_stage_img in enumerate(self.first_stage_images):
                    first_stage_basename = os.path.basename(first_stage_img)
                    # 檢查文件名是否相似（可能是放大後的版本）
                    if img_basename in first_stage_basename or first_stage_basename in img_basename:
                        original_indices.append(idx)
                        found = True
                        print(f"通過文件名匹配找到圖片: {img_path} -> {first_stage_img}")
                        break
            
            if not found:
                print(f"警告：無法找到圖片路徑對應的索引: {img_path}")
        
        if not original_indices:
            print("警告：無法找到選擇的圖片對應的索引")
            return False
        
        # 生成影片（使用 first_stage_images 中的圖片，可能是放大後的）
        print(f"開始使用 {len(original_indices)} 張使用者選擇的圖片生成影片")
        self.generate_videos_from_selected_images(original_indices)
        self._videos_generated = True
        self._videos_reviewed = False  # 影片生成後需要審核
        return True
    
    def should_generate_article_now(self) -> bool:
        """判斷是否應該現在生成文章內容 - 應該在影片生成後才生成"""
        return self._videos_generated
    
    def generate_description(self):
        """生成描述 - 使用策略專用配置（text2image2video 的 text2image 階段）"""
        start_time = time.time()

        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
        # 獲取 style（優先使用策略專用配置）
        style = first_stage_config.get('style') or getattr(self.config, 'style', '')
        
        # 獲取 image_system_prompt（支援加權選擇或多個選項）
        # 優先檢查是否有 image_system_prompt_weights（加權選擇）
        image_system_prompt_weights = first_stage_config.get('image_system_prompt_weights')
        
        if image_system_prompt_weights:
            # 使用加權選擇，但排除雙角色互動
            image_system_prompt = self._process_weighted_choice(
                image_system_prompt_weights,
                exclude=['two_character_interaction_generate_system_prompt']
            )
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

        # 對於 text2image2video，強制不使用雙角色互動系統提示詞
        # 確保使用乾淨簡單的背景
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            print("警告：text2image2video 不支援雙角色互動，改用 stable_diffusion_prompt")
            image_system_prompt = 'stable_diffusion_prompt'
        
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
        """生成媒體 - 分多階段：text2image -> 篩選 -> 生成影片描述 -> 生成音頻描述 -> i2v"""
        start_time = time.time()
        
        # 第一階段：Text to Image
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
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
            
            # 等待第一階段完成
            print("\n第一階段生成完成，收集所有圖片供使用者審核...")
            
            # 收集第一階段生成的所有圖片（不做 AI 篩選）
            first_stage_output_dir = f"{self.config.output_dir}/first_stage"
            first_stage_images = glob.glob(f'{first_stage_output_dir}/*')
            image_paths = [p for p in first_stage_images if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            
            if not image_paths:
                print("警告：第一階段沒有生成任何圖片")
                return self
            
            # 保存所有圖片路徑（不做 AI 篩選，等待使用者選擇）
            self.first_stage_images = sorted(image_paths)  # 排序以確保順序一致
            self.original_images = sorted(image_paths).copy()  # 保存原始圖片路徑
            print(f"第一階段共生成 {len(self.first_stage_images)} 張圖片，將透過 Discord 供使用者審核選擇")
            
            # 在第一階段圖片生成後，立即生成發文內文（使用圖 + 描述）
            print("\n" + "=" * 60)
            print("生成發文內文（基於第一階段圖片和描述）")
            print("=" * 60)
            
            # 為每張圖片提取內容並生成描述（最多3張）
            # 使用原始圖片路徑，避免 upscale 後的圖片太大無法傳輸
            image_descriptions = []
            limited_images = self.original_images[:3] if self.original_images else self.first_stage_images[:3]  # 限制最多3張圖片，優先使用原始圖片
            print(f"將使用前 {len(limited_images)} 張原始圖片來生成文章內容（共 {len(self.first_stage_images)} 張）")
            for img_path in limited_images:
                image_content = self.current_vision_manager.extract_image_content(img_path)
                image_descriptions.append({
                    'media_path': img_path,
                    'description': image_content
                })
            
            # 暫時設置 filter_results 用於生成文章內容
            self.filter_results = image_descriptions
            
            # 生成文章內容
            self.generate_article_content()
            print(f"發文內文已生成: {self.article_content[:100]}..." if len(self.article_content) > 100 else f"發文內文已生成: {self.article_content}")
            
            # 注意：後續的影片生成將在使用者選擇圖片後進行
            # 請調用 generate_videos_from_selected_images 方法繼續流程
            
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有媒體生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Text2Image2Video 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def generate_videos_from_selected_images(self, selected_image_indices: List[int]):
        """根據使用者選擇的圖片生成影片
        
        Args:
            selected_image_indices: 使用者選擇的圖片索引列表（對應 self.first_stage_images）
        """
        if not selected_image_indices:
            print("警告：沒有選擇任何圖片")
            return self
        
        # 根據選擇的索引獲取圖片
        selected_images = [self.first_stage_images[i] for i in selected_image_indices if i < len(self.first_stage_images)]
        
        if not selected_images:
            print("警告：選擇的圖片索引無效")
            return self
        
        print(f"使用者選擇了 {len(selected_images)} 張圖片，開始生成影片...")
        
        # 更新 first_stage_images 為選擇的圖片（可能是 upscale 後的）
        self.first_stage_images = selected_images
        
        # 同步更新 original_images 為對應的原始圖片
        if self.original_images:
            selected_original_images = [self.original_images[i] for i in selected_image_indices if i < len(self.original_images)]
            self.original_images = selected_original_images
        else:
            # 如果 original_images 不存在，使用 first_stage_images（可能是原始圖片）
            self.original_images = selected_images.copy()
        
        start_time = time.time()
        
        # 第二階段：為每張圖片生成影片描述
        print("\n" + "=" * 60)
        print("第二階段：生成影片描述")
        print("=" * 60)
        
        for idx, img_path in enumerate(self.first_stage_images):
            print(f"為圖片生成影片描述: {img_path}")
            # 使用原始圖片路徑來提取內容，避免 upscale 後的圖片太大無法傳輸
            original_img_path = self.original_images[idx] if idx < len(self.original_images) else img_path
            # 從圖片中提取內容描述（使用原始圖片）
            image_content = self.current_vision_manager.extract_image_content(original_img_path)
            # 生成影片描述
            video_description = self.current_vision_manager.generate_video_prompts(
                user_input=image_content,
                system_prompt_key='video_description_system_prompt'
            )
            self.video_descriptions[img_path] = video_description
            print(f"影片描述: {video_description}")
        
        # 第三階段：為每張圖片生成音頻描述
        print("\n" + "=" * 60)
        print("第三階段：生成音頻描述")
        print("=" * 60)
        
        for idx, img_path in enumerate(self.first_stage_images):
            print(f"為圖片生成音頻描述: {img_path}")
            # 使用原始圖片路徑來生成音頻描述，避免 upscale 後的圖片太大無法傳輸
            original_img_path = self.original_images[idx] if idx < len(self.original_images) else img_path
            video_desc = self.video_descriptions.get(img_path, '')
            audio_description = self.current_vision_manager.generate_audio_description(
                image_path=original_img_path,  # 使用原始圖片
                video_description=video_desc
            )
            self.audio_descriptions[img_path] = audio_description
            print(f"音頻描述: {audio_description}")
        
        # 第四階段：使用 wan2.2 workflow 生成含音頻的影片
        print("\n" + "=" * 60)
        print("第四階段：Image to Video 生成（含音頻）")
        print("=" * 60)
        
        # 確保 WebSocket 連接存在
        if not self.communicator or not self.communicator.ws or not self.communicator.ws.connected:
            self.communicator = ComfyUICommunicator()
            self.communicator.connect_websocket()
            print("已重新建立 WebSocket 連接")
        
        try:
            # 獲取策略專用配置
            video_config = self._get_strategy_config('text2image2video', 'video')
            
            # 載入 i2v workflow
            i2v_workflow_path = video_config.get('i2v_workflow_path') or 'configs/workflow/wan2.2_gguf_i2v_audio.json'
            i2v_workflow = self._load_workflow(i2v_workflow_path)
            
            videos_per_image = video_config.get('videos_per_image', 1)
            
            # 檢查是否使用 noise_seed
            use_noise_seed = video_config.get('use_noise_seed', False)
            
            total_videos = len(self.first_stage_images) * videos_per_image
            current_video = 0
            
            for img_idx, input_image_path in enumerate(self.first_stage_images):
                image_filename = self._upload_image_to_comfyui(input_image_path)
                video_desc = self.video_descriptions.get(input_image_path, '')
                audio_desc = self.audio_descriptions.get(input_image_path, '')
                
                # 為每張圖片生成影片時，使用不同的 seed 起始值
                seed_start = random.randint(1, 999999999999)
                
                for i in range(videos_per_image):
                    current_video += 1
                    print(f'\n[第四階段 {current_video}/{total_videos}] 使用圖片 {img_idx+1}/{len(self.first_stage_images)}，生成第 {i+1}/{videos_per_image} 個影片')
                    
                    custom_updates = video_config.get('custom_node_updates', []).copy()  # 複製列表避免修改原始配置
                    
                    # 添加 LoadImage 節點更新
                    custom_updates.append({
                        "node_type": "LoadImage",
                        "node_index": 0,
                        "inputs": {"image": image_filename}
                    })
                    
                    # 添加 Positive prompt 更新（影片描述）
                    # 使用 node_id 直接更新，避免過濾條件問題
                    # 根據 workflow JSON，節點 70 是 "Positive prompt" (PrimitiveString)
                    custom_updates.append({
                        "node_id": "70",
                        "inputs": {"value": video_desc}
                    })
                    
                    # 添加 Audio prompt 更新
                    # 根據 workflow JSON，節點 94 是 "Audio prompt" (PrimitiveString)
                    custom_updates.append({
                        "node_id": "94",
                        "inputs": {"value": audio_desc}
                    })
                    
                    merged_params = {**self.config.additional_params, **video_config}
                    # 從 merged_params 中移除 use_noise_seed，因為它已經被明確傳遞
                    merged_params.pop('use_noise_seed', None)
                    
                    # 確保每個影片使用不同的 seed（基於圖片索引和影片索引）
                    video_seed = seed_start + (img_idx * videos_per_image) + i
                    
                    updates = self.node_manager.generate_updates(
                        workflow=i2v_workflow,
                        updates_config=custom_updates,
                        description=video_desc,
                        seed=video_seed,
                        use_noise_seed=use_noise_seed,  # 傳遞 use_noise_seed 參數
                        **merged_params
                    )
                    
                    is_last_video = (img_idx == len(self.first_stage_images) - 1 and i == videos_per_image - 1)
                    self.communicator.process_workflow(
                        workflow=i2v_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/videos",
                        file_name=f"{self.config.character}_i2v_{img_idx}_{i}",
                        auto_close=False
                    )
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有影片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        # 更新 filter_results 為生成的影片結果，以便後續生成文章內容
        video_output_dir = f"{self.config.output_dir}/videos"
        if os.path.exists(video_output_dir):
            media_paths = glob.glob(f'{video_output_dir}/*')
            video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]
            
            if video_paths:
                self.filter_results = []
                for video_path in video_paths:
                    # 嘗試從文件名匹配對應的圖片和描述
                    match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                    if match:
                        img_idx = int(match.group(1))
                        if img_idx < len(self.first_stage_images):
                            img_path = self.first_stage_images[img_idx]
                            video_desc = self.video_descriptions.get(img_path, '')
                            self.filter_results.append({
                                'media_path': video_path,
                                'description': video_desc,
                                'similarity': 1.0
                            })
                    else:
                        # 如果無法匹配，使用第一個描述
                        self.filter_results.append({
                            'media_path': video_path,
                            'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                            'similarity': 1.0
                        })
                print(f"已更新 filter_results，包含 {len(self.filter_results)} 個影片")
        
        print(f'\n✅ 影片生成總耗時: {time.time() - start_time:.2f} 秒')
        return self
    
    def analyze_media_text_match(self, similarity_threshold):
        """分析生成的媒體
        
        對於 Text2Image2Video：
        - 如果影片已生成，返回所有影片（不做 AI 判讀）
        - 如果影片未生成，返回所有第一階段的圖片供 Discord 審核
        """
        # 檢查是否已有生成的影片
        video_output_dir = f"{self.config.output_dir}/videos"
        if os.path.exists(video_output_dir):
            media_paths = glob.glob(f'{video_output_dir}/*')
            video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]

            if video_paths:
                # 影片已生成，返回所有影片（不做 AI 判讀）
                self.filter_results = []
                for video_path in video_paths:
                    # 嘗試從文件名匹配對應的圖片和描述
                    match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                    if match:
                        img_idx = int(match.group(1))
                        if img_idx < len(self.first_stage_images):
                            img_path = self.first_stage_images[img_idx]
                            video_desc = self.video_descriptions.get(img_path, '')
                            self.filter_results.append({
                                'media_path': video_path,
                                'description': video_desc,
                                'similarity': 1.0  # 不做 AI 判讀，設為 1.0
                            })
                    else:
                        # 如果無法匹配，使用第一個描述
                        self.filter_results.append({
                            'media_path': video_path,
                            'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                            'similarity': 1.0
                        })
                return self
        
        # 影片未生成，返回所有第一階段的圖片供 Discord 審核（不做 AI 篩選）
        if not self.first_stage_images:
            self.filter_results = []
            return self
        
        # 返回所有圖片，不做 AI 篩選
        self.filter_results = []
        for img_path in self.first_stage_images:
            self.filter_results.append({
                'media_path': img_path,
                'description': self.descriptions[0] if self.descriptions else '',
                'similarity': 1.0  # 不做 AI 判讀，設為 1.0
            })
        
        print(f"返回 {len(self.filter_results)} 張圖片供 Discord 審核（不做 AI 篩選）")
        return self
