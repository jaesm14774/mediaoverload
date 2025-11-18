import json
import uuid
import websocket
import urllib
from urllib import request
import os
import time
from typing import Dict, List, Optional, Tuple


from lib.comfyui.analyze import analyze_workflow

class ComfyUICommunicator:
    def __init__(self, host=None, port=None, timeout=900):
        # 從環境變數讀取，如果沒有則使用預設值
        self.host = host or os.environ.get('COMFYUI_HOST', 'host.docker.internal')
        self.port = port or int(os.environ.get('COMFYUI_PORT', '8188'))
        self.client_id = str(uuid.uuid4())
        self.server_address = f"{self.host}:{self.port}"
        self.timeout = timeout
        self.ws = None

    def connect_websocket(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(
            f"ws://{self.server_address}/ws?clientId={self.client_id}",
            ping_interval=20, # 每 20 秒發送一次 ping
            ping_timeout=10   # 10 秒內未收到 pong 則超時
        )

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(request.urlopen(req).read())
    
    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = False) -> str:
        """上傳圖片到 ComfyUI 伺服器
        
        Args:
            image_path: 本地圖片路徑
            subfolder: 子資料夾名稱（可選）
            overwrite: 是否覆蓋已存在的文件
            
        Returns:
            上傳後的圖片文件名
        """
        import mimetypes
        
        # 獲取文件類型
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/png'
        
        # 讀取圖片文件
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        filename = os.path.basename(image_path)
        
        # 構建 multipart/form-data 請求
        boundary = '----WebKitFormBoundary' + str(uuid.uuid4()).replace('-', '')
        
        # 構建請求體
        body_parts = []
        
        # 添加 overwrite 參數
        body_parts.append(f'--{boundary}\r\n'.encode())
        body_parts.append(f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'.encode())
        body_parts.append(str(overwrite).lower().encode())
        body_parts.append('\r\n'.encode())
        
        # 如果有 subfolder，添加 subfolder 參數
        if subfolder:
            body_parts.append(f'--{boundary}\r\n'.encode())
            body_parts.append(f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'.encode())
            body_parts.append(subfolder.encode())
            body_parts.append('\r\n'.encode())
        
        # 添加圖片文件
        body_parts.append(f'--{boundary}\r\n'.encode())
        body_parts.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode())
        body_parts.append(f'Content-Type: {mime_type}\r\n\r\n'.encode())
        body_parts.append(image_data)
        body_parts.append(f'\r\n--{boundary}--\r\n'.encode())
        
        body = b''.join(body_parts)
        
        # 發送上傳請求
        req = request.Request(
            f"http://{self.server_address}/upload/image",
            data=body,
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}',
            }
        )
        
        try:
            response = request.urlopen(req)
            result = json.loads(response.read().decode('utf-8'))
            uploaded_filename = result.get('name', filename)
            print(f"✅ 圖片已上傳到 ComfyUI: {uploaded_filename}")
            return uploaded_filename
        except Exception as e:
            # 如果上傳失敗，嘗試直接使用文件名（假設圖片已經在 ComfyUI 的 input 目錄）
            print(f"⚠️ 圖片上傳失敗: {e}")
            print(f"   嘗試直接使用文件名: {filename}")
            print(f"   💡 提示：請確保圖片已手動複製到 ComfyUI 的 input 目錄")
            return filename
            
    def get_media_file(self, filename, subfolder, folder_type):
        """
        獲取媒體檔案（圖片、影片、GIF等）
        """
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()
            
    def get_history(self, prompt_id):
        with request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def wait_for_completion(self, prompt_id):
        start_time = time.time()
        last_message_time = start_time
        last_node = None
        
        print(f"開始等待工作流 {prompt_id} 完成...")
        
        while True:
            # 檢查是否超時
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                raise TimeoutError(f"工作流 {prompt_id} 執行超時（{self.timeout} 秒）。最後處理的節點: {last_node}")
            
            # 檢查 WebSocket 是否仍然連接
            if not self.ws or not self.ws.connected:
                raise Exception(f"WebSocket 連接已斷開。最後處理的節點: {last_node}")
            
            try:
                # 設置 websocket 接收超時時間為 5 秒
                self.ws.settimeout(5.0)
                out = self.ws.recv()
                
                if isinstance(out, str):
                    message = json.loads(out)
                    message_type = message.get('type', 'unknown')
                    
                    # 更新最後接收消息的時間
                    last_message_time = time.time()
                    
                    # 處理不同類型的消息
                    if message_type == 'executing':
                        data = message.get('data', {})
                        current_node = data.get('node')
                        current_prompt_id = data.get('prompt_id')
                        
                        # 檢查是否是我們的 prompt_id
                        if current_prompt_id == prompt_id:
                            if current_node is None:
                                # 工作流執行完成
                                print(f"✓ 工作流 {prompt_id} 執行完成（耗時 {elapsed_time:.2f} 秒）")
                                break
                            else:
                                # 更新當前處理的節點
                                if current_node != last_node:
                                    last_node = current_node
                                    print(f"  → 正在處理節點: {current_node}")
                    
                    elif message_type == 'progress':
                        # 顯示進度信息
                        data = message.get('data', {})
                        value = data.get('value', 0)
                        max_value = data.get('max', 100)
                        if max_value > 0:
                            progress = (value / max_value) * 100
                            print(f"  → 進度: {progress:.1f}% ({value}/{max_value})")
                    
                    elif message_type == 'status':
                        # 顯示狀態信息
                        data = message.get('data', {})
                        status_data = data.get('status', {})
                        exec_info = status_data.get('exec_info', {})
                        queue_remaining = exec_info.get('queue_remaining', 0)
                        if queue_remaining > 0:
                            print(f"  → 佇列中還有 {queue_remaining} 個任務")
                    
                    elif message_type == 'execution_error':
                        # 執行錯誤
                        data = message.get('data', {})
                        error_prompt_id = data.get('prompt_id')
                        if error_prompt_id == prompt_id:
                            error_node = data.get('node_id')
                            error_type = data.get('exception_type')
                            error_message = data.get('exception_message')
                            raise Exception(f"工作流執行錯誤 - 節點: {error_node}, 類型: {error_type}, 消息: {error_message}")
                            
            except websocket.WebSocketTimeoutException:
                # 接收超時，檢查是否長時間沒有收到消息
                time_since_last_message = time.time() - last_message_time
                if time_since_last_message > 60:  # 60 秒沒有收到任何消息
                    print(f"⚠ 警告: 已經 {time_since_last_message:.1f} 秒沒有收到消息了...")
                # 繼續等待
                continue
                
            except json.JSONDecodeError as e:
                # JSON 解析錯誤，忽略並繼續
                print(f"⚠ 收到無效的 JSON 消息: {e}")
                continue
                
            except Exception as e:
                # 其他錯誤
                raise Exception(f"等待工作流完成時發生錯誤: {str(e)}")

    def analyze_node_connections(self, workflow: Dict) -> Dict[str, Dict]:
        """分析節點之間的連接關係"""
        connections = {}
        
        for node_id, node_data in workflow.items():
            # 如果 node_data 不是 dict，跳過
            if not isinstance(node_data, dict):
                continue
                
            connections[node_id] = {
                "inputs": {},
                "outputs": {},
                "class_type": node_data.get("class_type"),
                "node_data": node_data
            }
            
            # 分析輸入連接
            if "inputs" in node_data:
                for input_name, input_value in node_data["inputs"].items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        source_id, output_index = input_value
                        connections[node_id]["inputs"][input_name] = {
                            "source_node": str(source_id),
                            "output_index": output_index
                        }

        return connections

    def find_nodes_by_type(self, workflow: Dict, node_type: str) -> List[Tuple[str, Dict]]:
        """找出特定類型的所有節點，並返回節點ID和節點數據"""
        return [
            (node_id, node_data) 
            for node_id, node_data in workflow.items()
            if isinstance(node_data, dict) and node_data.get("class_type") == node_type
        ]

    def trace_back_to_text_encoder(self, node_id: str, connections: Dict) -> Optional[str]:
        """追蹤節點的輸入直到找到 CLIPTextEncode 節點"""
        def trace(current_id: str, visited: set) -> Optional[str]:
            if current_id in visited:
                return None
            visited.add(current_id)
            
            node_info = connections.get(current_id)
            if not node_info:
                return None
                
            if node_info["class_type"] == "CLIPTextEncode":
                return current_id
                
            for input_info in node_info["inputs"].values():
                if isinstance(input_info, dict):
                    source_id = input_info.get("source_node")
                    if source_id:
                        result = trace(source_id, visited)
                        if result:
                            return result
            return None
            
        return trace(node_id, set())
                        
    def save_results(self, prompt_id: str, output_path: str, file_name) -> Tuple[bool, List[str]]:
        """
        儲存執行結果並返回儲存的檔案列表
        支援圖片和影片的儲存
        """
        try:
            # 獲取歷史記錄
            history = self.get_history(prompt_id)[prompt_id]
            saved_files = []
            
            def process_media_files(media_list: List[Dict], default_extension: str = None):
                """處理媒體文件的通用函數"""
                for media in media_list:
                    # 獲取媒體數據
                    media_data = self.get_media_file(
                        media['filename'],
                        media['subfolder'],
                        media['type']
                    )
                    
                    # 決定保存路徑
                    if not file_name:
                        save_path = os.path.join(output_path, media['filename'])
                    else:
                        # 保持原始副檔名，若沒有則使用默認副檔名
                        base_name = os.path.splitext(media['filename'])[0]
                        extension = os.path.splitext(media['filename'])[1]
                        
                        if not extension and default_extension:
                            extension = default_extension
                        elif not extension:
                            extension = ''
                            
                        # 處理圖片的特殊情況（移除檔名中的副檔名部分）
                        if default_extension == '.png':
                            suffix = media['filename'].replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                            save_path = f'{output_path}/{suffix}_{file_name}{extension}'
                        else:
                            save_path = f'{output_path}/{base_name}_{file_name}{extension}'
                    
                    # 寫入文件
                    with open(save_path, 'wb') as f:
                        f.write(media_data)
                    saved_files.append(save_path)
            
            # 處理所有輸出節點
            for node_id, node_output in history['outputs'].items():
                # 處理圖片輸出
                if 'images' in node_output:
                    process_media_files(node_output['images'], '.png')
                
                # 處理 GIF 影片輸出
                if 'gifs' in node_output:
                    process_media_files(node_output['gifs'])
                
                # 處理影片輸出 (MP4, AVI 等)
                if 'videos' in node_output:
                    process_media_files(node_output['videos'])
            
            return True, saved_files
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False, []
        
    def identify_all_nodes(self, workflow: Dict) -> Dict[str, List[Dict]]:
        """
        識別工作流中所有節點並按類型分類
        """
        connections = self.analyze_node_connections(workflow)
        node_types = {}
        
        # 收集所有節點類型
        for node_id, node_data in workflow.items():
            # 如果 node_data 不是 dict，跳過
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get("class_type")
            if not class_type:
                continue
                
            if class_type not in node_types:
                node_types[class_type] = []
                
            # 收集節點資訊
            node_info = {
                "id": node_id,
                "data": node_data,
                "connections": connections.get(node_id, {}),
                "metadata": {}  # 用於存儲額外資訊
            }
            
            # 為特定類型節點添加額外資訊
            if class_type in ["PrimitiveString", "CLIPTextEncode"]:
                current_text = node_data.get('_meta', {}).get('title', '').lower()
                node_info["metadata"]["is_negative"] = 'negative' in current_text
            
            # 為所有節點添加 title 到 metadata（用於通用識別）
            title = node_data.get('_meta', {}).get('title', '')
            if title:
                node_info["metadata"]["title"] = title
                node_info["metadata"]["title_lower"] = title.lower()
            
            node_types[class_type].append(node_info)
            
        return node_types

    def update_node_inputs(self, workflow: Dict, node_id: str, 
                          updates: Dict[str, any]) -> None:
        """
        更新節點的輸入參數
        """
        if node_id in workflow:
            node = workflow[node_id]
            inputs = node.get("inputs", {})
            
            for key, value in updates.items():
                if key in inputs:
                    inputs[key] = value

        return workflow

    def process_workflow(self, workflow: Dict, updates: List[Dict], output_path: str, file_name = None, auto_close=True):
        """
        處理工作流，支援所有類型節點的更新
        
        updates 格式示例:
        [
            {
                "type": "CLIPTextEncode",  # 節點類型
                "node_index": 0,           # 第幾個同類型節點 (0-based)
                "is_negative": False,       # 可選的過濾條件
                "inputs": {                 # 要更新的輸入參數
                    "text": "new prompt"
                }
            },
            {
                "type": "KSampler",
                "node_index": 1,
                "inputs": {
                    "seed": 123456,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler"
                }
            },
            {
                "type": "VAEDecode",
                "node_index": 0,
                "inputs": {
                    "vae_name": "new_vae.safetensors"
                }
            }
        ]
        
        Args:
            workflow: 工作流配置
            updates: 節點更新配置列表
            output_path: 輸出路徑
            file_name: 檔案名稱
            auto_close: 是否自動關閉 WebSocket（預設 True，當需要連續處理多個工作流時設為 False）
        """
        try:
            # 只在 WebSocket 未連接時才建立新連線
            if not self.ws or not self.ws.connected:
                print("建立新的 WebSocket 連線")
                self.connect_websocket()
            
            os.makedirs(output_path, exist_ok=True)
            # 複製工作流以避免修改原始數據
            workflow_copy = json.loads(json.dumps(workflow))
            self.workflow = workflow_copy
            
            # 分析所有節點
            all_nodes = self.identify_all_nodes(workflow_copy)
            
            # 應用更新
            for update in updates:
                # 支持直接使用 node_id 更新
                if update.get("type") == "direct_update":
                    node_id = update.get("node_id")
                    node_inputs = update.get("inputs", {})
                    if node_id in workflow_copy:
                        workflow_copy = self.update_node_inputs(
                            workflow_copy,
                            node_id,
                            node_inputs
                        )
                    else:
                        print(f"Warning: Node ID '{node_id}' not found in workflow")
                    continue
                
                node_type = update.get("type")
                node_index = update.get("node_index", 0)
                node_inputs = update.get("inputs", {})
                
                if node_type not in all_nodes:
                    print(f"Warning: Node type '{node_type}' not found in workflow")
                    continue
                
                matching_nodes = all_nodes[node_type]
                
                # 應用額外的過濾條件（如果有的話）
                if "is_negative" in update:
                    matching_nodes = [
                        node for node in matching_nodes
                        if node["metadata"].get("is_negative") == update["is_negative"]
                    ]
                
                # 更新指定索引的節點
                if node_index < len(matching_nodes):
                    target_node = matching_nodes[node_index]
                    workflow_copy = self.update_node_inputs(
                        workflow_copy,
                        target_node["id"],
                        node_inputs
                    )
                else:
                    print(f"Warning: Node index {node_index} out of range for type '{node_type}'")

            # 執行工作流
            prompt_result = self.queue_prompt(workflow_copy)
            prompt_id = prompt_result['prompt_id']
            print(f"工作流已提交，prompt_id: {prompt_id}")
            
            # 等待完成
            self.wait_for_completion(prompt_id)
            print(f"工作流 {prompt_id} 執行完成")
            
            # 儲存並返回結果
            return self.save_results(prompt_id, output_path, file_name)
            
        except Exception as e:
            print(f"Error processing workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, []
        finally:
            # 只在 auto_close=True 時關閉 WebSocket
            if auto_close and self.ws and self.ws.connected:
                print("關閉 WebSocket 連線")
                self.ws.close()