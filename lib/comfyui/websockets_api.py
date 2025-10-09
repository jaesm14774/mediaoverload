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
    def __init__(self, host="host.docker.internal", port=8188, timeout=900):
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self.server_address = f"{host}:{port}"
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
        while True:
            # 檢查是否超時
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Operation timed out after {self.timeout} seconds")
            
            try:
                # 設置 websocket 接收超時時間為 1 秒，這樣可以定期檢查總時間
                self.ws.settimeout(5.0)
                out = self.ws.recv()
                
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break
                            
            except websocket.WebSocketTimeoutException:
                # 接收超時，繼續循環
                continue
            except Exception as e:
                # 其他錯誤，拋出異常
                raise Exception(f"Error while waiting for completion: {str(e)}")

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

    def process_workflow(self, workflow: Dict, updates: List[Dict], output_path: str, file_name = None):
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
        """
        try:
            # 為每個工作流程建立獨立的 WebSocket 連線
            self.connect_websocket()
            os.makedirs(output_path, exist_ok=True)
            # 複製工作流以避免修改原始數據
            workflow_copy = json.loads(json.dumps(workflow))
            self.workflow = workflow_copy
            
            # 分析所有節點
            all_nodes = self.identify_all_nodes(workflow_copy)
            
            # 應用更新
            for update in updates:
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
            
            # 等待完成
            self.wait_for_completion(prompt_id)
            
            # 儲存並返回結果
            return self.save_results(prompt_id, output_path, file_name)
            
        except Exception as e:
            print(f"Error processing workflow: {str(e)}")
            return False, []
        finally:
            # 確保 WebSocket 連線在結束時被關閉
            if self.ws and self.ws.connected:
                self.ws.close()