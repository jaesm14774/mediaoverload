from typing import Dict, List, Any

class NodeManager:
    """管理 ComfyUI 工作流程中的節點操作"""
    
    @staticmethod
    def create_node_update(node_type: str, node_index: int, inputs: Dict[str, Any], **additional_params) -> Dict[str, Any]:
        """
        創建節點更新配置
        
        Args:
            node_type (str): 節點類型，例如 'CLIPTextEncode', 'KSampler' 等
            node_index (int): 節點在工作流中的索引
            inputs (dict): 節點的輸入參數
            additional_params: 其他額外參數
        
        Returns:
            Dict: 節點更新配置
        """
        return {
            "type": node_type,
            "node_index": node_index,
            "inputs": inputs,
            **additional_params
        }
    
    @staticmethod
    def get_node_indices(workflow: Dict[str, Any], node_type: str, **filters) -> List[int]:
        """
        獲取指定類型節點的索引列表
        
        Args:
            workflow (Dict): 工作流配置
            node_type (str): 節點類型
            filters: 額外的過濾條件，例如 is_negative=False
            
        Returns:
            List[int]: 符合條件的節點索引列表
        """
        from lib.comfyui.websockets_api import ComfyUICommunicator
        
        # 使用 ComfyUICommunicator 的方法分析節點
        communicator = ComfyUICommunicator()
        all_nodes = communicator.identify_all_nodes(workflow)
        
        # 獲取指定類型的節點
        matching_nodes = all_nodes.get(node_type, [])

        # 應用過濾條件
        if filters:
            filtered_nodes = []
            for node in matching_nodes:
                if all(
                    node["metadata"].get(key) == value 
                    for key, value in filters.items()
                ):
                    filtered_nodes.append(node)
            matching_nodes = filtered_nodes
        
        # 返回節點索引列表
        return list(range(len(matching_nodes)))
    
    @staticmethod
    def generate_text_updates(workflow: Dict[str, Any], description: str, **additional_params) -> List[Dict[str, Any]]:
        """
        生成文字編碼節點的更新配置
        先以PrimitiveString節點 當作text 的統一輸入，若無則再以CLIPTextEncode 當作text 的統一輸入
        
        Args:
            workflow (Dict): 工作流配置
            description (str): 要編碼的文字描述
            additional_params: 其他額外參數，例如 is_negative=False
        
        Returns:
            List[Dict]: 文字編碼節點的更新配置列表
        """
        text_node_name = 'PrimitiveString'
        text_node_key = 'value'
        indices = NodeManager.get_node_indices(
            workflow, 
            text_node_name, 
            **{'is_negative' : additional_params.get('is_negative', False)}
        )
        if len(indices) == 0:
            indices = NodeManager.get_node_indices(
                workflow,
                'CLIPTextEncode',
                **{'is_negative' : additional_params.get('is_negative', False)}
            )
            text_node_name = 'CLIPTextEncode'
            text_node_key = 'text' 
        return [
            NodeManager.create_node_update(
                text_node_name, 
                i, 
                {text_node_key: description}, 
                **additional_params
            )
            for i in indices
        ]
    
    @staticmethod
    def generate_sampler_updates(workflow: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        """
        生成採樣器或隨機噪聲節點的更新配置
        優先更新 RandomNoise 節點的 seed，如果不存在則更新 KSampler 節點的 seed。
        
        Args:
            workflow (Dict): 工作流配置
            seed (int): 隨機種子值
        
        Returns:
            List[Dict]: 目標節點的更新配置列表
        """
        # 配置採樣器節點的優先級和對應的 seed 參數名稱
        sampler_configs = [
            {"node_type": "RandomNoise", "seed_key": "noise_seed"},
            {"node_type": "KSampler", "seed_key": "seed"},
            {"node_type": "MMAudioSampler", "seed_key": "seed"}
        ]
        
        # 依序嘗試每個配置，直到找到符合的節點
        for config in sampler_configs:
            indices = NodeManager.get_node_indices(workflow, config["node_type"])
            if indices:
                return [
                    NodeManager.create_node_update(
                        config["node_type"], 
                        i, 
                        {config["seed_key"]: seed}
                    )
                    for i in indices
                ]
        
        # 如果都沒找到，顯示警告
        node_types = [config["node_type"] for config in sampler_configs]
        print(f"Warning: None of the following node types found in the workflow for seed update: {', '.join(node_types)}")
        return []

    @staticmethod
    def generate_updates(workflow: Dict[str, Any], description: str, seed: int, **additional_params) -> List[Dict[str, Any]]:
        """
        生成所有需要的節點更新配置
        
        Args:
            workflow (Dict): 工作流配置
            description (str): 文字描述
            seed (int): 隨機種子值
            additional_params: 其他額外參數
        
        Returns:
            List[Dict]: 所有節點的更新配置列表
        """
        # 生成文字編碼節點更新
        text_updates = NodeManager.generate_text_updates(
            workflow,
            description, 
            **additional_params
        )
        
        # 生成採樣器節點更新
        sampler_updates = NodeManager.generate_sampler_updates(
            workflow,
            seed
        )
        
        # 合併所有更新
        return text_updates + sampler_updates
    
    @staticmethod
    def generate_generic_updates(workflow: Dict[str, Any], updates_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        生成通用的節點更新配置，支援任何類型的節點
        
        Args:
            workflow (Dict): 工作流配置
            updates_config (List[Dict]): 更新配置列表，每個配置包含：
                - node_type: 節點類型
                - node_index: 節點索引（可選，預設為0）
                - inputs: 要更新的輸入參數
                - filter: 過濾條件字典（可選）
        
        Returns:
            List[Dict]: 節點更新配置列表
        
        Example:
            updates_config = [
                {
                    "node_type": "PrimitiveInt",
                    "node_index": 0,  # 第一個 PrimitiveInt 節點
                    "inputs": {"value": 1280}
                },
                {
                    "node_type": "PrimitiveString",
                    "filter": {"is_negative": True},  # 使用過濾條件
                    "inputs": {"value": "negative prompt"}
                },
                {
                    "node_type": "KSampler",
                    "node_index": 1,  # 第二個 KSampler
                    "inputs": {
                        "seed": 12345,
                        "steps": 30,
                        "cfg": 7.5
                    }
                }
            ]
        """
        result_updates = []
        
        for config in updates_config:
            node_type = config.get("node_type")
            node_index = config.get("node_index", 0)
            inputs = config.get("inputs", {})
            filter_params = config.get("filter", {})
            
            # 獲取符合條件的節點索引
            indices = NodeManager.get_node_indices(workflow, node_type, **filter_params)
            
            # 如果指定了 node_index，只更新該索引的節點
            if indices and node_index < len(indices):
                result_updates.append(
                    NodeManager.create_node_update(
                        node_type,
                        indices[node_index],
                        inputs,
                        **filter_params
                    )
                )
            elif indices:
                # 如果沒有指定 node_index，更新所有符合條件的節點
                for idx in indices:
                    result_updates.append(
                        NodeManager.create_node_update(
                            node_type,
                            idx,
                            inputs,
                            **filter_params
                        )
                    )
        
        return result_updates