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
        # 嘗試尋找 RandomNoise 節點
        target_node_type = "RandomNoise"
        seed_key_name = 'noise_seed'
        indices = NodeManager.get_node_indices(workflow, target_node_type)
        
        # 如果沒有找到 RandomNoise，則尋找 KSampler 節點
        if not indices:
            target_node_type = "KSampler"
            seed_key_name = 'seed'
            indices = NodeManager.get_node_indices(workflow, target_node_type)
            
        # 如果兩者都沒找到，返回空列表或拋出錯誤？目前返回空列表
        if not indices:
            print(f"Warning: Neither RandomNoise nor KSampler node found in the workflow for seed update.")
            return []

        return [
            NodeManager.create_node_update(
                target_node_type, 
                i, 
                {seed_key_name: seed}
            )
            for i in indices
        ]

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