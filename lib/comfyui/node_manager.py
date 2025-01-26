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
    def generate_text_updates(description: str, indices: List[int], **additional_params) -> List[Dict[str, Any]]:
        """
        生成文字編碼節點的更新配置
        
        Args:
            description (str): 要編碼的文字描述
            indices (List[int]): 需要更新的節點索引列表
            additional_params: 其他額外參數
        
        Returns:
            List[Dict]: 文字編碼節點的更新配置列表
        """
        return [
            NodeManager.create_node_update(
                "CLIPTextEncode", 
                i, 
                {"text": description}, 
                **additional_params
            )
            for i in indices
        ]
    
    @staticmethod
    def generate_sampler_updates(seed: int, indices: List[int]) -> List[Dict[str, Any]]:
        """
        生成採樣器節點的更新配置
        
        Args:
            seed (int): 隨機種子值
            indices (List[int]): 需要更新的節點索引列表
        
        Returns:
            List[Dict]: 採樣器節點的更新配置列表
        """
        return [
            NodeManager.create_node_update(
                "KSampler", 
                i, 
                {"seed": seed}
            )
            for i in indices
        ]

    @staticmethod
    def generate_updates(description: str, seed: int, **additional_params) -> List[Dict[str, Any]]:
        """
        生成所有需要的節點更新配置
        
        Args:
            description (str): 文字描述
            seed (int): 隨機種子值
            additional_params: 其他額外參數
        
        Returns:
            List[Dict]: 所有節點的更新配置列表
        """
        # 生成文字編碼節點更新
        text_updates = NodeManager.generate_text_updates(
            description, 
            range(3),  # 假設有3個文字編碼節點
            **additional_params
        )
        
        # 生成採樣器節點更新
        sampler_updates = NodeManager.generate_sampler_updates(
            seed,
            range(3)  # 假設有3個採樣器節點
        )
        
        # 合併所有更新
        return text_updates + sampler_updates