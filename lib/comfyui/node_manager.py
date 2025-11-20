from typing import Dict, List, Any

class NodeManager:
    """管理 ComfyUI 工作流程中的節點操作"""
    
    # 內建策略配置
    BUILTIN_STRATEGIES = {
        'text': {
            'priority': [
                {'node_type': 'PrimitiveString', 'input_key': 'value', 'filter_key': 'is_negative'},
                {'node_type': 'CLIPTextEncode', 'input_key': 'text', 'filter_key': 'is_negative'}
            ]
        },
        'sampler': {
            'priority': [
                {'node_type': 'RandomNoise', 'input_key': 'noise_seed'},
                {'node_type': 'KSamplerAdvanced', 'input_key': 'noise_seed'},  # wan2.2 使用 KSamplerAdvanced
                {'node_type': 'KSampler', 'input_key': 'seed'},
                {'node_type': 'MMAudioSampler', 'input_key': 'seed'}
            ]
        }
    }
    
    @staticmethod
    def create_node_update(node_type: str, node_index: int, inputs: Dict[str, Any], **additional_params) -> Dict[str, Any]:
        """
        創建節點更新配置
        
        Args:
            node_type (str): 節點類型
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
            filters: 額外的過濾條件
            
        Returns:
            List[int]: 符合條件的節點索引列表
        """
        from lib.comfyui.websockets_api import ComfyUICommunicator
        
        communicator = ComfyUICommunicator()
        all_nodes = communicator.identify_all_nodes(workflow)
        matching_nodes = all_nodes.get(node_type, [])

        # 應用過濾條件
        if filters:
            filtered_nodes = []
            for node in matching_nodes:
                match = True
                for key, value in filters.items():
                    # 支持 title 的模糊匹配（不區分大小寫）
                    if key == "title":
                        node_title = node["metadata"].get("title_lower", "")
                        filter_title = str(value).lower()
                        if filter_title not in node_title:
                            match = False
                            break
                    else:
                        if node["metadata"].get(key) != value:
                            match = False
                            break
                if match:
                    filtered_nodes.append(node)
            matching_nodes = filtered_nodes
        
        return list(range(len(matching_nodes)))
    
    @staticmethod
    def generate_updates(workflow: Dict[str, Any], updates_config: List[Dict[str, Any]] = None, 
                        description: str = None, seed: int = None, use_noise_seed: bool = False, **additional_params) -> List[Dict[str, Any]]:
        """
        統一的節點更新生成方法
        
        Args:
            workflow (Dict): 工作流配置
            updates_config (List[Dict]): 自定義更新配置列表
            description (str): 文字描述（用於內建文字策略）
            seed (int): 隨機種子值（用於內建採樣器策略）
            use_noise_seed (bool): 是否使用 noise_seed 而不是 seed（用於 wan2.2 等工作流）
            additional_params: 其他額外參數
        
        Returns:
            List[Dict]: 節點更新配置列表
        """
        result_updates = []
        
        # 處理自定義配置
        if updates_config:
            result_updates.extend(
                NodeManager._generate_custom_updates(workflow, updates_config)
            )
        
        # 自動檢查是否已經有文字更新配置，避免重複
        has_text_update = False
        if updates_config:
            has_text_update = any(
                u.get('node_type') in ['PrimitiveString', 'CLIPTextEncode'] 
                for u in updates_config
            )
        
        # 處理內建文字策略（如果沒有自定義文字更新）
        if description is not None and not has_text_update:
            result_updates.extend(
                NodeManager._generate_builtin_text_updates(workflow, description, **additional_params)
            )
        
        # 處理內建採樣器策略
        if seed is not None:
            result_updates.extend(
                NodeManager._generate_builtin_sampler_updates(workflow, seed, use_noise_seed=use_noise_seed)
            )
        
        return result_updates
    
    @staticmethod
    def _generate_custom_updates(workflow: Dict[str, Any], updates_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成自定義節點更新配置"""
        result_updates = []
        
        for config in updates_config:
            # 如果直接指定了 node_id，使用直接更新方式
            if "node_id" in config:
                node_id = config.get("node_id")
                inputs = config.get("inputs", {})
                result_updates.append({
                    "type": "direct_update",
                    "node_id": node_id,
                    "inputs": inputs
                })
                continue
            
            node_type = config.get("node_type")
            node_index = config.get("node_index", 0)
            inputs = config.get("inputs", {})
            filter_params = config.get("filter", {})
            
            indices = NodeManager.get_node_indices(workflow, node_type, **filter_params)
            
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
    
    @staticmethod
    def _generate_builtin_text_updates(workflow: Dict[str, Any], description: str, **additional_params) -> List[Dict[str, Any]]:
        """生成內建文字策略的更新配置"""
        strategy = NodeManager.BUILTIN_STRATEGIES['text']
        filter_value = additional_params.get('is_negative', False)
        
        for priority_config in strategy['priority']:
            node_type = priority_config['node_type']
            input_key = priority_config['input_key']
            filter_key = priority_config['filter_key']
            
            # 使用過濾條件查找節點
            indices = NodeManager.get_node_indices(
                workflow, 
                node_type, 
                **{filter_key: filter_value}
            )
            
            if indices:
                return [
                    NodeManager.create_node_update(
                        node_type,
                        i,
                        {input_key: description},
                        **{filter_key: filter_value}
                    )
                    for i in indices
                ]
        
        return []
    
    @staticmethod
    def _generate_builtin_sampler_updates(workflow: Dict[str, Any], seed: int, use_noise_seed: bool = False) -> List[Dict[str, Any]]:
        """生成內建採樣器策略的更新配置
        
        會檢查所有類型的採樣器節點，並更新所有找到的節點的 seed/noise_seed。
        每個節點會使用不同的 seed（seed + node_index），確保生成的圖片都不同。
        
        Args:
            workflow: 工作流配置
            seed: 隨機種子值（基礎 seed）
            use_noise_seed: 如果為 True，優先使用 noise_seed 類型的節點（如 KSamplerAdvanced）
        """
        strategy = NodeManager.BUILTIN_STRATEGIES['sampler']
        result_updates = []
        
        # 如果指定使用 noise_seed，優先處理 noise_seed 類型的節點
        priority_order = strategy['priority']
        if use_noise_seed:
            # 重新排序，將 noise_seed 類型的節點放在前面
            noise_seed_configs = [c for c in priority_order if c['input_key'] == 'noise_seed']
            other_configs = [c for c in priority_order if c['input_key'] != 'noise_seed']
            priority_order = noise_seed_configs + other_configs
        
        for priority_config in priority_order:
            node_type = priority_config['node_type']
            input_key = priority_config['input_key']
            
            # 如果指定使用 noise_seed，跳過 seed 類型的節點
            if use_noise_seed and input_key == 'seed':
                continue
            
            indices = NodeManager.get_node_indices(workflow, node_type)
            
            if indices:
                # 為每個節點使用不同的 seed（seed + node_index），確保生成的圖片都不同
                result_updates.extend([
                    NodeManager.create_node_update(
                        node_type,
                        i,
                        {input_key: seed + i}
                    )
                    for i in indices
                ])
                # 如果找到 noise_seed 類型的節點，優先使用它
                if use_noise_seed and input_key == 'noise_seed':
                    break
        
        # 如果都沒找到，顯示警告
        if not result_updates:
            node_types = [config['node_type'] for config in strategy['priority']]
            print(f"Warning: None of the following node types found in the workflow for seed update: {', '.join(node_types)}")
        
        return result_updates
    
