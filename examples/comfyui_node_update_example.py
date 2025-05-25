"""
ComfyUI 通用節點更新範例
展示如何更新工作流中的任意節點
"""

from lib.comfyui.websockets_api import ComfyUICommunicator
from lib.comfyui.node_manager import NodeManager
from lib.media_auto.process import WobbuffetProcess
import json
import os
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.strategies.image_strategies import Text2ImageStrategy

# 範例 1: 直接使用 ComfyUICommunicator 的 process_workflow
def example_direct_update():
    """直接使用 process_workflow 方法更新任意節點"""
    
    # 載入工作流
    with open('/app/configs/workflow/nova-anime-xl.json', 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # 建立通訊器
    communicator = ComfyUICommunicator()
    communicator.connect_websocket()
    
    # 準備更新 - 可以更新任何類型的節點
    updates = [
        # 更新圖片寬度
        {
            "type": "PrimitiveInt",
            "node_index": 0,  # 第一個 PrimitiveInt 節點（寬度）
            "inputs": {"value": 1280}
        },
        # 更新圖片高度
        {
            "type": "PrimitiveInt",
            "node_index": 1,  # 第二個 PrimitiveInt 節點（高度）
            "inputs": {"value": 720}
        },
        # 更新第一組的正面提示詞
        {
            "type": "PrimitiveString",
            "node_index": 0,
            "is_negative": False,
            "inputs": {"value": "beautiful anime girl, high quality"}
        },
        # 更新第二個 KSampler 的參數
        {
            "type": "KSampler",
            "node_index": 1,  # 工作流中的第二個 KSampler
            "inputs": {
                "seed": 123456,
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras"
            }
        },
        # 更新第三個 Checkpoint Loader
        {
            "type": "CheckpointLoaderSimple",
            "node_index": 2,
            "inputs": {
                "ckpt_name": "sdxl\\different_model.safetensors"
            }
        }
    ]
    
    # 執行工作流
    success, saved_files = communicator.process_workflow(
        workflow=workflow,
        updates=updates,
        output_path="./output_image",
        file_name="custom_update_test"
    )
    
    communicator.ws.close()
    return success, saved_files


# 範例 2: 使用 NodeManager 的通用更新方法
def example_node_manager_generic():
    """使用 NodeManager 的 generate_generic_updates 方法"""
    
    # 載入工作流
    with open('/app/configs/workflow/example/flux_dev.json', 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    node_manager = NodeManager()
    
    # 定義更新配置
    updates_config = [
        # 更新寬度和高度
        {
            "node_type": "EmptySD3LatentImage",
            "node_index": 0,
            "inputs": {"width": 512, "height":512}
        },
        # 使用過濾條件更新負面提示詞
        {
            "node_type": "CLIPTextEncode",
            "node_index": 0,
            "filter": {"is_negative": False},
            "inputs": {"text": "Pandas climb mountain"}
        },
        # 更新所有cfg 值
        {
            "node_type": "FluxGuidance",
            # 不指定 node_index，會更新所有 KSampler
            "inputs": {"guidance": 4.5}
        },
        # 更新第二個使用過濾條件更新負面提示詞
        {
            "node_type": "CLIPTextEncode",
            "node_index": 1,
            "filter": {"is_negative": False},
            "inputs": {"text": "Mao test"}
        },
    ]
    
    # 生成更新列表
    updates = node_manager.generate_generic_updates(workflow, updates_config)
    
    # 執行工作流
    communicator = ComfyUICommunicator()
    communicator.connect_websocket()
    
    success, saved_files = communicator.process_workflow(
        workflow=workflow,
        updates=updates,
        output_path="./output_image",
        file_name="generic_update_test"
    )
    
    communicator.ws.close()
    return success, saved_files


# 範例 3: 在角色處理流程中使用自定義節點更新
def example_character_with_custom_updates():
    """在角色處理流程中使用自定義節點更新"""
    
    # 創建角色實例
    character = WobbuffetProcess()
    character.workflow_path = '/app/configs/workflow/example/image_to_image.json'
    character.config = character.get_default_config()
    
    # 準備自定義的節點更新配置
    custom_node_updates = [
        # 設定自定義解析度
        {
            "node_type": "PrimitiveInt",
            "node_index": 0,
            "inputs": {"value": 768}  # 寬度
        },
        {
            "node_type": "PrimitiveInt", 
            "node_index": 1,
            "inputs": {"value": 1024}  # 高度
        },
        # 更新第一個 KSampler 的參數
        {
            "node_type": "KSampler",
            "node_index": 0,
            "inputs": {
                "steps": 15,
                "cfg": 9.0,
                "sampler_name": "dpm_fast",
                "scheduler":"karras"
            }
        },
        # 如果有 LoadImage 節點，更新圖片
        {
            "node_type": "LoadImage",
            "node_index": 0,
            "inputs": {"image": "Mario_Cat.png"}
        }
    ]
    
    # 將自定義更新加入 additional_params
    character.config.additional_params['custom_node_updates'] = custom_node_updates
    
    # 獲取生成配置
    generation_config = character.get_generation_config(
        prompt="wobbuffet wear mario hat in the summer beach and wobbuffet play with wobbuffet"
    )
    generation_config['output_dir'] = os.path.join(generation_config['output_dir'], generation_config['character'])
    # 這裡可以繼續執行生成流程
    strategy = Text2ImageStrategy()
    strategy.load_config(GenerationConfig(**generation_config))
    strategy.generate_description()
    strategy.generate_image()
    
    return generation_config


# 範例 4: 動態更新多個相同類型的節點
def example_update_multiple_nodes():
    """展示如何精確更新多個相同類型的節點"""
    
    with open('/app/configs/workflow/example/flux_dev.json', 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # 這個工作流有多個 CLIPTextEncode 節點
    # 我們想要分別更新它們
    updates = [{
        'type': 'CLIPTextEncode', 'node_index': 0, 'inputs': {'text': 'test mao 1'}}, 
        {'type': 'CLIPTextEncode', 'node_index': 1, 'inputs': {'text': 'test mao 2'}}
    ]
    
    communicator = ComfyUICommunicator()
    communicator.connect_websocket()
    success, saved_files = communicator.process_workflow(
        workflow=workflow,
        updates=updates,
        output_path="./output_image",
        file_name="multi_node_test"
    )
    
    communicator.ws.close()
    return success, saved_files


if __name__ == "__main__":
    # 執行範例
    # print("範例 1: 直接更新")
    # example_direct_update()
    
    # print("\n範例 2: 使用 NodeManager 通用方法")
    # example_node_manager_generic()
    
    # print("\n範例 3: 角色處理流程中的自定義更新")
    # config = example_character_with_custom_updates()
    # print(f"生成配置: {config}")
    
    print("\n範例 4: 更新多個相同類型節點")
    example_update_multiple_nodes() 