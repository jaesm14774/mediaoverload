"""Text2Image2Image 使用範例

展示如何使用 Text2Image2ImageStrategy 進行兩階段生成：
1. 第一階段：Text to Image 生成多張圖片
2. 篩選階段：自動篩選符合描述的圖片
3. 第二階段：對篩選後的圖片進行 Image to Image 二次生成

使用前請確保：
1. ComfyUI 已啟動並運行在 8188 端口
2. 環境變數已配置（media_overload.env）
3. 有可用的視覺模型（OpenRouter/Gemini）用於圖文匹配分析
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
import os


def example_text2image2image_basic():
    """範例 1: 基本的 Text2Image2Image 生成"""
    print("\n" + "="*60)
    print("範例 1: 基本的 Text2Image2Image 生成")
    print("="*60)
    
    # 創建視覺管理器（用於圖文匹配分析）
    vision_manager = VisionManagerBuilder() \
        .with_vision_model('openrouter') \
        .with_text_model('openrouter') \
        .with_random_models(True) \
        .build()
    
    # 配置參數
    config = GenerationConfig(
        generation_type='text2image2image',
        character='kirby',
        prompt='Kirby eating ramen with chopsticks',
        workflow_path='configs/workflow/nova-anime-xl.json',  # 第一階段工作流
        output_dir='output_media/t2i2i_output',
        image_system_prompt='stable_diffusion_prompt',
        similarity_threshold=0.9,  # 第一階段篩選閾值
        additional_params={
            'strategies': {
                'text2image2image': {
                    'first_stage': {
                        'images_per_description': 4  # 第一階段生成 4 張圖片
                    },
                    'second_stage': {
                        'images_per_input': 1,  # 第二階段每個輸入生成 1 張
                        'denoise': 0.6,  # denoise 權重
                        'i2i_workflow_path': 'configs/workflow/example/image_to_image.json'  # 第二階段工作流
                    }
                }
            }
        }
    )
    
    # 創建策略並執行
    strategy = StrategyFactory.get_strategy(
        'text2image2image',
        vision_manager=vision_manager
    )
    strategy.load_config(config)
    
    print(f"\n📝 提示詞: {config.prompt}")
    print(f"📝 第一階段生成數量: {config.additional_params['strategies']['text2image2image']['first_stage']['images_per_description']}")
    print(f"📝 相似度閾值: {config.similarity_threshold}")
    print(f"📝 第二階段 denoise: {config.additional_params['strategies']['text2image2image']['second_stage']['denoise']}")
    print(f"📂 輸出目錄: {config.output_dir}")
    
    # 生成描述
    print("\n🔄 步驟 1: 生成圖片描述...")
    strategy.generate_description()
    print(f"📝 生成的描述: {strategy.descriptions[0] if strategy.descriptions else 'N/A'}")
    
    # 生成圖片（包含兩階段）
    print("\n🔄 步驟 2: 開始兩階段生成...")
    strategy.generate_media()
    
    # 分析第二階段結果
    print("\n🔄 步驟 3: 分析第二階段結果...")
    strategy.analyze_media_text_match(similarity_threshold=0.8)
    
    print(f"\n✅ 生成完成！")
    print(f"📊 第一階段生成: {len(strategy.first_stage_images)} 張圖片通過篩選")
    print(f"📊 第二階段結果: {len(strategy.filter_results)} 張圖片通過匹配度檢查")
    
    return strategy


def example_text2image2image_custom_params():
    """範例 2: 自定義參數的 Text2Image2Image"""
    print("\n" + "="*60)
    print("範例 2: 自定義參數的 Text2Image2Image")
    print("="*60)
    
    vision_manager = VisionManagerBuilder() \
        .with_vision_model('openrouter') \
        .with_text_model('openrouter') \
        .with_random_models(True) \
        .build()
    
    config = GenerationConfig(
        generation_type='text2image2image',
        character='kirby',
        prompt='Kirby floating in space with stars',
        workflow_path='configs/workflow/nova-anime-xl.json',
        output_dir='output_media/t2i2i_custom',
        image_system_prompt='stable_diffusion_prompt',
        similarity_threshold=0.85,  # 較低的閾值，保留更多圖片
        additional_params={
            'strategies': {
                'text2image2image': {
                    'first_stage': {
                        'images_per_description': 6,  # 第一階段生成更多圖片
                    },
                    'second_stage': {
                        'images_per_input': 2,  # 第二階段每個輸入生成 2 張
                        'denoise': 0.55,  # 較低的 denoise，更接近原圖
                        'i2i_workflow_path': 'configs/workflow/example/image_to_image.json'
                    }
                }
            }
        }
    )
    
    strategy = StrategyFactory.get_strategy(
        'text2image2image',
        vision_manager=vision_manager
    )
    strategy.load_config(config)
    
    print(f"\n📝 自定義參數:")
    print(f"   - 第一階段生成: {config.additional_params['strategies']['text2image2image']['first_stage']['images_per_description']} 張")
    print(f"   - 相似度閾值: {config.similarity_threshold}")
    print(f"   - 第二階段每個輸入: {config.additional_params['strategies']['text2image2image']['second_stage']['images_per_input']} 張")
    print(f"   - Denoise: {config.additional_params['strategies']['text2image2image']['second_stage']['denoise']}")
    
    strategy.generate_description()
    strategy.generate_media()
    strategy.analyze_media_text_match(similarity_threshold=0.8)
    
    print(f"\n✅ 生成完成！")
    print(f"📊 第一階段通過篩選: {len(strategy.first_stage_images)} 張")
    print(f"📊 第二階段通過檢查: {len(strategy.filter_results)} 張")
    
    return strategy


def example_text2image2image_two_character():
    """範例 3: 雙角色互動的 Text2Image2Image"""
    print("\n" + "="*60)
    print("範例 3: 雙角色互動的 Text2Image2Image")
    print("="*60)
    
    vision_manager = VisionManagerBuilder() \
        .with_vision_model('openrouter') \
        .with_text_model('openrouter') \
        .with_random_models(True) \
        .build()
    
    config = GenerationConfig(
        generation_type='text2image2image',
        character='kirby',
        secondary_character='waddle dee',  # 次要角色
        prompt='friendship and adventure',
        workflow_path='configs/workflow/nova-anime-xl.json',
        output_dir='output_media/t2i2i_two_char',
        image_system_prompt='two_character_interaction_generate_system_prompt',  # 使用雙角色提示詞
        similarity_threshold=0.9,
        additional_params={
            'strategies': {
                'text2image2image': {
                    'first_stage': {
                        'images_per_description': 4
                    },
                    'second_stage': {
                        'images_per_input': 1,
                        'denoise': 0.6,
                        'i2i_workflow_path': 'configs/workflow/example/image_to_image.json'
                    }
                }
            }
        }
    )
    
    strategy = StrategyFactory.get_strategy(
        'text2image2image',
        vision_manager=vision_manager
    )
    strategy.load_config(config)
    
    print(f"\n📝 主角色: {config.character}")
    print(f"📝 次要角色: {config.secondary_character}")
    print(f"📝 提示詞: {config.prompt}")
    
    strategy.generate_description()
    print(f"📝 生成的描述: {strategy.descriptions[0] if strategy.descriptions else 'N/A'}")
    
    strategy.generate_media()
    strategy.analyze_media_text_match(similarity_threshold=0.8)
    
    print(f"\n✅ 生成完成！")
    print(f"📊 第一階段通過篩選: {len(strategy.first_stage_images)} 張")
    print(f"📊 第二階段通過檢查: {len(strategy.filter_results)} 張")
    
    return strategy


def main():
    """主函數"""
    print("\n" + "="*60)
    print("Text2Image2Image 使用範例")
    print("兩階段生成：Text2Image -> 篩選 -> Image2Image")
    print("="*60)
    
    examples = {
        '1': ('基本的 Text2Image2Image 生成', example_text2image2image_basic),
        '2': ('自定義參數', example_text2image2image_custom_params),
        '3': ('雙角色互動', example_text2image2image_two_character),
    }
    
    print("\n請選擇要運行的範例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. 退出")
    
    choice = input("\n請輸入選項 (1-3/q): ").strip().lower()
    
    if choice == 'q':
        print("\n👋 再見！")
        return
    
    try:
        if choice in examples:
            name, func = examples[choice]
            func()
        else:
            print("\n❌ 無效的選項")
            return
        
        print("\n" + "="*60)
        print("✅ 範例執行完成！")
        print("="*60)
        print("\n💡 提示：")
        print("   - 第一階段圖片保存在: output_dir/first_stage/")
        print("   - 第二階段圖片保存在: output_dir/second_stage/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

