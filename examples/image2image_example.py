"""Image to Image 使用範例

展示如何使用 Image2ImageStrategy 直接對現有圖片進行 image to image 生成。

使用前請確保：
1. ComfyUI 已啟動並運行在 8188 端口
2. 有一張輸入圖片可用於測試
3. 環境變數已配置（media_overload.env）
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
import os


def example_image2image_basic():
    """範例 1: 基本的 Image to Image 生成"""
    print("\n" + "="*60)
    print("範例 1: 基本的 Image to Image 生成")
    print("="*60)
    
    # 配置參數
    config = GenerationConfig(
        generation_type='image2image',
        character='kirby',
        prompt='A beautiful sunset scene with vibrant colors',  # 可選：提示詞
        input_image_path='output_media/1.png',  # ⚠️ 請替換為實際存在的圖片路徑
        workflow_path='configs/workflow/example/image_to_image.json',
        output_dir='output_media/i2i_output',
        image_system_prompt='stable_diffusion_prompt',
        additional_params={
            'image': {
                'denoise': 0.6,  # denoise 權重 (0.5-0.7)
                'images_per_input': 2  # 每個輸入圖片生成 2 張
            }
        }
    )
    
    # 檢查輸入圖片是否存在
    if not os.path.exists(config.input_image_path):
        print(f"\n⚠️ 警告：找不到輸入圖片: {config.input_image_path}")
        print("請先運行 text2image 生成一些圖片，或提供一個存在的圖片路徑")
        return None
    
    # 創建策略並執行
    strategy = StrategyFactory.get_strategy('image2image')
    strategy.load_config(config)
    
    print(f"\n📝 輸入圖片: {config.input_image_path}")
    print(f"📝 提示詞: {config.prompt}")
    print(f"📝 Denoise 權重: {config.additional_params['image']['denoise']}")
    print(f"📂 輸出目錄: {config.output_dir}")
    
    # 生成描述（可選）
    strategy.generate_description()
    
    # 生成圖片
    strategy.generate_media()
    
    # 分析結果
    strategy.analyze_media_text_match(similarity_threshold=0.7)
    
    print(f"\n✅ 生成完成！")
    print(f"📊 篩選結果: {len(strategy.filter_results)} 張圖片通過匹配度檢查")
    
    return strategy


def example_image2image_different_denoise():
    """範例 2: 使用不同的 denoise 權重生成多張圖片"""
    print("\n" + "="*60)
    print("範例 2: 使用不同的 denoise 權重")
    print("="*60)
    
    input_image = 'output_media/1.png'  # ⚠️ 請替換為實際存在的圖片路徑
    
    if not os.path.exists(input_image):
        print(f"\n⚠️ 警告：找不到輸入圖片: {input_image}")
        return None
    
    # 測試不同的 denoise 值
    denoise_values = [0.5, 0.6, 0.7]
    
    for denoise in denoise_values:
        print(f"\n🔄 使用 denoise={denoise} 生成圖片...")
        
        config = GenerationConfig(
            generation_type='image2image',
            character='kirby',
            prompt='Enhanced version with more details',
            input_image_path=input_image,
            workflow_path='configs/workflow/example/image_to_image.json',
            output_dir=f'output_media/i2i_denoise_{denoise}',
            additional_params={
                'image': {
                    'denoise': denoise,
                    'images_per_input': 1
                }
            }
        )
        
        strategy = StrategyFactory.get_strategy('image2image')
        strategy.load_config(config)
        strategy.generate_description()
        strategy.generate_media()
        
        print(f"✅ denoise={denoise} 完成")
    
    print(f"\n✅ 所有 denoise 值測試完成！")
    print("💡 提示：denoise 值越小，生成的圖片越接近原圖")


def example_image2image_extract_description():
    """範例 3: 從圖片中提取描述後再生成"""
    print("\n" + "="*60)
    print("範例 3: 從圖片中提取描述")
    print("="*60)
    
    input_image = 'output_media/1.png'  # ⚠️ 請替換為實際存在的圖片路徑
    
    if not os.path.exists(input_image):
        print(f"\n⚠️ 警告：找不到輸入圖片: {input_image}")
        return None
    
    config = GenerationConfig(
        generation_type='image2image',
        character='kirby',
        input_image_path=input_image,
        workflow_path='configs/workflow/example/image_to_image.json',
        output_dir='output_media/i2i_extracted',
        extract_description=True,  # 從圖片中提取描述
        additional_params={
            'image': {
                'denoise': 0.6,
                'images_per_input': 1
            }
        }
    )
    
    strategy = StrategyFactory.get_strategy('image2image')
    strategy.load_config(config)
    
    # 生成描述（會從圖片中提取）
    print("\n📝 正在從圖片中提取描述...")
    strategy.generate_description()
    print(f"📝 提取的描述: {strategy.descriptions[0] if strategy.descriptions else 'N/A'}")
    
    # 生成圖片
    strategy.generate_media()
    
    print(f"\n✅ 生成完成！")


def main():
    """主函數"""
    print("\n" + "="*60)
    print("Image to Image 使用範例")
    print("="*60)
    
    examples = {
        '1': ('基本的 Image to Image 生成', example_image2image_basic),
        '2': ('使用不同的 denoise 權重', example_image2image_different_denoise),
        '3': ('從圖片中提取描述', example_image2image_extract_description),
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
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

