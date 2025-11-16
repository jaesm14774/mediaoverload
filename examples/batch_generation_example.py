"""批次生成範例

展示如何使用 FlexibleGenerator 進行批次生成
適合需要一次生成多組不同 prompt 的情況

使用情境：
- 一次生成多個主題的內容
- 測試不同的 prompt 效果
- 批次生產內容
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.quick_draw.helpers.flexible_generator import FlexibleGenerator


def example_batch_images_basic():
    """範例 1: 基本批次圖片生成"""
    print("\n" + "="*60)
    print("範例 1: 基本批次圖片生成")
    print("="*60)

    generator = FlexibleGenerator()

    # 定義多組 prompt
    prompts = [
        {
            "prompt": "A peaceful morning scene with soft sunlight",
            "keywords": ["morning", "peaceful", "sunlight"]
        },
        {
            "prompt": "A rainy afternoon with cloudy sky",
            "keywords": ["rain", "afternoon", "cloudy"]
        },
        {
            "prompt": "A beautiful sunset with warm colors",
            "keywords": ["sunset", "warm", "beautiful"]
        },
        {
            "prompt": "A starry night with moon and stars",
            "keywords": ["night", "stars", "moon"]
        }
    ]

    # 批次生成
    results = generator.batch_generate(
        prompts=prompts,
        media_type="image",
        base_config={
            "character": "kirby",
            "num_images": 2
        }
    )

    print(f"\n✅ 批次生成完成！")
    print(f"📊 總共生成: {sum(len(r['result']['media_files']) for r in results)} 張圖片")

    return results


def example_batch_with_different_styles():
    """範例 2: 批次生成不同風格的圖片"""
    print("\n" + "="*60)
    print("範例 2: 批次生成不同風格的圖片")
    print("="*60)

    generator = FlexibleGenerator()

    # 定義不同風格的 prompt
    prompts = [
        {
            "prompt": "A cute character in minimalist style",
            "keywords": ["cute", "minimalist"],
            "style": "minimalist style, simple background, clean design"
        },
        {
            "prompt": "A cute character in fantasy style",
            "keywords": ["cute", "fantasy"],
            "style": "fantasy art style, magical atmosphere, detailed background"
        },
        {
            "prompt": "A cute character in cinematic style",
            "keywords": ["cute", "cinematic"],
            "style": "cinematic lighting, dramatic composition, movie-like"
        }
    ]

    results = generator.batch_generate(
        prompts=prompts,
        media_type="image",
        base_config={
            "character": "kirby",
            "num_images": 3
        }
    )

    print(f"\n✅ 批次生成完成！")

    return results


def example_batch_videos():
    """範例 3: 批次影片生成"""
    print("\n" + "="*60)
    print("範例 3: 批次影片生成")
    print("="*60)

    generator = FlexibleGenerator()

    # 定義影片 prompts
    prompts = [
        {
            "prompt": "Character flying through the sky",
            "keywords": ["flying", "sky"]
        },
        {
            "prompt": "Character walking in a forest",
            "keywords": ["walking", "forest"]
        },
        {
            "prompt": "Character playing in the water",
            "keywords": ["playing", "water"]
        }
    ]

    results = generator.batch_generate(
        prompts=prompts,
        media_type="video",
        base_config={
            "character": "kirby",
            "num_videos": 1
        }
    )

    print(f"\n✅ 批次生成完成！")
    print(f"📊 總共生成: {sum(len(r['result']['media_files']) for r in results)} 個影片")

    return results


def example_batch_with_different_characters():
    """範例 4: 批次生成不同角色"""
    print("\n" + "="*60)
    print("範例 4: 批次生成不同角色")
    print("="*60)

    generator = FlexibleGenerator()

    # 每組使用不同角色
    prompts = [
        {
            "prompt": "Character sleeping peacefully",
            "keywords": ["sleeping", "peaceful"],
            "character": "kirby"
        },
        {
            "prompt": "Character sleeping peacefully",
            "keywords": ["sleeping", "peaceful"],
            "character": "waddle dee"
        },
        {
            "prompt": "Character sleeping peacefully",
            "keywords": ["sleeping", "peaceful"],
            "character": "meta knight"
        }
    ]

    results = generator.batch_generate(
        prompts=prompts,
        media_type="image",
        base_config={
            "num_images": 2,
            "style": "cozy and warm atmosphere"
        }
    )

    print(f"\n✅ 批次生成完成！")

    return results


def example_batch_themed_content():
    """範例 5: 批次生成主題內容"""
    print("\n" + "="*60)
    print("範例 5: 批次生成主題內容（四季主題）")
    print("="*60)

    generator = FlexibleGenerator()

    # 四季主題
    prompts = [
        {
            "prompt": "Spring scene with cherry blossoms and butterflies",
            "keywords": ["spring", "cherry blossoms", "butterflies"],
            "style": "pastel colors, fresh and bright"
        },
        {
            "prompt": "Summer scene with beach and ocean waves",
            "keywords": ["summer", "beach", "ocean"],
            "style": "vibrant colors, sunny and warm"
        },
        {
            "prompt": "Autumn scene with falling leaves and warm colors",
            "keywords": ["autumn", "leaves", "warm colors"],
            "style": "orange and brown tones, cozy atmosphere"
        },
        {
            "prompt": "Winter scene with snow and ice crystals",
            "keywords": ["winter", "snow", "ice"],
            "style": "cool colors, peaceful and quiet"
        }
    ]

    results = generator.batch_generate(
        prompts=prompts,
        media_type="image",
        base_config={
            "character": "kirby",
            "num_images": 3
        }
    )

    print(f"\n✅ 四季主題批次生成完成！")

    return results


def example_batch_keyword_variations():
    """範例 6: 基於關鍵字變化的批次生成"""
    print("\n" + "="*60)
    print("範例 6: 基於關鍵字變化的批次生成")
    print("="*60)

    generator = FlexibleGenerator()

    # 基於不同關鍵字組合
    base_prompt = "Character in a magical environment"
    keyword_sets = [
        ["magical", "forest", "glowing mushrooms"],
        ["magical", "castle", "floating islands"],
        ["magical", "cave", "crystals"],
        ["magical", "garden", "fairy lights"]
    ]

    prompts = [
        {
            "prompt": f"{base_prompt} - {', '.join(keywords)}",
            "keywords": keywords
        }
        for keywords in keyword_sets
    ]

    results = generator.batch_generate(
        prompts=prompts,
        media_type="image",
        base_config={
            "character": "kirby",
            "num_images": 2,
            "style": "fantasy art style, dreamy and colorful"
        }
    )

    print(f"\n✅ 關鍵字變化批次生成完成！")

    return results


def main():
    """主函數"""
    print("\n" + "="*60)
    print("批次生成範例")
    print("一次生成多組不同的 prompt")
    print("="*60)

    examples = {
        '1': ('基本批次圖片生成', example_batch_images_basic),
        '2': ('批次生成不同風格', example_batch_with_different_styles),
        '3': ('批次影片生成', example_batch_videos),
        '4': ('批次生成不同角色', example_batch_with_different_characters),
        '5': ('批次生成主題內容（四季）', example_batch_themed_content),
        '6': ('基於關鍵字變化的批次生成', example_batch_keyword_variations),
    }

    print("\n請選擇要運行的範例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  q. 退出")

    choice = input("\n請輸入選項 (1-6/q): ").strip().lower()

    if choice == 'q':
        print("\n👋 再見！")
        return

    try:
        if choice in examples:
            # 運行單個範例
            name, func = examples[choice]
            func()
        else:
            print("\n❌ 無效的選項")
            return

        print("\n" + "="*60)
        print("✅ 範例執行完成！")
        print("="*60)
        print("\n💡 批次生成的優勢：")
        print("   - 自動化：一次設定多組 prompt，自動逐個生成")
        print("   - 組織化：每組結果自動保存在獨立子目錄")
        print("   - 高效率：適合需要生成大量內容的場景")
        print("   - 可追蹤：返回詳細的批次結果資訊")
        print("\n💡 使用技巧：")
        print("   - 使用 base_config 設定所有批次共用的參數")
        print("   - 每個 prompt 可以有自己獨特的參數（會覆蓋 base_config）")
        print("   - 適合測試不同 prompt、style、keywords 的效果")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
