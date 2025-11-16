"""彈性生成範例

展示如何使用 FlexibleGenerator 來輕鬆生成圖片和影片
支援自訂 prompt 和 keywords，無需深入了解內部架構

使用前請確保：
1. ComfyUI 已啟動並運行
2. 環境變數已配置（media_overload.env）
3. 資料庫連接正常
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.quick_draw.helpers.flexible_generator import FlexibleGenerator


def example_simple_image_generation():
    """範例 1: 簡單的圖片生成"""
    print("\n" + "="*60)
    print("範例 1: 簡單的圖片生成")
    print("="*60)

    generator = FlexibleGenerator()

    # 使用自訂 prompt 生成圖片
    result = generator.generate_images(
        prompt="A peaceful sunset scene with vibrant orange and pink colors",
        keywords=["sunset", "peaceful", "vibrant colors"],
        character="kirby",
        num_images=3
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")
    print(f"📝 描述: {result['descriptions'][0] if result['descriptions'] else 'N/A'}")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")

    return result


def example_styled_generation():
    """範例 2: 帶風格的圖片生成"""
    print("\n" + "="*60)
    print("範例 2: 帶風格的圖片生成")
    print("="*60)

    generator = FlexibleGenerator()

    result = generator.generate_images(
        prompt="A magical forest with glowing mushrooms and fireflies",
        keywords=["forest", "magical", "glowing", "fireflies"],
        character="kirby",
        style="fantasy art style, dreamy atmosphere, soft lighting",
        num_images=4,
        output_subdir="magical_forest"
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")
    print(f"📂 保存在子目錄: magical_forest/")

    return result


def example_two_character_generation():
    """範例 3: 雙角色互動圖片生成"""
    print("\n" + "="*60)
    print("範例 3: 雙角色互動圖片生成")
    print("="*60)

    generator = FlexibleGenerator()

    result = generator.generate_images(
        prompt="Two friends sharing a happy moment together",
        keywords=["friendship", "happy", "together"],
        character="kirby",
        secondary_character="waddle dee",
        style="warm and cozy atmosphere",
        num_images=3,
        image_system_prompt="two_character_interaction_generate_system_prompt",
        output_subdir="friendship"
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")

    return result


def example_video_generation():
    """範例 4: 影片生成"""
    print("\n" + "="*60)
    print("範例 4: 影片生成")
    print("="*60)

    generator = FlexibleGenerator()

    result = generator.generate_videos(
        prompt="Kirby flying through a beautiful starry night sky",
        keywords=["flying", "stars", "night sky"],
        character="kirby",
        style="cinematic, smooth motion",
        num_videos=2,
        output_subdir="flying_video"
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 個影片")
    print(f"📂 影片路徑:")
    for video in result['media_files']:
        print(f"   - {video}")

    return result


def example_custom_workflow():
    """範例 5: 使用自訂工作流"""
    print("\n" + "="*60)
    print("範例 5: 使用自訂工作流")
    print("="*60)

    generator = FlexibleGenerator()

    result = generator.generate_images(
        prompt="A cute character in anime style with detailed background",
        keywords=["anime", "cute", "detailed"],
        character="kirby",
        num_images=2,
        workflow="nova-anime-xl",  # 指定工作流
        output_subdir="custom_workflow"
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")

    return result


def example_with_configbuilder():
    """範例 6: 結合 ConfigBuilder（進階用法）"""
    print("\n" + "="*60)
    print("範例 6: 結合 ConfigBuilder（進階用法）")
    print("="*60)

    from examples.quick_draw.helpers import ConfigBuilder

    generator = FlexibleGenerator()

    # 使用 ConfigBuilder 建立更複雜的配置
    config = ConfigBuilder() \
        .with_character('kirby') \
        .with_prompt('epic adventure scene with dramatic lighting') \
        .with_keywords(['adventure', 'epic', 'dramatic']) \
        .with_style('cinematic, high contrast') \
        .with_workflow('configs/workflow/nova-anime-xl.json') \
        .with_output_dir('output_media/epic_adventure') \
        .with_images_per_description(3) \
        .with_image_system_prompt('cinematic_stable_diffusion_prompt') \
        .build()

    # 使用配置生成
    result = generator.generate_from_config(config)

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")

    return result


def example_minimal():
    """範例 7: 最簡化使用（只提供 prompt）"""
    print("\n" + "="*60)
    print("範例 7: 最簡化使用")
    print("="*60)

    generator = FlexibleGenerator()

    # 最簡單的使用方式
    result = generator.generate_images(
        prompt="A happy cat playing in the garden"
    )

    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")

    return result


def main():
    """主函數"""
    print("\n" + "="*60)
    print("彈性生成範例")
    print("使用 FlexibleGenerator 輕鬆生成圖片和影片")
    print("="*60)

    examples = {
        '1': ('簡單的圖片生成', example_simple_image_generation),
        '2': ('帶風格的圖片生成', example_styled_generation),
        '3': ('雙角色互動圖片生成', example_two_character_generation),
        '4': ('影片生成', example_video_generation),
        '5': ('使用自訂工作流', example_custom_workflow),
        '6': ('結合 ConfigBuilder（進階）', example_with_configbuilder),
        '7': ('最簡化使用', example_minimal),
    }

    print("\n請選擇要運行的範例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  a. 運行所有範例")
    print("  q. 退出")

    choice = input("\n請輸入選項 (1-7/a/q): ").strip().lower()

    if choice == 'q':
        print("\n👋 再見！")
        return

    try:
        if choice == 'a':
            # 運行所有範例
            for name, func in examples.values():
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ 範例執行失敗: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice in examples:
            # 運行單個範例
            name, func = examples[choice]
            func()
        else:
            print("\n❌ 無效的選項")
            return

        print("\n" + "="*60)
        print("✅ 範例執行完成！")
        print("="*60)
        print("\n💡 提示：")
        print("   - 所有生成的檔案保存在 output_media/ 目錄")
        print("   - 可以使用 output_subdir 參數來組織不同的生成結果")
        print("   - 支援自訂 workflow、style、keywords 等參數")
        print("   - 查看 FLEXIBLE_USAGE.md 了解更多用法")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
