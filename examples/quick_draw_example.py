"""Quick Draw 使用範例

這個範例展示如何使用簡化的內容生成服務來快速生成圖片。
跳過耗時的圖文匹配分析和文章生成步驟，專注於圖片生成本身。

適合用於：
- 快速測試和開發
- 需要人工審核的情況
- 範例展示

如需完整功能（包含分析和文章生成），請使用:
from lib.services.implementations.content_generation_service import ContentGenerationService
"""

import sys
from pathlib import Path

# 確保可以導入專案模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.quick_draw.use_cases import (
    SingleCharacterUseCase,
    CharacterInteractionUseCase,
    NewsBasedUseCase,
    BuddhistStyleUseCase,
    BlackHumorUseCase,
    CinematicUseCase
)


def example_single_character():
    """範例 1: 單角色圖片生成"""
    print("\n" + "="*60)
    print("範例 1: 單角色圖片生成")
    print("="*60)
    
    use_case = SingleCharacterUseCase()
    result = use_case.execute(
        character='Kirby',
        topic='peaceful sleeping',
        style='minimalist style, simple white background',
        images_per_description=2
    )
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")
    print(f"📝 描述: {result['descriptions'][0] if result['descriptions'] else 'N/A'}")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def example_character_interaction():
    """範例 2: 雙角色互動"""
    print("\n" + "="*60)
    print("範例 2: 雙角色互動")
    print("="*60)
    
    use_case = CharacterInteractionUseCase()
    result = use_case.execute(
        main_character='Kirby',
        secondary_character='Waddle Dee',
        topic='friendship and companionship',
        style='warm and cozy atmosphere',
        images_per_description=2
    )
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def example_news_based():
    """範例 3: 基於新聞關鍵字"""
    print("\n" + "="*60)
    print("範例 3: 基於新聞關鍵字")
    print("="*60)
    
    use_case = NewsBasedUseCase()
    result = use_case.execute(
        character='Kirby',
        news_count=2,  # 只使用 2 條新聞作為範例
        images_per_description=2
    )
    
    print(f"\n✅ 處理了 {result['total_news']} 條新聞")
    print(f"📊 總結:")
    summary = result['summary']
    print(f"   - 描述數量: {summary['total_descriptions']}")
    print(f"   - 圖片數量: {summary['total_media_files']}")
    
    return result


def example_buddhist_style():
    """範例 4: 佛性/靈性風格"""
    print("\n" + "="*60)
    print("範例 4: 佛性/靈性風格")
    print("="*60)
    
    use_case = BuddhistStyleUseCase()
    result = use_case.execute(
        character='Kirby',
        spiritual_theme='meditation and enlightenment',
        use_news=True,
        images_per_description=2
    )
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張佛性風格圖片")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def example_black_humor():
    """範例 5: 黑色幽默"""
    print("\n" + "="*60)
    print("範例 5: 黑色幽默")
    print("="*60)
    
    use_case = BlackHumorUseCase()
    result = use_case.execute(
        main_character='Kirby',
        secondary_character='Waddle Dee',
        images_per_description=2
    )
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張黑色幽默圖片")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def example_cinematic():
    """範例 6: 電影級別"""
    print("\n" + "="*60)
    print("範例 6: 電影級別 (16:9 寬螢幕)")
    print("="*60)
    
    use_case = CinematicUseCase()
    result = use_case.execute(
        main_character='Kirby',
        secondary_character='Meta Knight',
        aspect_ratio='cinematic',  # 1280x720
        use_news=True,
        images_per_description=2
    )
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張電影級別圖片")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def example_custom_config():
    """範例 7: 使用 ConfigBuilder 自定義配置"""
    print("\n" + "="*60)
    print("範例 7: 使用 ConfigBuilder 自定義配置")
    print("="*60)
    
    from examples.quick_draw.helpers import ConfigBuilder
    from examples.simple_content_service import SimpleContentGenerationService
    from lib.repositories.character_repository import CharacterRepository
    from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
    from lib.database import db_pool
    from dotenv import load_dotenv
    import os
    
    # 載入環境變數
    load_dotenv('media_overload.env')
    
    # 初始化資料庫
    db_pool.initialize('mysql',
                      host=os.environ['mysql_host'],
                      port=int(os.environ['mysql_port']),
                      user=os.environ['mysql_user'],
                      password=os.environ['mysql_password'],
                      db_name=os.environ['mysql_db_name'])
    
    # 初始化服務
    mysql_conn = db_pool.get_connection('mysql')
    character_repository = CharacterRepository(mysql_conn)
    
    vision_manager = VisionManagerBuilder() \
        .with_vision_model('openrouter') \
        .with_text_model('openrouter') \
        .with_random_models(True) \
        .build()
    
    # 使用 ConfigBuilder 建立配置
    config = ConfigBuilder() \
        .with_character('Kirby') \
        .with_workflow('configs/workflow/nova-anime-xl.json') \
        .with_output_dir('output_media') \
        .with_prompt('peaceful sleeping under the stars') \
        .with_style('dreamy, soft lighting') \
        .with_image_system_prompt('stable_diffusion_prompt') \
        .with_images_per_description(2) \
        .build()
    
    # 使用簡化的內容生成服務
    service = SimpleContentGenerationService(
        character_repository=character_repository,
        vision_manager=vision_manager
    )
    result = service.generate_content(config)
    
    print(f"\n✅ 生成了 {len(result['media_files'])} 張圖片")
    print(f"📂 圖片路徑:")
    for img in result['media_files']:
        print(f"   - {img}")
    
    return result


def main():
    """主函數 - 運行所有範例"""
    print("\n" + "="*60)
    print("Quick Draw 使用範例")
    print("簡化版內容生成服務 - 跳過圖文匹配分析和文章生成")
    print("="*60)
    
    # 選擇要運行的範例
    examples = {
        '1': ('單角色圖片生成', example_single_character),
        '2': ('雙角色互動', example_character_interaction),
        '3': ('基於新聞關鍵字', example_news_based),
        '4': ('佛性/靈性風格', example_buddhist_style),
        '5': ('黑色幽默', example_black_humor),
        '6': ('電影級別', example_cinematic),
        '7': ('自定義配置 (ConfigBuilder)', example_custom_config),
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
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用戶中斷執行")
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

