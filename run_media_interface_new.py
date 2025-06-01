"""新版主程式 - 使用重構後的架構"""
import argparse
import asyncio
import os
from lib.services.service_factory import ServiceFactory
from lib.media_auto.character_base import ConfigurableCharacterWithSocialMedia
import lib.media_auto.process as media_process


async def main():
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description='Media processing interface (新架構)')
    parser.add_argument('--config', type=str, help='角色配置文件路徑')
    parser.add_argument('--character', type=str, help='角色名稱')
    parser.add_argument('--prompt', type=str, default='', help='提示詞')
    parser.add_argument('--temperature', type=float, default=1.0, help='溫度參數')
    args = parser.parse_args()
    
    # 初始化服務工廠
    service_factory = ServiceFactory()
    
    try:
        # 創建角色實例
        character = None
        
        if args.config:
            # 使用配置檔案
            print(f"從配置檔案載入角色: {args.config}")
            character = ConfigurableCharacterWithSocialMedia(config_path=args.config)
        else:
            # 使用舊的方式（向後相容）
            print(f"使用內建角色類別: {args.character}Process")
            character_class = getattr(media_process, f'{args.character}Process')
            character = character_class()
        
        # 註冊社群媒體平台
        publishing_service = service_factory.get_publishing_service()
        
        # 如果使用配置檔案，從配置中註冊平台
        if args.config and hasattr(character, '_social_media_config'):
            for platform_name, platform_config in character._social_media_config.items():
                publishing_service.register_platform(platform_name, platform_config)
        # 如果使用舊的方式，從角色的 social_media_manager 中註冊平台
        elif hasattr(character, 'social_media_manager'):
            for platform_name, platform in character.social_media_manager.platforms.items():
                publishing_service.social_media_manager.register_platform(platform_name, platform)
        
        # 獲取協調服務
        orchestration_service = service_factory.get_orchestration_service()
        
        # 執行工作流程
        result = await orchestration_service.execute_workflow(
            character=character,
            prompt=args.prompt,
            temperature=args.temperature
        )
        
        print(f"工作流程執行完成: {result}")
        
    finally:
        # 清理資源
        service_factory.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
    print("完成") 