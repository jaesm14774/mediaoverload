"""新版主程式 - 使用重構後的架構"""
import argparse
import asyncio
import os
from lib.services.service_factory import ServiceFactory
from lib.media_auto.character_base import ConfigurableCharacterWithSocialMedia


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
        elif args.character:
            # 使用角色名稱
            config_path = f"configs/characters/{args.character.lower()}.yaml"
            if os.path.exists(config_path):
                print(f"從配置檔案載入角色: {config_path}")
                character = ConfigurableCharacterWithSocialMedia(config_path=config_path)
            else:
                raise ValueError(f"找不到角色配置文件: {config_path}")
        else:
            raise ValueError("必須提供 --config 或 --character 參數")
        
        # 註冊社群媒體平台
        publishing_service = service_factory.get_publishing_service()
        
        # 如果使用配置檔案，從配置中註冊平台
        if hasattr(character, '_social_media_config'):
            for platform_name, platform_config in character._social_media_config.items():
                publishing_service.register_platform(platform_name, platform_config)
        
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