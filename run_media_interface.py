import argparse
import asyncio
import os
from lib.services.service_factory import ServiceFactory
from lib.media_auto.character_base import ConfigurableCharacterWithSocialMedia


async def main():
    parser = argparse.ArgumentParser(description='Media processing interface')
    parser.add_argument('--config', type=str, help='Path to character config file')
    parser.add_argument('--character', type=str, help='Character name')
    parser.add_argument('--prompt', type=str, default='', help='Prompt text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter')
    args = parser.parse_args()
    
    service_factory = ServiceFactory()
    
    try:
        character = None
        
        if args.config:
            print(f"Loading character from config: {args.config}")
            character = ConfigurableCharacterWithSocialMedia(config_path=args.config)
        elif args.character:
            config_path = f"configs/characters/{args.character.lower()}.yaml"
            if os.path.exists(config_path):
                print(f"Loading character from config: {config_path}")
                character = ConfigurableCharacterWithSocialMedia(config_path=config_path)
            else:
                raise ValueError(f"Character config not found: {config_path}")
        else:
            raise ValueError("Must provide --config or --character argument")
        
        publishing_service = service_factory.get_publishing_service()
        
        if hasattr(character, '_social_media_config'):
            for platform_name, platform_config in character._social_media_config.items():
                publishing_service.register_platform(platform_name, platform_config)
        
        orchestration_service = service_factory.get_orchestration_service()
        
        result = await orchestration_service.execute_workflow(
            character=character,
            prompt=args.prompt,
            temperature=args.temperature
        )
        
        print(f"Workflow execution completed: {result}")
        
    finally:
        service_factory.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
    print("Done")