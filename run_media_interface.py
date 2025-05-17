from lib.media_auto.media_main_logic import ContentProcessor
import lib.media_auto.process as media_process
import argparse
import asyncio

def main():
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description='Media processing interface')
    parser.add_argument('--character', type=str, default='Wobbuffet',
                       help='Character name for processing')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt for the ETL process')
    
    args = parser.parse_args()
    
    # 初始化處理器
    character_class = getattr(media_process, f'{args.character}Process')
    
    # 只保留舊版文生圖工作流
    process = ContentProcessor(character_class())
    asyncio.run(process.etl_process(prompt=args.prompt))

if __name__ == "__main__":
    main()
    print("done")