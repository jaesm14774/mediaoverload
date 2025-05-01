from lib.media_auto.media_main_logic import ContentProcessor
import lib.media_auto.process as media_process
import argparse
import asyncio
from lib.media_auto.orchestrator import PipelineOrchestrator

def main():
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description='Media processing interface')
    parser.add_argument('--character', type=str, default='Wobbuffet',
                       help='Character name for processing')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt for the ETL process')
    parser.add_argument('--pipeline', type=str, default='text2image',
                        help='Pipeline to execute: text2image, image2image, text2video, complex_media')
    
    args = parser.parse_args()
    # 處理 pipeline 名稱（向後相容）
    pipeline = args.pipeline.lower()
    if pipeline == 'text2img':
        pipeline = 'text2image'
    
    # 初始化處理器
    character_class = getattr(media_process, f'{args.character}Process')
    # 根據選擇的 pipeline 執行對應流程
    if pipeline == 'text2image':
        # 舊版文生圖工作流
        process = ContentProcessor(character_class())
        asyncio.run(process.etl_process(prompt=args.prompt))
    else:
        # 新版多階段工作流
        orchestrator = PipelineOrchestrator()
        char_proc = character_class()
        asyncio.run(orchestrator.run(pipeline, char_proc, args.prompt))

if __name__ == "__main__":
    main()
    print("done")