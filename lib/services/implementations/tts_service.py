"""TTS 服務"""
import asyncio
import os
import tempfile
import concurrent.futures
from typing import Optional, List
from pathlib import Path
import logging

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class TTSService:
    """TTS 服務"""
    
    VOICES = {
        # English voices
        "en-US-AriaNeural": "Female, American English (default)",
        "en-US-GuyNeural": "Male, American English",
        "en-GB-SoniaNeural": "Female, British English",
        "en-GB-RyanNeural": "Male, British English",
        
        # Chinese voices
        "zh-CN-XiaoxiaoNeural": "Female, Mandarin Chinese",
        "zh-CN-YunxiNeural": "Male, Mandarin Chinese",
        "zh-TW-HsiaoChenNeural": "Female, Taiwanese Mandarin",
        "zh-TW-YunJheNeural": "Male, Taiwanese Mandarin",
        
        # Japanese voices
        "ja-JP-NanamiNeural": "Female, Japanese",
        "ja-JP-KeitaNeural": "Male, Japanese",
    }
    
    def __init__(self, default_voice: str = "en-US-AriaNeural"):
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts is not installed. Install it with: pip install edge-tts")
        
        self.default_voice = default_voice
        self.logger = logging.getLogger(__name__)
    
    async def generate_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: str = "+0%"
    ) -> str:
        """生成語音
        
        Args:
            text: 要轉換的文字
            output_path: 輸出音訊檔案路徑
            voice: 語音（None 則使用預設語音）
            rate: 語速調整（例如: "+10%", "-20%", "+0%"）
                  正值 = 更快，負值 = 更慢
                  
        Returns:
            生成的音訊檔案路徑
            
        Raises:
            RuntimeError: 生成失敗時
        """
        voice = voice or self.default_voice
        
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(output_path)
            
            if not os.path.exists(output_path):
                raise RuntimeError(f"Audio file was not created: {output_path}")
            
            file_size = os.path.getsize(output_path)
            self.logger.info(f"已生成語音: {len(text)} 字元 → {output_path} ({file_size} bytes, 語音: {voice})")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to generate speech: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate_speech_sync(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: str = "+0%"
    ) -> str:
        """同步版本的 generate_speech
        
        用於從非 async 程式碼呼叫 TTS。
        
        Args:
            text: 要轉換的文字
            output_path: 輸出音訊檔案路徑
            voice: 語音（None 則使用預設語音）
            rate: 語速調整
            
        Returns:
            生成的音訊檔案路徑
        """
        voice = voice or self.default_voice
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        def run_tts_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                loop.run_until_complete(communicate.save(output_path))
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_tts_in_thread)
            future.result(timeout=60)
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Audio file was not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        self.logger.info(f"已生成語音: {len(text)} 字元 → {output_path} ({file_size} bytes, 語音: {voice})")
        
        return output_path
    
    async def generate_multiple_speeches(
        self,
        texts: List[str],
        output_dir: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        filename_prefix: str = "segment"
    ) -> List[str]:
        """生成多個語音檔案
        
        用於一次生成所有影片片段的旁白。
        
        Args:
            texts: 要轉換的文字列表
            output_dir: 儲存音訊檔案的目錄
            voice: 所有片段使用的語音
            rate: 語速調整
            filename_prefix: 輸出檔名前綴
            
        Returns:
            生成的音訊檔案路徑列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = []
        
        for i, text in enumerate(texts, 1):
            output_path = os.path.join(
                output_dir,
                f"{filename_prefix}_{i:03d}.mp3"
            )
            
            await self.generate_speech(text, output_path, voice, rate)
            audio_files.append(output_path)
        
        self.logger.info(f"已生成 {len(audio_files)} 個音訊檔案於 {output_dir}")
        return audio_files
    
    @staticmethod
    async def list_available_voices() -> List[dict]:
        """列出所有可用的語音
        
        Returns:
            語音字典列表，包含名稱、性別和地區資訊
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts is not installed")
        
        voices = await edge_tts.list_voices()
        return voices
    
    def get_voice_info(self, voice_name: str) -> Optional[str]:
        """取得語音資訊
        
        Args:
            voice_name: 語音短名稱（例如: "en-US-AriaNeural"）
            
        Returns:
            語音描述，如果不在預定義列表中則返回 None
        """
        return self.VOICES.get(voice_name)
    
    def list_popular_voices(self) -> dict:
        """列出常用語音
        
        Returns:
            語音名稱和描述的字典
        """
        return self.VOICES.copy()
