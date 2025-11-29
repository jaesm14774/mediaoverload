"""TTS Service - Text-to-Speech using Edge-TTS"""
import asyncio
import os
import tempfile
from typing import Optional, List
from pathlib import Path
import logging

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class TTSService:
    """
    Text-to-Speech service using Edge-TTS.
    
    Provides narration audio generation for long-form video segments.
    Uses Microsoft Edge's neural TTS voices for high-quality speech synthesis.
    
    Features:
    - Multiple voice options (different languages and genders)
    - Adjustable speech rate
    - High-quality neural voices
    - Free to use (no API key required)
    
    Example:
        >>> tts = TTSService()
        >>> audio_path = await tts.generate_speech(
        ...     "Hello, this is a test narration",
        ...     output_path="narration.mp3",
        ...     voice="en-US-AriaNeural"
        ... )
    """
    
    # Popular voice options
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
        """
        Initialize TTS service.
        
        Args:
            default_voice: Default voice to use for speech generation
        
        Raises:
            ImportError: If edge-tts is not installed
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError(
                "edge-tts is not installed. "
                "Install it with: pip install edge-tts"
            )
        
        self.default_voice = default_voice
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TTSService initialized with voice: {default_voice}")
    
    async def generate_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: str = "+0%"
    ) -> str:
        """
        Generate speech from text using Edge-TTS.
        
        Args:
            text: Text to convert to speech
            output_path: Path for output audio file (e.g., 'narration.mp3')
            voice: Voice to use (if None, uses default_voice)
            rate: Speech rate adjustment (e.g., "+10%", "-20%", "+0%")
                  Positive values = faster, negative = slower
        
        Returns:
            Path to generated audio file
        
        Raises:
            RuntimeError: If speech generation fails
        
        Example:
            >>> audio = await tts.generate_speech(
            ...     "Welcome to this video",
            ...     "segment1_audio.mp3",
            ...     rate="+10%"  # Slightly faster
            ... )
        """
        voice = voice or self.default_voice
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Create TTS communicator
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            
            # Generate and save audio
            await communicate.save(output_path)
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Audio file was not created: {output_path}")
            
            file_size = os.path.getsize(output_path)
            self.logger.info(
                f"Generated speech: {len(text)} chars → "
                f"{output_path} ({file_size} bytes, voice: {voice})"
            )
            
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
        """
        Synchronous wrapper for generate_speech.
        
        Use this if you need to call TTS from non-async code.
        
        Args:
            text: Text to convert to speech
            output_path: Path for output audio file
            voice: Voice to use (if None, uses default_voice)
            rate: Speech rate adjustment
        
        Returns:
            Path to generated audio file
        """
        return asyncio.run(
            self.generate_speech(text, output_path, voice, rate)
        )
    
    async def generate_multiple_speeches(
        self,
        texts: List[str],
        output_dir: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        filename_prefix: str = "segment"
    ) -> List[str]:
        """
        Generate multiple speech files from a list of texts.
        
        Useful for generating narration for all video segments at once.
        
        Args:
            texts: List of text strings to convert
            output_dir: Directory to save audio files
            voice: Voice to use for all segments
            rate: Speech rate adjustment
            filename_prefix: Prefix for output filenames
        
        Returns:
            List of paths to generated audio files
        
        Example:
            >>> segments = ["Intro text", "Middle text", "Outro text"]
            >>> audio_files = await tts.generate_multiple_speeches(
            ...     segments,
            ...     "output/audio",
            ...     filename_prefix="narration"
            ... )
            >>> # Returns: ["output/audio/narration_001.mp3", ...]
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
        
        self.logger.info(
            f"Generated {len(audio_files)} audio files in {output_dir}"
        )
        
        return audio_files
    
    @staticmethod
    async def list_available_voices() -> List[dict]:
        """
        List all available voices from Edge-TTS.
        
        Returns:
            List of voice dictionaries with name, gender, and locale info
        
        Example:
            >>> voices = await TTSService.list_available_voices()
            >>> for voice in voices[:5]:
            ...     print(f"{voice['ShortName']}: {voice['Gender']}, {voice['Locale']}")
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts is not installed")
        
        voices = await edge_tts.list_voices()
        return voices
    
    def get_voice_info(self, voice_name: str) -> Optional[str]:
        """
        Get description of a voice from the predefined list.
        
        Args:
            voice_name: Voice short name (e.g., "en-US-AriaNeural")
        
        Returns:
            Voice description or None if not in predefined list
        """
        return self.VOICES.get(voice_name)
    
    def list_popular_voices(self) -> dict:
        """
        Get list of popular pre-configured voices.
        
        Returns:
            Dictionary of voice names and descriptions
        """
        return self.VOICES.copy()
