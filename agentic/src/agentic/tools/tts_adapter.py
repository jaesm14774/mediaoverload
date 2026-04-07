from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import edge_tts
except ImportError:  # pragma: no cover - exercised in runtime environments without edge-tts
    edge_tts = None


class TTSAdapter:
    """Agentic-native wrapper for edge-tts."""

    def __init__(self, default_voice: str = "en-US-AriaNeural") -> None:
        self.default_voice = default_voice

    def generate_speech_sync(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        rate: str = "+0%",
    ) -> str:
        if edge_tts is None:
            raise RuntimeError("edge-tts is not installed. Install it to enable TTS output.")

        resolved_voice = voice or self.default_voice
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                communicator = edge_tts.Communicate(text, resolved_voice, rate=rate)
                loop.run_until_complete(communicator.save(output_path))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(_run).result(timeout=60)

        if not Path(output_path).exists():
            raise RuntimeError(f"Audio file was not created: {output_path}")
        return output_path
