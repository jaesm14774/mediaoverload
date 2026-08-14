from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.h3_reference import format_ref2va_prompt, normalize_reference_manifest


class H3ReferenceTests(unittest.TestCase):
    def test_normalizes_image_and_video_with_deterministic_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "subject.png"
            video = root / "motion.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            refs = normalize_reference_manifest(
                [
                    {"path": str(image), "type": "image", "role": "identity"},
                    {"path": str(video), "type": "video", "role": "motion"},
                ],
                require_files=True,
            )
            self.assertEqual([ref["tag"] for ref in refs], ["reference_image_1", "reference_video_1"])
            self.assertEqual([ref["type"] for ref in refs], ["image", "video"])

    def test_rejects_reference_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reference audio"):
            normalize_reference_manifest([{"path": "voice.wav", "type": "audio"}])

    def test_rejects_audio_extension_without_declared_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "type=image or type=video"):
            normalize_reference_manifest([{"path": "voice.wav"}])

    def test_ref2va_prompt_has_contract_and_no_audio_reference_input(self) -> None:
        refs = [{"tag": "reference_image_1", "prompt_label": "<Picture 1>", "role": "identity", "retention": "identity_and_appearance"}]
        prompt = format_ref2va_prompt("A Kirby scene", refs)
        self.assertIn("<Subject Definitions>", prompt)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Overall Soundscape>", prompt)
        self.assertIn("do not use reference audio", prompt.lower())
        self.assertNotIn("<Reference Audio>", prompt)


if __name__ == "__main__":
    unittest.main()
