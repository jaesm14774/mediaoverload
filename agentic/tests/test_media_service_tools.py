from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.tools.comfy_backend import AgenticComfyCommunicator
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.media_services import MediaServiceTools
from agentic.tools.social_native import InstagramGraphPlatform, MediaPost
from agentic.tools.social_services import SocialServiceTools
from agentic.tools.tts_adapter import TTSAdapter


class FFmpegAdapterTests(unittest.TestCase):
    @patch("agentic.tools.ffmpeg_adapter.shutil.copy2")
    def test_concat_videos_copies_single_input(self, copy_mock) -> None:
        adapter = FFmpegAdapter()
        result = adapter.concat_videos(["input.mp4"], "output.mp4")

        self.assertEqual(result, "output.mp4")
        copy_mock.assert_called_once_with("input.mp4", "output.mp4")

    @patch("agentic.tools.ffmpeg_adapter.subprocess.run")
    def test_video_to_gif_uses_palette_pipeline(self, run_mock) -> None:
        adapter = FFmpegAdapter()
        output_path = "preview.gif"

        result = adapter.video_to_gif("clip.mp4", output_path, fps=10, max_colors=128, scale_width=320)

        self.assertEqual(result, output_path)
        self.assertEqual(run_mock.call_count, 4)
        palette_command = run_mock.call_args_list[2].args[0]
        gif_command = run_mock.call_args_list[3].args[0]
        self.assertEqual(palette_command[0], "ffmpeg")
        self.assertIn("palettegen=max_colors=128", " ".join(palette_command))
        self.assertIn("paletteuse", " ".join(gif_command))


class AgenticComfyCommunicatorTests(unittest.TestCase):
    def test_wait_for_completion_ignores_binary_preview_frames(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188, timeout=1)

        class FakeSocket:
            def __init__(self) -> None:
                self.connected = True
                self.messages = [
                    b"\x89PNG\r\n\x1a\n",
                    '{"type":"executing","data":{"prompt_id":"prompt-1","node":null}}',
                ]

            def settimeout(self, value: float) -> None:
                del value

            def recv(self):
                return self.messages.pop(0)

        communicator.ws = FakeSocket()  # type: ignore[assignment]
        communicator.wait_for_completion("prompt-1")


class TTSAdapterTests(unittest.TestCase):
    def test_generate_speech_sync_raises_when_edge_tts_missing(self) -> None:
        adapter = TTSAdapter()
        with patch("agentic.tools.tts_adapter.edge_tts", None):
            with self.assertRaises(RuntimeError):
                adapter.generate_speech_sync("hello", str(Path(tempfile.gettempdir()) / "voice.mp3"))


class MediaServiceToolsTests(unittest.TestCase):
    def test_generate_tts_delegates_to_agentic_adapter(self) -> None:
        tools = MediaServiceTools(Path.cwd())

        class FakeTTS:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str, str, str]] = []

            def generate_speech_sync(self, text: str, output_path: str, voice: str, rate: str) -> str:
                self.calls.append((text, output_path, voice, rate))
                return output_path

        fake_tts = FakeTTS()
        tools._tts = fake_tts  # type: ignore[assignment]

        result = tools.generate_tts(
            {
                "text": "narrate this",
                "output_path": "speech.mp3",
                "voice": "zh-TW-HsiaoChenNeural",
                "rate": "+10%",
            }
        )

        self.assertEqual(result["audio_path"], "speech.mp3")
        self.assertEqual(
            fake_tts.calls,
            [("narrate this", "speech.mp3", "zh-TW-HsiaoChenNeural", "+10%")],
        )


class SocialServiceToolsTests(unittest.TestCase):
    def test_publish_social_delegates_to_agentic_publishing_adapter(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        class FakePublishing:
            def __init__(self) -> None:
                self.registered: list[tuple[str, dict[str, object]]] = []
                self.published: list[tuple[object, list[str] | None]] = []

            def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
                return media_paths

            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                self.registered.append((platform_name, platform_config))

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                self.published.append((post, platforms))
                platform_name = (platforms or ["instagram"])[0]
                return {platform_name: True}

        fake = FakePublishing()
        tools._publishing = fake  # type: ignore[assignment]

        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "launch",
                "hashtags": "#kirby",
                "platforms": ["instagram"],
                "platform_configs": {"instagram": {"config_folder_path": "configs/social"}},
                "additional_params": {"kind": "reel"},
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"], {"instagram": True})
        self.assertEqual(fake.registered, [("instagram", {"config_folder_path": "configs/social"})])
        self.assertEqual(fake.published[0][1], ["instagram"])
        post = fake.published[0][0]
        self.assertEqual(post.caption, "launch")
        self.assertEqual(post.hashtags, "#kirby")
        self.assertEqual(post.media_paths, ["clip.mp4"])

    def test_publish_social_uses_platform_bundle_payloads(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        class FakePublishing:
            def __init__(self) -> None:
                self.registered: list[tuple[str, dict[str, object]]] = []
                self.published: list[tuple[object, list[str] | None]] = []

            def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
                return media_paths

            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                self.registered.append((platform_name, platform_config))

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                self.published.append((post, platforms))
                platform_name = (platforms or ["instagram_graph"])[0]
                return {platform_name: True}

        fake = FakePublishing()
        tools._publishing = fake  # type: ignore[assignment]

        result = tools.publish_social(
            {
                "media_paths": ["shared-a.jpg", "shared-b.jpg"],
                "caption": "generic",
                "hashtags": "#generic",
                "platforms": ["instagram_graph", "facebook"],
                "platform_configs": {
                    "instagram_graph": {"config_folder_path": "configs/ig"},
                    "facebook": {"config_folder_path": "configs/fb"},
                },
                "platform_bundle": {
                    "instagram_graph": {
                        "caption": "ig caption",
                        "hashtags": "#ig",
                        "media_paths": ["ig-a.jpg", "ig-b.jpg"],
                    },
                    "facebook": {
                        "caption": "fb caption",
                        "hashtags": "#fb",
                        "media_paths": ["fb-a.jpg", "fb-b.jpg"],
                    },
                },
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"], {"instagram_graph": True, "facebook": True})
        self.assertEqual(len(fake.published), 2)
        ig_post = fake.published[0][0]
        fb_post = fake.published[1][0]
        self.assertEqual(ig_post.caption, "ig caption")
        self.assertEqual(ig_post.hashtags, "#ig")
        self.assertEqual(ig_post.media_paths, ["ig-a.jpg", "ig-b.jpg"])
        self.assertEqual(fb_post.caption, "fb caption")
        self.assertEqual(fb_post.hashtags, "#fb")
        self.assertEqual(fb_post.media_paths, ["fb-a.jpg", "fb-b.jpg"])

    def test_publish_social_collects_partial_failures(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                del post
                platform_name = (platforms or ["instagram_graph"])[0]
                if platform_name == "twitter":
                    raise RuntimeError("404 Not Found")
                return {platform_name: True}

        tools._publishing = FakePublishing()  # type: ignore[assignment]

        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "launch",
                "hashtags": "#kirby",
                "platforms": ["instagram_graph", "twitter"],
                "platform_configs": {
                    "instagram_graph": {"config_folder_path": "configs/ig"},
                    "twitter": {"config_folder_path": "configs/twitter"},
                },
            }
        )

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["results"], {"instagram_graph": True, "twitter": False})
        self.assertIn("twitter", result["errors"])

    def test_publish_social_treats_false_result_as_failure(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                del post
                platform_name = (platforms or ["instagram_graph"])[0]
                return {platform_name: False}

        tools._publishing = FakePublishing()  # type: ignore[assignment]

        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "launch",
                "hashtags": "#kirby",
                "platforms": ["instagram_graph"],
                "platform_configs": {
                    "instagram_graph": {"config_folder_path": "configs/ig"},
                },
            }
        )

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["results"], {"instagram_graph": False})
        self.assertIn("instagram_graph", result["errors"])


class InstagramGraphPlatformTests(unittest.TestCase):
    @patch.object(InstagramGraphPlatform, "authenticate")
    @patch.object(InstagramGraphPlatform, "load_config")
    def test_publish_image_raises_clear_error_without_public_media_url(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = InstagramGraphPlatform("configs/social_media/credentials/kirby")
        platform.ig_user_id = "123"
        platform.access_token = "token"
        platform.media_base_url = None
        platform.cloudinary = type("CloudinaryStub", (), {"upload": staticmethod(lambda _: None)})()

        with self.assertRaisesRegex(RuntimeError, "configure Cloudinary or IG_GRAPH_MEDIA_BASE_URL"):
            platform._publish_image_url("sample.jpg", "caption")

    @patch.object(InstagramGraphPlatform, "authenticate")
    @patch.object(InstagramGraphPlatform, "load_config")
    def test_upload_post_truncates_caption_to_instagram_limit(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = InstagramGraphPlatform("configs/social_media/credentials/kirby")
        long_caption = "a" * 2300

        tmpdir = Path.cwd() / "agentic" / ".tmp-tests" / "ig-caption-limit"
        tmpdir.mkdir(parents=True, exist_ok=True)
        image_path = str(tmpdir / "frame.jpg")
        Path(image_path).write_bytes(b"test")

        captured: dict[str, str] = {}

        def fake_publish_image(path: str, caption: str) -> bool:
            captured["path"] = path
            captured["caption"] = caption
            return True

        platform._publish_image_url = fake_publish_image  # type: ignore[method-assign]
        result = platform.upload_post(MediaPost(media_paths=[image_path], caption=long_caption))

        self.assertTrue(result)
        self.assertEqual(captured["path"], image_path)
        self.assertEqual(len(captured["caption"]), platform.CAPTION_MAX)


if __name__ == "__main__":
    unittest.main()
