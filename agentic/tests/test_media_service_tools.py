from __future__ import annotations

import os
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from agentic.tools.comfy_backend import AgenticComfyCommunicator
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.media_services import MediaServiceTools
from agentic.tools.social_native import FacebookPlatform, InstagramGraphPlatform, MediaPost, YouTubePlatform
from agentic.tools.social_services import (
    SocialServiceTools,
    complete_facebook_profile_handoff,
    record_facebook_profile_handoff_delivery,
)
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

    @patch.object(FFmpegAdapter, "_run")
    @patch.object(
        FFmpegAdapter,
        "_run_capture",
        return_value=json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "width": 608, "height": 352, "avg_frame_rate": "24/1"},
                    {"codec_type": "audio", "channels": 2},
                ],
                "format": {"duration": "5.0"},
            }
        ),
    )
    def test_change_video_speed_synchronizes_audio_and_video(self, _capture_mock, run_mock) -> None:
        adapter = FFmpegAdapter()
        adapter._checked = True
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.mp4"
            output_path = Path(directory) / "output_2x.mp4"
            input_path.write_bytes(b"fixture")

            result = adapter.change_video_speed(str(input_path), str(output_path), 2.0)

        self.assertEqual(result, str(output_path))
        command = run_mock.call_args.args[0]
        command_text = " ".join(command)
        self.assertIn("setpts=0.5*PTS", command_text)
        self.assertIn("fps=24", command_text)
        self.assertIn("atempo=2", command_text)
        self.assertIn("-map [v] -map [a]", command_text)

    @patch.object(FFmpegAdapter, "_run")
    def test_trim_video_keeps_video_and_optional_audio(self, run_mock) -> None:
        adapter = FFmpegAdapter()
        adapter._checked = True
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.mp4"
            output_path = Path(directory) / "output_trimmed.mp4"
            input_path.write_bytes(b"fixture")

            result = adapter.trim_video(str(input_path), str(output_path), 20.0)

        self.assertEqual(result, str(output_path))
        command = run_mock.call_args.args[0]
        command_text = " ".join(command)
        self.assertIn("-t 20.000000", command_text)
        self.assertIn("-map 0:v:0 -map 0:a?", command_text)
        self.assertIn("-shortest", command_text)


class AgenticComfyCommunicatorTests(unittest.TestCase):
    def test_default_timeout_is_configurable_for_lowvram_generation(self) -> None:
        with patch.dict(os.environ, {"COMFYUI_TIMEOUT_SECONDS": "1800"}):
            communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188)

        self.assertEqual(communicator.timeout, 1800)

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

    def test_wait_for_completion_sets_timeout_on_underlying_socket(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188, timeout=0.2)

        class FakeSocket:
            def __init__(self) -> None:
                self.connected = True
                self.sock = self
                self.timeout_values: list[float] = []

            def settimeout(self, value: float) -> None:
                self.timeout_values.append(value)

            def recv(self):
                raise socket.timeout()

        fake_socket = FakeSocket()
        communicator.ws = fake_socket  # type: ignore[assignment]
        with self.assertRaises(TimeoutError):
            communicator.wait_for_completion("prompt-timeout")
        self.assertTrue(fake_socket.timeout_values)

    def test_wait_for_completion_recovers_when_terminal_websocket_event_is_lost(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188, timeout=1)

        class FakeSocket:
            connected = True
            sock = None

            def settimeout(self, value: float) -> None:
                del value

            def recv(self):
                raise socket.timeout()

        communicator.ws = FakeSocket()  # type: ignore[assignment]
        with patch.object(
            communicator,
            "get_history",
            return_value={"prompt-history-success": {"status": {"completed": True, "status_str": "success"}}},
        ):
            communicator.wait_for_completion("prompt-history-success")

    def test_comfy_media_filename_cannot_escape_run_output_directory(self) -> None:
        output_dir = Path(tempfile.mkdtemp())

        with self.assertRaises(ValueError):
            AgenticComfyCommunicator._safe_output_path(str(output_dir), "..\\outside.mp4")

    def test_cancel_prompt_deletes_only_a_pending_prompt(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188)

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.payload

        calls = []
        with patch("agentic.tools.comfy_backend.request.urlopen") as urlopen:
            urlopen.side_effect = [
                FakeResponse(
                    json.dumps(
                        {
                            "queue_running": [[1, "other-prompt"]],
                            "queue_pending": [[2, "stale-prompt"]],
                        }
                    ).encode()
                ),
                FakeResponse(b"{}"),
            ]
            communicator.cancel_prompt("stale-prompt")
            calls.extend(urlopen.call_args_list)

        self.assertEqual(len(calls), 2)
        delete_request = calls[1].args[0]
        self.assertEqual(delete_request.method, "DELETE")
        self.assertIn("stale-prompt", delete_request.data.decode())

    def test_cancel_prompt_interrupts_only_a_running_prompt(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188)

        class FakeResponse:
            def __init__(self, payload: bytes = b"{}") -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.payload

        with patch("agentic.tools.comfy_backend.request.urlopen") as urlopen:
            urlopen.side_effect = [
                FakeResponse(
                    json.dumps(
                        {
                            "queue_running": [[1, "owned-prompt"]],
                            "queue_pending": [[2, "other-prompt"]],
                        }
                    ).encode()
                ),
                FakeResponse(b"{}"),
            ]
            communicator.cancel_prompt("owned-prompt")

        interrupt_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(interrupt_request.method, "POST")
        self.assertIn("/interrupt", interrupt_request.full_url)

    def test_save_results_uses_preview_when_comfy_history_has_no_persistent_media(self) -> None:
        communicator = AgenticComfyCommunicator.__new__(AgenticComfyCommunicator)
        communicator.get_history = lambda _prompt_id: {  # type: ignore[method-assign]
            "prompt-1": {
                "outputs": {
                    "preview": {
                        "images": [
                            {
                                "filename": "ComfyUI_temp_preview_00001_.png",
                                "subfolder": "",
                                "type": "temp",
                            }
                        ]
                    }
                }
            }
        }
        communicator.get_media_file = lambda filename, subfolder, folder_type: (  # type: ignore[method-assign]
            b"preview-bytes"
        )

        output_dir = Path(tempfile.mkdtemp())
        success, saved_files = communicator.save_results("prompt-1", str(output_dir), "render")

        self.assertTrue(success)
        self.assertEqual(len(saved_files), 1)
        self.assertEqual(Path(saved_files[0]).read_bytes(), b"preview-bytes")

    def test_free_memory_uses_comfy_endpoint_with_timeout(self) -> None:
        communicator = AgenticComfyCommunicator(host="127.0.0.1", port=8188)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        captured: dict[str, object] = {}

        def fake_urlopen(req, timeout):
            captured["request"] = req
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("agentic.tools.comfy_backend.request.urlopen", side_effect=fake_urlopen):
            communicator.free_memory()

        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(captured["request"].get_method(), "POST")  # type: ignore[union-attr]
        self.assertIn(b"unload_models", captured["request"].data)  # type: ignore[union-attr]


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
    def test_facebook_profile_handoff_creates_mobile_package_without_graph_publisher(self) -> None:
        output_dir = Path(tempfile.mkdtemp())
        media_path = output_dir / "clip.mp4"
        media_path.write_bytes(b"test media")
        tools = SocialServiceTools(output_dir)

        result = tools.publish_social(
            {
                "media_paths": [str(media_path)],
                "caption": "Profile caption",
                "hashtags": "#kirby",
                "platforms": ["facebook_profile_handoff"],
                "platform_configs": {},
                "additional_params": {
                    "facebook_profile_share_url": "https://cdn.example.test/clip.mp4"
                },
                "manifest_dir": str(output_dir),
            }
        )

        self.assertEqual(result["status"], "awaiting_user_action")
        self.assertEqual(result["publication_state"], "awaiting_user_action")
        self.assertFalse(result["publicly_visible"])
        self.assertEqual(result["handoff_media_paths"], [str(media_path.resolve())])
        handoff = result["handoffs"]["facebook_profile_handoff"]
        self.assertTrue(handoff["requires_human_confirmation"])
        self.assertIn("sharer/sharer.php?u=", handoff["share_dialog_url"])
        handoff_file = output_dir / "facebook_profile_handoff.json"
        self.assertTrue(handoff_file.exists())
        persisted_handoff = json.loads(handoff_file.read_text(encoding="utf-8"))
        self.assertIn("receipt", persisted_handoff)
        self.assertEqual(result["handoff_receipts"]["facebook_profile_handoff"]["verified"], False)
        self.assertIn(str(handoff_file.resolve()), result["handoff_attachment_paths"])

    def test_facebook_profile_handoff_records_delivery_and_manual_post_receipt(self) -> None:
        output_dir = Path(tempfile.mkdtemp())
        media_path = output_dir / "clip.mp4"
        media_path.write_bytes(b"test media")
        result = SocialServiceTools(output_dir).publish_social(
            {
                "media_paths": [str(media_path)],
                "caption": "Profile caption",
                "platforms": ["facebook_profile_handoff"],
                "manifest_dir": str(output_dir),
            }
        )
        handoff_path = str(output_dir / "facebook_profile_handoff.json")
        delivery = record_facebook_profile_handoff_delivery(
            handoff_path,
            {
                "status": "sent",
                "message_id": "discord-message-1",
                "attachment_count": len(result["handoff_attachment_paths"]),
            },
            expected_attachment_count=len(result["handoff_attachment_paths"]),
        )
        self.assertTrue(delivery["delivery"]["delivered"])
        completed = complete_facebook_profile_handoff(
            handoff_path,
            "https://www.facebook.com/share/p/profile-post-1",
        )
        self.assertEqual(completed["publication_state"], "published")
        manifest = json.loads((output_dir / "publish_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["publicly_visible"])
        self.assertEqual(
            manifest["handoff_receipts"]["facebook_profile_handoff"]["external_id"],
            "https://www.facebook.com/share/p/profile-post-1",
        )

    def test_facebook_profile_handoff_rejects_invalid_share_url(self) -> None:
        output_dir = Path(tempfile.mkdtemp())
        media_path = output_dir / "clip.mp4"
        media_path.write_bytes(b"test media")

        result = SocialServiceTools(output_dir).publish_social(
            {
                "media_paths": [str(media_path)],
                "caption": "Profile caption",
                "platforms": ["facebook_profile_handoff"],
                "additional_params": {"facebook_profile_share_url": "local-file"},
            }
        )
        self.assertEqual(result["status"], "partial_failure")
        self.assertIn("https URL", result["errors"]["facebook_profile_handoff"])

    def test_publish_social_skips_youtube_for_image_only_media(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                del post, platforms
                raise AssertionError("YouTube must not receive image-only media")

        tools._publishing = FakePublishing()  # type: ignore[assignment]
        result = tools.publish_social(
            {
                "media_paths": ["still.png"],
                "caption": "image post",
                "platforms": ["youtube"],
                "platform_configs": {"youtube": {"config_folder_path": "configs/yt"}},
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["dispatch_status"], "skipped")
        self.assertEqual(result["skipped_platforms"], {"youtube": "requires_video_media"})
        self.assertEqual(result["publication_state"], "not_applicable")

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

    def test_publish_social_writes_external_receipts_to_manifest(self) -> None:
        tools = SocialServiceTools(Path(tempfile.mkdtemp()))

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                del post
                platform_name = (platforms or ["facebook"])[0]
                return {platform_name: True}

            def get_publish_receipts(self) -> dict[str, dict[str, object]]:
                return {
                    "facebook": {
                        "platform": "facebook",
                        "external_id": "video-1",
                        "status": "draft_ready",
                        "verified": True,
                        "visibility": "draft",
                    }
                }

        manifest_dir = Path(tempfile.mkdtemp())
        tools._publishing = FakePublishing()  # type: ignore[assignment]
        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "manifest test",
                "platforms": ["facebook"],
                "platform_configs": {"facebook": {"config_folder_path": "configs/fb"}},
                "platform_bundle": {
                    "facebook": {
                        "caption": "manifest test",
                        "hashtags": "#kirby",
                        "format": "reel",
                        "additional_params": {"facebook_use_reels": True},
                        "content_strategy": {
                            "strategy_version": "2026-08-25.v1",
                            "platform": "facebook",
                            "format": "reel",
                        },
                        "validation": {
                            "is_publish_ready": True,
                            "is_platform_publish_ready": True,
                        },
                    }
                },
                "manifest_dir": str(manifest_dir),
            }
        )

        manifest_path = Path(str(result["manifest_path"]))
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["publish_receipts"]["facebook"]["external_id"], "video-1")
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["publication_state"], "staged")
        self.assertFalse(manifest["publicly_visible"])
        self.assertEqual(manifest["platform_content"]["facebook"]["format"], "reel")
        self.assertEqual(
            manifest["platform_content"]["facebook"]["derived_metadata"]["facebook_use_reels"],
            True,
        )

    def test_publish_social_reports_public_only_after_verified_public_receipts(self) -> None:
        tools = SocialServiceTools(Path(tempfile.mkdtemp()))

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                del post
                platform_name = (platforms or ["youtube"])[0]
                return {platform_name: True}

            def get_publish_receipts(self) -> dict[str, dict[str, object]]:
                return {
                    "youtube": {
                        "platform": "youtube",
                        "external_id": "public-video-1",
                        "status": "processed",
                        "verified": True,
                        "visibility": "public",
                    }
                }

        tools._publishing = FakePublishing()  # type: ignore[assignment]
        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "public test",
                "platforms": ["youtube"],
                "platform_configs": {"youtube": {"config_folder_path": "configs/yt"}},
            }
        )

        self.assertEqual(result["publication_state"], "published")
        self.assertTrue(result["publicly_visible"])


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

        with tempfile.TemporaryDirectory() as directory:
            image_path = str(Path(directory) / "frame.jpg")
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


class SocialServiceToolsYouTubeTests(unittest.TestCase):
    def test_publish_social_passes_platform_bundle_additional_params_to_youtube(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        captured_posts: dict[str, object] = {}

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
                return media_paths

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                platform_name = (platforms or ["youtube"])[0]
                captured_posts[platform_name] = post
                return {platform_name: True}

        tools._publishing = FakePublishing()  # type: ignore[assignment]

        result = tools.publish_social(
            {
                "media_paths": ["video.mp4"],
                "caption": "Generic caption",
                "hashtags": "#kirby",
                "platforms": ["youtube", "instagram_graph"],
                "platform_configs": {
                    "youtube": {"config_folder_path": "configs/yt"},
                    "instagram_graph": {"config_folder_path": "configs/ig"},
                },
                "platform_bundle": {
                    "youtube": {
                        "caption": "YouTube caption",
                        "hashtags": "#kirby #youtube",
                        "media_paths": ["video.mp4"],
                        "additional_params": {
                            "youtube_title": "Kirby Neon Adventure",
                            "youtube_tags": ["kirby", "animation"],
                        },
                    },
                    "instagram_graph": {
                        "caption": "IG caption",
                        "hashtags": "#kirby",
                        "media_paths": ["video.mp4"],
                    },
                },
            }
        )

        self.assertEqual(result["status"], "success")
        yt_post = captured_posts["youtube"]
        self.assertEqual(yt_post.additional_params.get("youtube_title"), "Kirby Neon Adventure")  # type: ignore[union-attr]
        self.assertEqual(yt_post.additional_params.get("youtube_tags"), ["kirby", "animation"])  # type: ignore[union-attr]
        ig_post = captured_posts["instagram_graph"]
        self.assertNotIn("youtube_title", ig_post.additional_params or {})  # type: ignore[operator]

    def test_publish_social_platform_bundle_additional_params_merged_with_global(self) -> None:
        tools = SocialServiceTools(Path.cwd())

        captured_posts: dict[str, object] = {}

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
                return media_paths

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                platform_name = (platforms or ["youtube"])[0]
                captured_posts[platform_name] = post
                return {platform_name: True}

        tools._publishing = FakePublishing()  # type: ignore[assignment]

        tools.publish_social(
            {
                "media_paths": ["video.mp4"],
                "caption": "caption",
                "hashtags": "#kirby",
                "platforms": ["youtube"],
                "platform_configs": {"youtube": {"config_folder_path": "configs/yt"}},
                "additional_params": {"youtube_made_for_kids": False},
                "platform_bundle": {
                    "youtube": {
                        "caption": "YT caption",
                        "media_paths": ["video.mp4"],
                        "additional_params": {"youtube_title": "Title Override"},
                    }
                },
            }
        )

        yt_post = captured_posts["youtube"]
        self.assertEqual(yt_post.additional_params.get("youtube_title"), "Title Override")  # type: ignore[union-attr]
        self.assertFalse(yt_post.additional_params.get("youtube_made_for_kids"))  # type: ignore[union-attr]

    def test_publish_social_explicit_params_override_derived_platform_defaults(self) -> None:
        tools = SocialServiceTools(Path.cwd())
        captured_posts: dict[str, object] = {}

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                platform_name = (platforms or ["youtube"])[0]
                captured_posts[platform_name] = post
                return {platform_name: True}

        tools._publishing = FakePublishing()  # type: ignore[assignment]
        tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "caption",
                "platforms": ["youtube"],
                "platform_configs": {"youtube": {"config_folder_path": "configs/yt"}},
                "additional_params": {
                    "youtube_title": "Explicit reviewed title",
                    "youtube_contains_synthetic_media": False,
                },
                "platform_bundle": {
                    "youtube": {
                        "metadata_source": "derived",
                        "content_strategy": {"synthetic_media_disclosed": True},
                        "additional_params": {
                            "youtube_title": "Derived fallback title",
                            "youtube_contains_synthetic_media": True,
                        },
                        "validation": {
                            "is_publish_ready": True,
                            "is_platform_publish_ready": True,
                        },
                    }
                },
            }
        )

        yt_post = captured_posts["youtube"]
        self.assertEqual(
            yt_post.additional_params.get("youtube_title"),  # type: ignore[union-attr]
            "Explicit reviewed title",
        )
        self.assertTrue(
            yt_post.additional_params.get("youtube_contains_synthetic_media")  # type: ignore[union-attr]
        )

    def test_safe_poc_overrides_platform_metadata_and_does_not_enable_x(self) -> None:
        tools = SocialServiceTools(Path.cwd())
        captured_posts: dict[str, object] = {}

        class FakePublishing:
            def register_platform(self, platform_name: str, platform_config: dict[str, object]) -> None:
                del platform_name, platform_config

            def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
                return media_paths

            def publish(self, post: object, platforms: list[str] | None = None) -> dict[str, bool]:
                platform_name = (platforms or ["youtube"])[0]
                captured_posts[platform_name] = post
                return {platform_name: True}

        tools._publishing = FakePublishing()  # type: ignore[assignment]
        result = tools.publish_social(
            {
                "media_paths": ["clip.mp4"],
                "caption": "safe POC",
                "platforms": ["youtube", "facebook", "instagram_graph"],
                "platform_configs": {
                    "youtube": {"config_folder_path": "configs/yt"},
                    "facebook": {"config_folder_path": "configs/fb"},
                    "instagram_graph": {"config_folder_path": "configs/ig"},
                },
                "publish_mode": "safe_poc",
                "platform_bundle": {
                    "youtube": {"additional_params": {"youtube_privacy_status": "public"}},
                    "facebook": {"additional_params": {"facebook_video_state": "PUBLISHED"}},
                },
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(captured_posts["youtube"].additional_params["youtube_privacy_status"], "private")  # type: ignore[union-attr]
        self.assertEqual(captured_posts["facebook"].additional_params["facebook_video_state"], "DRAFT")  # type: ignore[union-attr]
        self.assertTrue(captured_posts["facebook"].additional_params["facebook_use_reels"])  # type: ignore[union-attr]
        self.assertEqual(captured_posts["instagram_graph"].additional_params["instagram_publish_mode"], "container_only")  # type: ignore[union-attr]


class FacebookPlatformTests(unittest.TestCase):
    @patch.object(FacebookPlatform, "authenticate")
    @patch.object(FacebookPlatform, "load_config")
    def test_video_defaults_to_reels_when_no_flag_is_configured(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = FacebookPlatform("configs/social_media/credentials/kirby")
        platform.page_access_token = "token"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            video_path = handle.name
        self.addCleanup(lambda: Path(video_path).unlink(missing_ok=True))

        with patch.object(platform, "_upload_reel", return_value=True) as upload_reel:
            self.assertTrue(platform.upload_post(MediaPost(media_paths=[video_path], caption="caption")))
        upload_reel.assert_called_once()

    @patch.object(FacebookPlatform, "authenticate")
    @patch.object(FacebookPlatform, "load_config")
    def test_reels_finishes_upload_without_waiting_for_ready(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = FacebookPlatform("configs/social_media/credentials/kirby")
        platform.page_access_token = "token"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            video_path = handle.name
        self.addCleanup(lambda: Path(video_path).unlink(missing_ok=True))

        platform.ffmpeg.probe_media = lambda _path: {"duration": 15.0, "width": 540, "height": 960}  # type: ignore[method-assign]

        class FakeResponse:
            def __init__(self, body: dict[str, object]) -> None:
                self.body = body

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return self.body

        responses = [
            FakeResponse({"video_id": "video-1", "upload_url": "https://upload.test/video-1"}),
            FakeResponse({"success": True}),
            FakeResponse({"success": True}),
        ]
        status_response = FakeResponse(
            {
                "id": "video-1",
                "status": {
                    "video_status": "ready",
                    "uploading_phase": {"status": "complete"},
                    "processing_phase": {"status": "complete"},
                    "publishing_phase": {"status": "not_started"},
                },
            }
        )
        with patch("agentic.tools.social_native.requests.post", side_effect=responses) as post_mock, patch(
            "agentic.tools.social_native.requests.get", return_value=status_response
        ) as get_mock:
            result = platform._upload_reel(video_path, "caption", {"facebook_video_state": "DRAFT"})

        self.assertTrue(result)
        self.assertEqual(platform.last_publish_receipt["external_id"], "video-1")  # type: ignore[index]
        self.assertTrue(platform.last_publish_receipt["verified"])  # type: ignore[index]
        self.assertEqual(post_mock.call_count, 3)
        finish_params = post_mock.call_args_list[2].kwargs["params"]
        self.assertEqual(finish_params["upload_phase"], "finish")
        self.assertEqual(finish_params["video_state"], "DRAFT")
        get_mock.assert_called_once()

    @patch.object(FacebookPlatform, "authenticate")
    @patch.object(FacebookPlatform, "load_config")
    def test_reels_reconciles_finish_http_error_when_external_draft_is_ready(
        self,
        load_config_mock,
        authenticate_mock,
    ) -> None:
        del load_config_mock, authenticate_mock
        platform = FacebookPlatform("configs/social_media/credentials/kirby")
        platform.page_access_token = "token"
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            video_path = handle.name
        self.addCleanup(lambda: Path(video_path).unlink(missing_ok=True))

        platform.ffmpeg.probe_media = lambda _path: {"duration": 15.0, "width": 540, "height": 960}  # type: ignore[method-assign]

        class FakeResponse:
            def __init__(self, body: dict[str, object], status_code: int = 200, error: bool = False) -> None:
                self.body = body
                self.status_code = status_code
                self.error = error

            def raise_for_status(self) -> None:
                if self.error:
                    raise requests.HTTPError("finish transient error")

            def json(self) -> dict[str, object]:
                return self.body

        responses = [
            FakeResponse({"video_id": "video-2", "upload_url": "https://upload.test/video-2"}),
            FakeResponse({"success": True}),
            FakeResponse({"error": {"code": 1, "error_subcode": 99}}, status_code=500, error=True),
        ]
        status_response = FakeResponse(
            {
                "id": "video-2",
                "status": {
                    "video_status": "ready",
                    "uploading_phase": {"status": "complete"},
                    "processing_phase": {"status": "complete"},
                    "publishing_phase": {"status": "complete", "publish_status": "draft"},
                },
            }
        )
        with patch("agentic.tools.social_native.requests.post", side_effect=responses), patch(
            "agentic.tools.social_native.requests.get", return_value=status_response
        ):
            result = platform._upload_reel(video_path, "caption", {"facebook_video_state": "DRAFT"})

        self.assertTrue(result)
        self.assertEqual(platform.last_publish_receipt["external_id"], "video-2")  # type: ignore[index]
        self.assertEqual(platform.last_publish_receipt["status"], "draft_ready")  # type: ignore[index]
        self.assertTrue(platform.last_publish_receipt["details"]["reconciled_after_finish_error"])  # type: ignore[index]


class YouTubePlatformTests(unittest.TestCase):
    @patch.object(YouTubePlatform, "authenticate")
    @patch.object(YouTubePlatform, "load_config")
    def test_build_video_request_derives_metadata_from_caption(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = YouTubePlatform("configs/social_media/credentials/kirby")
        platform.default_privacy_status = "private"
        platform.default_category_id = "22"
        platform.default_notify_subscribers = False
        platform.default_made_for_kids = False
        platform.default_contains_synthetic_media = True

        body, notify_subscribers = platform._build_video_request(
            MediaPost(media_paths=["clip.mp4"], caption="Kirby launch\nWith neon rain", hashtags="#kirby #mediaoverload"),
            "clip.mp4",
        )

        self.assertEqual(body["snippet"]["title"], "Kirby launch")
        self.assertIn("With neon rain", body["snippet"]["description"])
        self.assertEqual(body["snippet"]["categoryId"], "22")
        self.assertEqual(body["status"]["privacyStatus"], "private")
        self.assertTrue(body["status"]["containsSyntheticMedia"])
        self.assertEqual(body["snippet"]["tags"], ["kirby", "mediaoverload"])
        self.assertFalse(notify_subscribers)

    @patch.object(YouTubePlatform, "authenticate")
    @patch.object(YouTubePlatform, "load_config")
    def test_upload_post_requires_verified_video_id(self, load_config_mock, authenticate_mock) -> None:
        del load_config_mock, authenticate_mock
        platform = YouTubePlatform("configs/social_media/credentials/kirby")
        platform.default_privacy_status = "private"
        platform.channel_id = "channel-1"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            video_path = handle.name
        self.addCleanup(lambda: Path(video_path).unlink(missing_ok=True))

        class FakeRequest:
            def next_chunk(self):
                return None, {
                    "id": "yt-video-1",
                    "snippet": {"channelId": "channel-1"},
                    "status": {"privacyStatus": "private"},
                }

        class FakeVideos:
            def insert(self, **kwargs):
                del kwargs
                return FakeRequest()

            def list(self, **kwargs):
                del kwargs

                class FakeExecute:
                    @staticmethod
                    def execute():
                        return {
                            "items": [
                                {
                                    "id": "yt-video-1",
                                    "snippet": {"channelId": "channel-1"},
                                    "status": {"uploadStatus": "processed", "privacyStatus": "private"},
                                    "processingDetails": {"processingStatus": "succeeded"},
                                }
                            ]
                        }

                return FakeExecute()

        class FakeService:
            def videos(self):
                return FakeVideos()

        platform.service = FakeService()
        platform._build_media_upload = lambda _path: object()  # type: ignore[method-assign]

        result = platform.upload_post(MediaPost(media_paths=[video_path], caption="verify"))

        self.assertTrue(result)
        self.assertEqual(platform.last_publish_receipt["external_id"], "yt-video-1")  # type: ignore[index]
        self.assertTrue(platform.last_publish_receipt["verified"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
