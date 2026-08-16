from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agentic.tools.comfy_workflow_tool import ComfyWorkflowSpec, ComfyWorkflowToolset, NodeBinding


class _FakeGenerator:
    def __init__(self) -> None:
        self.upload_calls: list[str] = []
        self.generate_calls: list[dict[str, object]] = []

    def upload_image(self, image_path: str) -> str:
        self.upload_calls.append(image_path)
        return "uploaded_input.png"

    def generate(
        self,
        *,
        workflow_path: str,
        updates: list[dict[str, object]],
        output_dir: str,
        file_prefix: str,
    ) -> list[str]:
        self.generate_calls.append(
            {
                "workflow_path": workflow_path,
                "updates": updates,
                "output_dir": output_dir,
                "file_prefix": file_prefix,
            }
        )
        saved_file = Path(output_dir) / f"{file_prefix}.png"
        saved_file.parent.mkdir(parents=True, exist_ok=True)
        saved_file.write_text("stub", encoding="utf-8")
        return [str(saved_file)]


class _OomOnceGenerator(_FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False
        self.communicator = SimpleNamespace(free_memory=self._free_memory)
        self.free_memory_calls = 0

    def _free_memory(self) -> None:
        self.free_memory_calls += 1

    def generate(
        self,
        *,
        workflow_path: str,
        updates: list[dict[str, object]],
        output_dir: str,
        file_prefix: str,
    ) -> list[str]:
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("Allocation on device 0 would exceed allowed memory (out of memory)")
        return super().generate(
            workflow_path=workflow_path,
            updates=updates,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )


class _FakeAdapter:
    def __init__(self) -> None:
        self.generator = _FakeGenerator()
        self.last_generate_updates: dict[str, object] | None = None

    def load_workflow(self, workflow_path: Path | str) -> dict[str, object]:
        return {"workflow_path": str(workflow_path)}

    def build_generator(self, host: str | None = None, port: int | None = None) -> _FakeGenerator:
        del host, port
        return self.generator

    def resolve_alias(self, workflow_path: str, alias: str) -> str | None:
        del workflow_path
        return {"positive_prompt": "42", "load_image": "84"}.get(alias)

    def generate_updates(
        self,
        *,
        workflow: dict[str, object],
        updates_config: list[dict[str, object]],
        description: str | None,
        seed: int | None,
        workflow_path: str,
    ) -> list[dict[str, object]]:
        self.last_generate_updates = {
            "workflow": workflow,
            "updates_config": updates_config,
            "description": description,
            "seed": seed,
            "workflow_path": workflow_path,
        }
        return updates_config


class _FakeAssetRegistry:
    def __init__(self, temp_root: Path) -> None:
        self.temp_root = temp_root
        self.manifest_requests: list[str] = []

    def get_manifest(self, workflow_name: str) -> SimpleNamespace:
        self.manifest_requests.append(workflow_name)
        return SimpleNamespace(name=workflow_name)

    def materialize_workflow(self, manifest: SimpleNamespace) -> Path:
        workflow_path = self.temp_root / f"{manifest.name}.json"
        workflow_path.write_text("{}", encoding="utf-8")
        return workflow_path


class ComfyWorkflowToolsetTests(unittest.TestCase):
    def test_execute_retries_same_workflow_after_provider_oom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            adapter = _FakeAdapter()
            adapter.generator = _OomOnceGenerator()
            toolset.adapter = adapter
            toolset.specs = {
                "comfy.workflow.image_to_video": ComfyWorkflowSpec(
                    name="comfy.workflow.image_to_video",
                    workflow_name="demo",
                    output_folder="videos",
                    file_prefix="agentic_i2v",
                    count_payload_key="video_count",
                )
            }
            with patch.object(toolset, "_check_server"):
                result = toolset.execute(
                    "comfy.workflow.image_to_video",
                    {"workflow_name": "demo", "run_dir": str(temp_root), "video_count": 1},
                )

        self.assertEqual(result["memory_retry_count"], 1)
        self.assertEqual(adapter.generator.free_memory_calls, 1)
        self.assertEqual(len(result["saved_files"]), 1)

    def test_ref2va_model_profile_overrides_loader_nodes_without_mutating_template(self) -> None:
        workflow = {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "q4.gguf"},
            },
            "2": {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {"clip_name": "q4.gguf", "type": "minimax"},
            },
        }
        overrides = ComfyWorkflowToolset._resolve_model_overrides(
            ComfyWorkflowSpec(
                name="comfy.workflow.reference_to_video",
                workflow_name="minimax_h3_ref2va",
                output_folder="videos",
                file_prefix="ref2va",
                reference_conditioning_node_type="MiniMaxH3ReferenceToVideo",
            ),
            {"model_profile": "native"},
        )

        runtime = ComfyWorkflowToolset._apply_model_overrides(workflow, overrides)

        self.assertEqual(workflow["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(runtime["1"]["class_type"], "UNETLoader")
        self.assertEqual(runtime["1"]["inputs"]["unet_name"], "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        self.assertEqual(runtime["2"]["class_type"], "CLIPLoader")
        self.assertEqual(runtime["2"]["inputs"]["clip_name"], "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")

    def test_ref2va_runtime_reference_graph_keeps_model_profile_overrides(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        workflow = json.loads((repo_root / "configs" / "workflow" / "minimax_h3_ref2va.json").read_text(encoding="utf-8"))
        spec = ComfyWorkflowSpec(
            name="comfy.workflow.reference_to_video",
            workflow_name="minimax_h3_ref2va",
            output_folder="videos",
            file_prefix="ref2va",
            reference_conditioning_node_type="MiniMaxH3ReferenceToVideo",
        )
        runtime = ComfyWorkflowToolset._apply_model_overrides(
            workflow,
            ComfyWorkflowToolset._resolve_model_overrides(spec, {"model_profile": "native"}),
        )
        runtime = ComfyWorkflowToolset._build_runtime_reference_workflow(
            runtime,
            [{"type": "image", "path": r"C:\reference.png"}],
            {"width": 608, "height": 352},
        )
        self.assertEqual(runtime["1"]["class_type"], "UNETLoader")
        self.assertEqual(runtime["2"]["class_type"], "CLIPLoader")
        self.assertEqual(runtime["5"]["inputs"]["ref_images.ref_image_0"], ["36", 0])

    def test_first_only_native_h3_clears_template_last_frame_connection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            workflow_path = temp_root / "workflow.json"
            workflow_path.write_text("{}", encoding="utf-8")
            spec = ComfyWorkflowSpec(
                name="comfy.workflow.image_to_video",
                workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
                output_folder="videos",
                file_prefix="agentic_i2v",
                prompt_binding=NodeBinding(kind="prompt", node_type="MiniMaxH3ImageToVideo", input_key="prompt"),
                image_binding=NodeBinding(kind="image", node_type="LoadImage", input_key="image"),
                last_frame_binding=NodeBinding(kind="last_frame", node_type="MiniMaxH3ImageToVideo", input_key="last_frame"),
            )

            updates = toolset._build_updates(
                spec,
                workflow_path,
                {
                    "prompt": "The hero protects the seed",
                    "image_path": r"C:\opening.png",
                    "use_last_frame": False,
                },
                toolset.adapter.generator,
            )

        self.assertEqual(updates[-1], {"node_type": "MiniMaxH3ImageToVideo", "node_index": 0, "inputs": {"last_frame": None}})
        self.assertEqual(updates[1]["filter"], {"title": "native 15s first frame"})
        self.assertEqual(toolset.adapter.generator.upload_calls, [r"C:\opening.png"])

    def test_build_updates_uses_alias_binding_and_uploaded_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            workflow_path = temp_root / "workflow.json"
            workflow_path.write_text("{}", encoding="utf-8")
            spec = ComfyWorkflowSpec(
                name="comfy.workflow.image_to_video",
                workflow_name="image_to_video_test",
                output_folder="videos",
                file_prefix="agentic_i2v",
                count_payload_key="video_count",
                prompt_binding=NodeBinding(kind="prompt", alias="positive_prompt", input_key="value"),
                image_binding=NodeBinding(kind="image", alias="load_image", node_type="LoadImage", input_key="image"),
            )

            updates = toolset._build_updates(
                spec,
                workflow_path,
                {"prompt": "Kirby jumps", "input_image_path": "C:\\input.png", "seed": 321},
                toolset.adapter.generator,
            )

        self.assertEqual(toolset.adapter.generator.upload_calls, ["C:\\input.png"])
        self.assertEqual(
            updates,
            [
                {"node_id": "42", "inputs": {"value": "Kirby jumps"}},
                {"node_id": "84", "inputs": {"image": "uploaded_input.png"}},
            ],
        )
        self.assertEqual(toolset.adapter.last_generate_updates["seed"], 321)

    def test_execute_respects_requested_workflow_and_writes_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            toolset.specs = {
                "comfy.workflow.text_to_image": ComfyWorkflowSpec(
                    name="comfy.workflow.text_to_image",
                    workflow_name="nova_model_plus_z_image_anime",
                    output_folder="images",
                    file_prefix="agentic_image",
                    prompt_binding=NodeBinding(kind="prompt", node_type="PrimitiveString", title="positive"),
                )
            }

            with patch.object(toolset, "_check_server", return_value=None):
                result = toolset.execute(
                    "comfy.workflow.text_to_image",
                    {
                        "run_dir": temp_root / "run-1",
                        "workflow_name": "anima_anime",
                        "prompt": "Kirby in rainy Taipei",
                    },
                )

            summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

            self.assertEqual(asset_registry.manifest_requests, ["anima_anime"])
            self.assertEqual(result["workflow_name"], "anima_anime")
            self.assertEqual(summary["requested_workflow_name"], "anima_anime")
            self.assertEqual(summary["workflow_name"], "anima_anime")
            self.assertEqual(summary["payload"]["run_dir"], str(temp_root / "run-1"))
            self.assertEqual(toolset.adapter.generator.generate_calls[0]["file_prefix"], "agentic_image")
            self.assertTrue(Path(result["saved_files"][0]).exists())

    def test_ref2va_binds_image_upload_and_local_video_path_without_audio(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            image = temp_root / "identity.png"
            video = temp_root / "motion.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            workflow_path = temp_root / "workflow.json"
            workflow_path.write_text("{}", encoding="utf-8")
            spec = ComfyWorkflowSpec(
                name="comfy.workflow.reference_to_video",
                workflow_name="minimax_h3_ref2va",
                output_folder="videos",
                file_prefix="ref2va",
                prompt_binding=NodeBinding(kind="prompt", node_type="MiniMaxH3ReferenceToVideo", input_key="prompt"),
                reference_image_bindings=(
                    NodeBinding(kind="reference_image", node_type="LoadImage", title="H3 reference image 1", input_key="image"),
                    NodeBinding(kind="reference_image", node_type="LoadImage", title="H3 reference image 2", input_key="image"),
                ),
                reference_video_bindings=(
                    NodeBinding(kind="reference_video", node_type="VHS_LoadVideoPath", title="H3 reference video 1", input_key="video"),
                    NodeBinding(kind="reference_video", node_type="VHS_LoadVideoPath", title="H3 reference video 2", input_key="video"),
                ),
                reference_conditioning_node_type="MiniMaxH3ReferenceToVideo",
            )
            updates = toolset._build_updates(
                spec,
                workflow_path,
                {
                    "prompt": "Ref2VA scene",
                    "reference_manifest": [
                        {"path": str(image), "type": "image", "role": "identity"},
                        {"path": str(video), "type": "video", "role": "motion"},
                    ],
                },
                toolset.adapter.generator,
            )

        self.assertEqual(toolset.adapter.generator.upload_calls, [str(image)])
        self.assertIn(
            {"node_type": "LoadImage", "node_index": 0, "inputs": {"image": "uploaded_input.png"}, "filter": {"title": "H3 reference image 1"}},
            updates,
        )
        self.assertIn(
            {"node_type": "VHS_LoadVideoPath", "node_index": 0, "inputs": {"video": str(video)}, "filter": {"title": "H3 reference video 1"}},
            updates,
        )
        self.assertFalse(any(None in update.get("inputs", {}).values() for update in updates))
        self.assertEqual(
            [update["inputs"] for update in updates if update.get("node_type") == "MiniMaxH3ReferenceToVideo"],
            [{"prompt": "Ref2VA scene"}],
        )


class AgenticNodeManagerCustomUpdatesTests(unittest.TestCase):
    """Regression tests for _generate_custom_updates PrimitiveString→CLIPTextEncode fallback."""

    # Minimal workflow that only has CLIPTextEncode nodes (like anima_anime.json)
    CLIP_ONLY_WORKFLOW: dict = {
        "11": {
            "inputs": {"text": "original positive", "clip": ["45", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
        },
        "12": {
            "inputs": {"text": "original negative", "clip": ["45", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
        },
    }

    def _updates_for(self, prompt: str, negative: str) -> list[dict]:
        from agentic.tools.comfy_backend import AgenticNodeManager
        return AgenticNodeManager._generate_custom_updates(
            self.CLIP_ONLY_WORKFLOW,
            [
                {"node_type": "PrimitiveString", "node_index": 0, "inputs": {"value": prompt}, "filter": {"title": "positive"}},
                {"node_type": "PrimitiveString", "node_index": 0, "inputs": {"value": negative}, "filter": {"title": "negative"}},
            ],
        )

    def test_positive_prompt_falls_back_to_clip_text_encode(self) -> None:
        # Regression: anima_anime.json has no PrimitiveString nodes — prompt was silently dropped.
        updates = self._updates_for("injected positive", "injected negative")
        types = [u["type"] for u in updates]
        self.assertIn("CLIPTextEncode", types, "CLIPTextEncode fallback update must be emitted")

    def test_positive_and_negative_are_separate_updates(self) -> None:
        updates = self._updates_for("positive text", "negative text")
        clip_updates = [u for u in updates if u["type"] == "CLIPTextEncode"]
        texts = {u["inputs"]["text"] for u in clip_updates}
        self.assertIn("positive text", texts)
        self.assertIn("negative text", texts)

    def test_positive_update_targets_positive_node(self) -> None:
        from agentic.tools.comfy_backend import AgenticComfyCommunicator
        updates = self._updates_for("MY PROMPT", "ugly")
        workflow_copy = {k: {"inputs": dict(v["inputs"]), "class_type": v["class_type"], "_meta": dict(v["_meta"])} for k, v in self.CLIP_ONLY_WORKFLOW.items()}
        communicator = AgenticComfyCommunicator.__new__(AgenticComfyCommunicator)
        all_nodes = communicator.identify_all_nodes(workflow_copy)
        for update in updates:
            if update.get("type") == "CLIPTextEncode":
                matching = list(all_nodes.get("CLIPTextEncode", []))
                if "title" in update:
                    expected = str(update["title"]).lower()
                    matching = [n for n in matching if expected in n["metadata"].get("title_lower", "")]
                node_index = int(update.get("node_index", 0))
                if node_index < len(matching):
                    node_id = str(matching[node_index]["id"])
                    workflow_copy = communicator.update_node_inputs(workflow_copy, node_id, dict(update.get("inputs", {})))
        self.assertEqual(workflow_copy["11"]["inputs"]["text"], "MY PROMPT")
        self.assertEqual(workflow_copy["12"]["inputs"]["text"], "ugly")


if __name__ == "__main__":
    unittest.main()
