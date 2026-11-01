from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agentic.tools.comfy_workflow_tool import ComfyWorkflowSpec, ComfyWorkflowToolset, NodeBinding
from agentic.tools.comfy_backend import ComfyUploadResult


class _FakeGenerator:
    def __init__(self) -> None:
        self.upload_calls: list[str] = []
        self.generate_calls: list[dict[str, object]] = []

    def upload_image(self, image_path: str) -> ComfyUploadResult:
        self.upload_calls.append(image_path)
        return ComfyUploadResult("uploaded_input.png", None)

    def generate(
        self,
        *,
        workflow_path: str,
        updates: list[dict[str, object]],
        output_dir: str,
        file_prefix: str,
        workflow: dict[str, object] | None = None,
        timing_sink: list[dict[str, float | bool]] | None = None,
    ) -> list[str]:
        del workflow
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
        if timing_sink is not None:
            timing_sink.append(
                {
                    "websocket_connect_seconds": 0.0,
                    "workflow_prepare_seconds": 0.0,
                    "queue_submit_seconds": 0.0,
                    "execution_wait_seconds": 0.0,
                    "queue_wait_seconds": 0.0,
                    "server_execution_seconds": 0.0,
                    "execution_start_observed": False,
                    "result_retrieval_seconds": 0.0,
                    "total_seconds": 0.0,
                }
            )
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
        workflow: dict[str, object] | None = None,
        timing_sink: list[dict[str, float | bool]] | None = None,
    ) -> list[str]:
        if not self.failed_once:
            self.failed_once = True
            if timing_sink is not None:
                timing_sink.append({"total_seconds": 0.0})
            raise RuntimeError("Allocation on device 0 would exceed allowed memory (out of memory)")
        return super().generate(
            workflow_path=workflow_path,
            updates=updates,
            output_dir=output_dir,
            file_prefix=file_prefix,
            workflow=workflow,
            timing_sink=timing_sink,
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


class _NodeInfoGenerator:
    def __init__(self) -> None:
        self.available = {"SamplerSelect": {"input": {}, "output": []}}

    def get_object_info(self, node_type: str) -> dict[str, object]:
        if node_type == "UnavailableNode":
            raise OSError("ComfyUI metadata request failed")
        metadata = self.available.get(node_type)
        return {node_type: metadata} if metadata is not None else {}


class ComfyWorkflowToolsetTests(unittest.TestCase):

    def test_user_given_a_runtime_workflow_and_model_override_when_overrides_are_applied_then_the_loaded_graph_has_the_requested_model(self) -> None:
        """User Given a freshly loaded runtime workflow and valid model overrides When overrides are applied Then the graph used for generation has the selected class and inputs."""
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "default.safetensors", "vae_name": "default.vae"},
            }
        }

        runtime_workflow = ComfyWorkflowToolset._apply_model_overrides(
            workflow,
            {
                "4": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": "selected.gguf"},
                }
            },
        )

        self.assertEqual(runtime_workflow["4"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(runtime_workflow["4"]["inputs"]["unet_name"], "selected.gguf")
        self.assertEqual(runtime_workflow["4"]["inputs"]["vae_name"], "default.vae")
        self.assertEqual(workflow["4"]["class_type"], "CheckpointLoaderSimple")

    def test_user_given_model_overrides_contain_an_invalid_node_when_applied_then_no_override_is_partially_written(self) -> None:
        """User Given one valid and one invalid model override When the override set is applied Then the runtime graph remains unchanged."""
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "default.safetensors"},
            }
        }

        with self.assertRaisesRegex(ValueError, "unknown workflow node"):
            ComfyWorkflowToolset._apply_model_overrides(
                workflow,
                {
                    "4": {"inputs": {"ckpt_name": "selected.safetensors"}},
                    "missing": {"inputs": {"ckpt_name": "ignored.safetensors"}},
                },
            )

        self.assertEqual(
            workflow["4"]["inputs"]["ckpt_name"], "default.safetensors"
        )

    def test_user_given_required_nodes_include_a_missing_and_unreachable_type_when_preflight_runs_then_it_reports_both(self) -> None:
        """User Given a workflow requires present, missing, and unreachable ComfyUI nodes When preflight runs Then it reports the unavailable node types together."""
        generator = _NodeInfoGenerator()
        spec = ComfyWorkflowSpec(
            name="comfy.workflow.demo",
            workflow_name="demo",
            output_folder="images",
            file_prefix="demo",
            required_node_types=("SamplerSelect", "MissingNode", "UnavailableNode"),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"ComfyUI is missing required node type\(s\): MissingNode, UnavailableNode",
        ):
            ComfyWorkflowToolset._check_required_node_types(generator, spec)

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
                    {"workflow_name": "demo", "run_dir": str(temp_root), "video_count": 1,
                     "prompt": "Kirby crosses a meadow. The camera follows his steps."},
                )

        self.assertEqual(result["memory_retry_count"], 1)
        self.assertEqual(adapter.generator.free_memory_calls, 1)
        self.assertEqual(len(result["saved_files"]), 1)




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
            uploaded_image_filename = toolset.adapter.generator.upload_image(
                "C:\\input.png"
            ).comfy_path

            updates = toolset._build_updates(
                spec,
                workflow_path,
                {"prompt": "Kirby jumps", "input_image_path": "C:\\input.png", "seed": 321},
                image_filename=uploaded_image_filename,
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

    def test_user_can_clear_a_workflow_negative_prompt(self) -> None:
        """User Given a workflow has a default negative prompt, When the user explicitly sends an empty one, Then the default is cleared."""
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            workflow_path = temp_root / "workflow.json"
            workflow_path.write_text("{}", encoding="utf-8")
            spec = ComfyWorkflowSpec(
                name="comfy.workflow.text_to_image",
                workflow_name="image_test",
                output_folder="images",
                file_prefix="agentic_image",
                negative_prompt_binding=NodeBinding(kind="negative_prompt", node_type="PrimitiveString", input_key="value"),
            )

            updates = toolset._build_updates(
                spec,
                workflow_path,
                {"negative_prompt": ""},
                image_filename=None,
            )

        self.assertEqual(
            updates,
            [{"node_type": "PrimitiveString", "node_index": 0, "inputs": {"value": ""}}],
        )

    def test_build_updates_can_override_img2img_denoise_for_story_card_motion(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            asset_registry = _FakeAssetRegistry(temp_root)
            with patch.object(ComfyWorkflowToolset, "_build_specs", return_value={}):
                toolset = ComfyWorkflowToolset(asset_registry=asset_registry, output_root=temp_root)
            toolset.adapter = _FakeAdapter()
            workflow_path = temp_root / "workflow.json"
            workflow_path.write_text("{}", encoding="utf-8")
            spec = ComfyWorkflowSpec(
                name="comfy.workflow.image_to_image",
                workflow_name="krea2_turbo_img2img",
                output_folder="img2img",
                file_prefix="story_card",
                denoise_binding=NodeBinding(kind="denoise", node_type="KSampler", input_key="denoise"),
            )

            updates = toolset._build_updates(
                spec,
                workflow_path,
                {"denoise": 0.4},
                image_filename=None,
            )

        self.assertEqual(
            updates,
            [{"node_type": "KSampler", "node_index": 0, "inputs": {"denoise": 0.4}}],
        )

    def test_execute_respects_requested_workflow_and_writes_summary(self) -> None:
        """User Given a ComfyUI render completes When its result is written Then the output summary preserves the per-phase execution timings."""
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
            self.assertEqual(
                result["execution_timings"], summary["execution_timings"]
            )
            candidate_run = summary["execution_timings"]["candidate_runs"][0]
            self.assertTrue(
                candidate_run["comfy_runs"][0]["queue_submit_seconds"] >= 0
            )



class AgenticNodeManagerCustomUpdatesTests(unittest.TestCase):
    """BDD contracts for prompt binding when workflow node types vary."""

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
        return AgenticNodeManager.generate_updates(
            self.CLIP_ONLY_WORKFLOW,
            updates_config=[
                {"node_type": "PrimitiveString", "node_index": 0, "inputs": {"value": prompt}, "filter": {"title": "positive"}},
                {"node_type": "PrimitiveString", "node_index": 0, "inputs": {"value": negative}, "filter": {"title": "negative"}},
            ],
        )

    def test_positive_prompt_falls_back_to_clip_text_encode(self) -> None:
        # Regression: anima_anime.json has no PrimitiveString nodes — prompt was silently dropped.
        updates = self._updates_for("injected positive", "injected negative")
        types = [u["type"] for u in updates]
        self.assertIn("CLIPTextEncode", types, "CLIPTextEncode fallback update must be emitted")

    def test_user_given_positive_and_negative_prompt_nodes_when_generating_updates_then_each_prompt_targets_its_own_node(self) -> None:
        """User Given a workflow with separate positive and negative prompt nodes When prompt updates are generated Then each value targets its matching node."""
        updates = self._updates_for("positive text", "negative text")
        clip_updates = [u for u in updates if u["type"] == "CLIPTextEncode"]
        texts_by_title = {u["title"]: u["inputs"]["text"] for u in clip_updates}
        self.assertEqual(
            texts_by_title,
            {"positive": "positive text", "negative": "negative text"},
        )

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


def test_user_given_a_changed_workflow_alias_config_when_resolving_again_then_the_updated_node_id_is_returned(
    tmp_path, monkeypatch
):
    """User Given a workflow alias config that changes on disk When the alias is resolved again Then the updated node ID is returned."""
    from agentic.tools.comfy_backend import AgenticNodeManager

    config_path = tmp_path / "configs" / "workflow" / "workflow_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        'workflows:\n  sample.json:\n    node_aliases:\n      positive_prompt: "node-1"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    first = AgenticNodeManager.resolve_alias("sample.json", "positive_prompt")
    config_path.write_text(
        'workflows:\n  sample.json:\n    node_aliases:\n      positive_prompt: "node-2-updated"\n',
        encoding="utf-8",
    )
    second = AgenticNodeManager.resolve_alias("sample.json", "positive_prompt")

    assert first == "node-1"
    assert second == "node-2-updated"


if __name__ == "__main__":
    unittest.main()
