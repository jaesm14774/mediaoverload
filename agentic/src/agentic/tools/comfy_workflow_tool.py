from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic.assets.registry import AssetRegistry
from agentic.assets.kirby_input import assert_kirby_input
from agentic.runtime.registry import ToolRegistry
from agentic.tools.comfy_adapter import ComfyAdapter


@dataclass(slots=True)
class NodeBinding:
    kind: str
    node_type: str | None = None
    node_index: int = 0
    title: str | None = None
    alias: str | None = None
    input_key: str = "value"


@dataclass(slots=True)
class ComfyWorkflowSpec:
    name: str
    workflow_name: str
    output_folder: str
    file_prefix: str
    count_payload_key: str = "image_count"
    prompt_binding: NodeBinding | None = None
    negative_prompt_binding: NodeBinding | None = None
    width_binding: NodeBinding | None = None
    height_binding: NodeBinding | None = None
    length_binding: NodeBinding | None = None
    steps_binding: NodeBinding | None = None
    image_binding: NodeBinding | None = None
    last_image_binding: NodeBinding | None = None
    seed_enabled: bool = True
    default_payload: dict[str, Any] = field(default_factory=dict)


class ComfyWorkflowToolset:
    DEFAULT_IMAGE_WORKFLOWS = ("nova_model_plus_z_image_anime", "nova-anime-xl", "anima_anime")
    DEFAULT_REFINE_WORKFLOWS = ("kirby_identity_img2img", "z_image_i2i_anime", "image_to_image")
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)
    DEFAULT_I2V_WORKFLOWS = ("minimax_h3_lowvram_i2v", "wan2.2_gguf_i2v", "wan2.2_gguf_i2v_audio")

    def __init__(self, asset_registry: AssetRegistry, output_root: Path, comfy_host: str | None = None, comfy_port: int | None = None) -> None:
        self.asset_registry = asset_registry
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.comfy_host = comfy_host or os.environ.get("COMFYUI_HOST") or "127.0.0.1"
        self.comfy_port = comfy_port or self._read_port(os.environ.get("COMFYUI_PORT"))
        self.adapter = ComfyAdapter()
        self.specs = self._build_specs()

    def _preferred_workflow_name(self, *workflow_names: str) -> str:
        for workflow_name in workflow_names:
            try:
                self.asset_registry.get_manifest(workflow_name)
                return workflow_name
            except KeyError:
                continue
        requested = ", ".join(workflow_names)
        raise KeyError(f"None of the preferred workflows are available (configs/workflow): {requested}")

    def register_tools(self, tool_registry: ToolRegistry) -> None:
        for spec_name in self.specs:
            tool_registry.register(
                spec_name,
                self._build_handler(spec_name),
                f"Execute ComfyUI workflow '{spec_name}'",
            )
        tool_registry.register("comfy.render_image", self._build_handler("comfy.workflow.text_to_image"), "Render a real image through ComfyUI")
        tool_registry.register("comfy.render_image_to_image", self._build_handler("comfy.workflow.image_to_image"), "Render a real image-to-image workflow through ComfyUI")
        tool_registry.register("comfy.upscale_image", self._build_handler("comfy.workflow.image_upscale"), "Upscale an image through ComfyUI")
        tool_registry.register("comfy.render_image_to_video", self._build_handler("comfy.workflow.image_to_video"), "Render a real image-to-video workflow through ComfyUI")
        if "comfy.workflow.text_to_video" in self.specs:
            tool_registry.register("comfy.render_text_to_video", self._build_handler("comfy.workflow.text_to_video"), "Render a native H3 text-to-video workflow through ComfyUI")

    def _build_handler(self, spec_name: str):
        def handler(payload: dict[str, object]) -> dict[str, object]:
            return self.execute(spec_name, payload)

        return handler

    def execute(self, spec_name: str, payload: dict[str, object]) -> dict[str, object]:
        spec = self.specs[spec_name]
        merged_payload: dict[str, Any] = {**spec.default_payload, **payload}
        requested_workflow_name = str(merged_payload.get("workflow_name") or spec.workflow_name)
        manifest = self.asset_registry.get_manifest(requested_workflow_name)
        workflow_path = self.asset_registry.materialize_workflow(manifest)
        workflow = self.adapter.load_workflow(workflow_path)
        run_dir = Path(str(merged_payload["run_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)

        self._check_server()

        try:
            generator = self.adapter.build_generator(host=self.comfy_host, port=self.comfy_port)
        except Exception as exc:
            raise RuntimeError(self._connection_error_message()) from exc
        output_dir = run_dir / spec.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        render_count = max(1, int(merged_payload.get(spec.count_payload_key, 1)))
        saved_files: list[str] = []
        try:
            for run_index in range(render_count):
                iteration_payload = dict(merged_payload)
                if spec.seed_enabled and "seed" not in iteration_payload:
                    iteration_payload["seed"] = random.randint(1, 999999999)
                updates = self._build_updates(spec, workflow_path, iteration_payload, generator)
                run_suffix = spec.file_prefix if render_count == 1 else f"{spec.file_prefix}_{run_index + 1:02d}"
                saved_files.extend(
                    generator.generate(
                        workflow_path=str(workflow_path),
                        updates=updates,
                        output_dir=str(output_dir),
                        file_prefix=run_suffix,
                    )
                )
        except Exception as exc:
            raise RuntimeError(f"ComfyUI generation failed for {spec_name}: {exc}") from exc

        summary_path = run_dir / f"{spec.file_prefix}_summary.json"
        summary = {
            "tool_name": spec_name,
            "workflow_name": manifest.name,
            "requested_workflow_name": requested_workflow_name,
            "workflow_path": str(workflow_path),
            "saved_files": saved_files,
            "payload": self._serialize_payload(merged_payload),
            "comfy_host": self.comfy_host or "127.0.0.1",
            "comfy_port": self.comfy_port or 8188,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        result: dict[str, object] = {
            "run_dir": str(run_dir),
            "saved_files": saved_files,
            "summary_path": str(summary_path),
            "workflow_name": manifest.name,
        }
        if not saved_files:
            fallback = merged_payload.get("image_path") or merged_payload.get("input_image_path")
            if fallback:
                result["image_path"] = str(fallback)
        return result

    def _build_updates(
        self,
        spec: ComfyWorkflowSpec,
        workflow_path: Path,
        payload: dict[str, Any],
        generator: Any,
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []

        if spec.prompt_binding and payload.get("prompt"):
            updates.append(self._binding_update(spec.prompt_binding, payload["prompt"], str(workflow_path)))
        if spec.negative_prompt_binding and payload.get("negative_prompt"):
            updates.append(self._binding_update(spec.negative_prompt_binding, payload["negative_prompt"], str(workflow_path)))
        if spec.width_binding and payload.get("width") is not None:
            updates.append(self._binding_update(spec.width_binding, int(payload["width"]), str(workflow_path)))
        if spec.height_binding and payload.get("height") is not None:
            updates.append(self._binding_update(spec.height_binding, int(payload["height"]), str(workflow_path)))
        if spec.length_binding and payload.get("length") is not None:
            updates.append(self._binding_update(spec.length_binding, int(payload["length"]), str(workflow_path)))
        if spec.steps_binding and payload.get("steps") is not None:
            updates.append(self._binding_update(spec.steps_binding, int(payload["steps"]), str(workflow_path)))

        image_path = payload.get("image_path") or payload.get("input_image_path")
        requested_workflow = str(payload.get("workflow_name") or spec.workflow_name)
        if image_path and spec.name == "comfy.workflow.image_to_video" and requested_workflow.startswith("minimax_h3_"):
            prompt_text = str(payload.get("prompt") or "").lower()
            if str(payload.get("character") or "").strip().lower() == "kirby" or "kirby" in prompt_text:
                assert_kirby_input(
                    image_path,
                    allow_external=bool(payload.get("allow_external_reference", False)),
                )
        if spec.image_binding and image_path:
            image_filename = generator.upload_image(str(image_path))
            updates.append(self._binding_update(spec.image_binding, image_filename, str(workflow_path)))

        last_image_path = payload.get("last_image_path") or payload.get("last_frame_path")
        if last_image_path and spec.last_image_binding and requested_workflow.startswith("minimax_h3_"):
            prompt_text = str(payload.get("prompt") or "").lower()
            if str(payload.get("character") or "").strip().lower() == "kirby" or "kirby" in prompt_text:
                assert_kirby_input(
                    last_image_path,
                    allow_external=bool(payload.get("allow_external_reference", False)),
                )
            last_image_filename = generator.upload_image(str(last_image_path))
            updates.append(self._binding_update(spec.last_image_binding, last_image_filename, str(workflow_path)))

        seed = payload.get("seed")
        if seed is None and spec.seed_enabled:
            seed = random.randint(1, 999999999)

        return self.adapter.generate_updates(
            workflow=self.adapter.load_workflow(workflow_path),
            updates_config=updates,
            description=None,
            seed=seed if spec.seed_enabled else None,
            workflow_path=str(workflow_path),
        )

    def _binding_update(self, binding: NodeBinding, value: Any, workflow_path: str) -> dict[str, Any]:
        if binding.alias:
            node_id = self.adapter.resolve_alias(workflow_path, binding.alias)
            if node_id:
                return {
                    "node_id": node_id,
                    "inputs": {binding.input_key: value},
                }

        update: dict[str, Any] = {
            "node_type": binding.node_type,
            "node_index": binding.node_index,
            "inputs": {binding.input_key: value},
        }
        if binding.title:
            update["filter"] = {"title": binding.title}
        return update

    def _build_specs(self) -> dict[str, ComfyWorkflowSpec]:
        image_workflow = self._preferred_workflow_name(*self.DEFAULT_IMAGE_WORKFLOWS)
        refine_workflow = self._preferred_workflow_name(*self.DEFAULT_REFINE_WORKFLOWS)
        upscale_workflow = self._preferred_workflow_name(*self.DEFAULT_UPSCALE_WORKFLOWS)
        i2v_workflow = self._preferred_workflow_name(*self.DEFAULT_I2V_WORKFLOWS)
        return {
            "comfy.workflow.text_to_image": ComfyWorkflowSpec(
                name="comfy.workflow.text_to_image",
                workflow_name=image_workflow,
                output_folder="images",
                file_prefix="agentic_image",
                count_payload_key="image_count",
                prompt_binding=NodeBinding(kind="prompt", node_type="PrimitiveString", title="positive"),
                negative_prompt_binding=NodeBinding(kind="negative_prompt", node_type="PrimitiveString", title="negative"),
                width_binding=NodeBinding(kind="width", node_type="PrimitiveInt", title="width"),
                height_binding=NodeBinding(kind="height", node_type="PrimitiveInt", title="height"),
            ),
            "comfy.workflow.image_to_image": ComfyWorkflowSpec(
                name="comfy.workflow.image_to_image",
                workflow_name=refine_workflow,
                output_folder="img2img",
                file_prefix="agentic_img2img",
                count_payload_key="image_count",
                prompt_binding=NodeBinding(kind="prompt", node_type="PrimitiveString", title="positive"),
                negative_prompt_binding=NodeBinding(kind="negative_prompt", node_type="PrimitiveString", title="negative"),
                image_binding=NodeBinding(kind="image", node_type="LoadImage", input_key="image"),
            ),
            "comfy.workflow.image_upscale": ComfyWorkflowSpec(
                name="comfy.workflow.image_upscale",
                workflow_name=upscale_workflow,
                output_folder="upscaled",
                file_prefix="agentic_upscale",
                count_payload_key="image_count",
                image_binding=NodeBinding(kind="image", alias="load_image", node_type="LoadImage", input_key="image"),
                seed_enabled=False,
            ),
            "comfy.workflow.image_to_video": ComfyWorkflowSpec(
                name="comfy.workflow.image_to_video",
                workflow_name=i2v_workflow,
                output_folder="videos",
                file_prefix="agentic_i2v",
                count_payload_key="video_count",
                prompt_binding=(
                    NodeBinding(kind="prompt", node_type="MiniMaxH3ImageToVideo", input_key="prompt")
                    if i2v_workflow.startswith("minimax_h3_")
                    else NodeBinding(kind="prompt", alias="positive_prompt", node_type="PrimitiveString", title="positive", input_key="value")
                ),
                width_binding=NodeBinding(kind="width", node_type="MiniMaxH3ImageToVideo", input_key="width") if i2v_workflow.startswith("minimax_h3_") else None,
                height_binding=NodeBinding(kind="height", node_type="MiniMaxH3ImageToVideo", input_key="height") if i2v_workflow.startswith("minimax_h3_") else None,
                length_binding=NodeBinding(kind="length", node_type="MiniMaxH3ImageToVideo", input_key="length") if i2v_workflow.startswith("minimax_h3_") else None,
                steps_binding=NodeBinding(kind="steps", node_type="BasicScheduler", input_key="steps") if i2v_workflow.startswith("minimax_h3_") else None,
                image_binding=NodeBinding(kind="image", node_type="LoadImage", input_key="image"),
                last_image_binding=NodeBinding(kind="last_image", node_type="LoadImage", title="native 15s last frame", input_key="image") if i2v_workflow.startswith("minimax_h3_") else None,
            ),
            "comfy.workflow.text_to_video": ComfyWorkflowSpec(
                name="comfy.workflow.text_to_video",
                workflow_name="minimax_h3_lowvram_t2v",
                output_folder="videos",
                file_prefix="agentic_h3_t2v",
                count_payload_key="video_count",
                prompt_binding=NodeBinding(kind="prompt", node_type="MiniMaxH3ImageToVideo", input_key="prompt"),
                width_binding=NodeBinding(kind="width", node_type="MiniMaxH3ImageToVideo", input_key="width"),
                height_binding=NodeBinding(kind="height", node_type="MiniMaxH3ImageToVideo", input_key="height"),
                length_binding=NodeBinding(kind="length", node_type="MiniMaxH3ImageToVideo", input_key="length"),
                steps_binding=NodeBinding(kind="steps", node_type="BasicScheduler", input_key="steps"),
            ),
        }

    def _check_server(self) -> None:
        host = self.comfy_host or "127.0.0.1"
        port = self.comfy_port or 8188
        url = f"http://{host}:{port}/system_stats"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status >= 400:
                    raise RuntimeError(self._connection_error_message())
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            raise RuntimeError(self._connection_error_message()) from exc

    def _connection_error_message(self) -> str:
        host = self.comfy_host or "127.0.0.1"
        port = self.comfy_port or 8188
        return (
            f"ComfyUI is not reachable at {host}:{port}. "
            "Start ComfyUI first, or pass --comfy-host/--comfy-port to agentic."
        )

    @staticmethod
    def _read_port(value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized


def register_comfy_workflow_tools(
    tool_registry: ToolRegistry,
    asset_registry: AssetRegistry,
    output_root: Path,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
) -> None:
    toolset = ComfyWorkflowToolset(
        asset_registry=asset_registry,
        output_root=output_root,
        comfy_host=comfy_host,
        comfy_port=comfy_port,
    )
    toolset.register_tools(tool_registry)
