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
    steps_binding: NodeBinding | None = None
    image_binding: NodeBinding | None = None
    seed_enabled: bool = True
    denoise_binding: NodeBinding | None = None
    default_payload: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int | None = None
    required_node_types: tuple[str, ...] = ()


class ComfyWorkflowToolset:
    DEFAULT_IMAGE_WORKFLOWS = ("krea2_turbo",)
    DEFAULT_REFINE_WORKFLOWS = ("krea2_turbo_img2img",)
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)

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
            generator_kwargs: dict[str, Any] = {"host": self.comfy_host, "port": self.comfy_port}
            if spec.timeout_seconds is not None:
                generator_kwargs["timeout"] = spec.timeout_seconds
            generator = self.adapter.build_generator(**generator_kwargs)
        except Exception as exc:
            raise RuntimeError(self._connection_error_message()) from exc
        self._check_required_node_types(generator, spec)
        runtime_workflow = workflow
        model_overrides = self._resolve_model_overrides(spec, merged_payload)
        if model_overrides:
            runtime_workflow = self._apply_model_overrides(runtime_workflow, model_overrides)
            merged_payload["model_overrides"] = model_overrides
        output_dir = run_dir / spec.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        render_count = max(1, int(merged_payload.get(spec.count_payload_key, 1)))
        saved_files: list[str] = []
        render_attempts: list[dict[str, Any]] = []
        memory_retry_count = 0
        base_seed = merged_payload.get("seed")
        if spec.seed_enabled:
            if base_seed is None:
                base_seed = random.randint(1, 999999999)
            base_seed = int(base_seed)
        try:
            for run_index in range(render_count):
                iteration_payload = dict(merged_payload)
                if spec.seed_enabled:
                    iteration_payload["seed"] = (base_seed + run_index) % (2**64)
                updates = self._build_updates(
                    spec,
                    workflow_path,
                    iteration_payload,
                    generator,
                    workflow=runtime_workflow if model_overrides else None,
                )
                run_suffix = spec.file_prefix if render_count == 1 else f"{spec.file_prefix}_{run_index + 1:02d}"
                generate_kwargs = {
                    "workflow_path": str(workflow_path),
                    "updates": updates,
                    "output_dir": str(output_dir),
                    "file_prefix": run_suffix,
                }
                if model_overrides:
                    generate_kwargs["workflow"] = runtime_workflow
                try:
                    rendered_files = generator.generate(**generate_kwargs)
                except Exception as exc:
                    # Recover actual image memory pressure by releasing cached
                    # models, then retry the exact same recipe once;
                    # never switch to another workflow as an implicit fallback.
                    if not self._is_memory_error(exc):
                        raise
                    self._release_comfy_memory(generator)
                    memory_retry_count += 1
                    rendered_files = generator.generate(**generate_kwargs)
                saved_files.extend(rendered_files)
                render_attempts.append(
                    {
                        "candidate_index": run_index + 1,
                        "seed": iteration_payload.get("seed"),
                        "saved_files": list(rendered_files),
                    }
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
            "render_attempts": render_attempts,
            "payload": self._serialize_payload(merged_payload),
            "memory_retry_count": memory_retry_count,
            "comfy_host": self.comfy_host or "127.0.0.1",
            "comfy_port": self.comfy_port or 8188,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        result: dict[str, object] = {
            "run_dir": str(run_dir),
            "saved_files": saved_files,
            "summary_path": str(summary_path),
            "workflow_name": manifest.name,
            "memory_retry_count": memory_retry_count,
        }
        if not saved_files:
            fallback = merged_payload.get("image_path") or merged_payload.get("input_image_path")
            if fallback:
                result["image_path"] = str(fallback)
        return result

    @staticmethod
    def _is_memory_error(error: BaseException) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda out of memory" in message or "allocation on device" in message

    @staticmethod
    def _release_comfy_memory(generator: Any) -> None:
        communicator = getattr(generator, "communicator", None)
        free_memory = getattr(communicator, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory()
            except Exception:
                return

    def _build_updates(
        self,
        spec: ComfyWorkflowSpec,
        workflow_path: Path,
        payload: dict[str, Any],
        generator: Any,
        *,
        workflow: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []

        if spec.prompt_binding and payload.get("prompt"):
            updates.append(self._binding_update(spec.prompt_binding, payload["prompt"], str(workflow_path)))
        if spec.negative_prompt_binding and "negative_prompt" in payload:
            updates.append(self._binding_update(spec.negative_prompt_binding, payload["negative_prompt"], str(workflow_path)))
        if spec.width_binding and payload.get("width") is not None:
            updates.append(self._binding_update(spec.width_binding, int(payload["width"]), str(workflow_path)))
        if spec.height_binding and payload.get("height") is not None:
            updates.append(self._binding_update(spec.height_binding, int(payload["height"]), str(workflow_path)))
        if spec.steps_binding and payload.get("steps") is not None:
            updates.append(self._binding_update(spec.steps_binding, int(payload["steps"]), str(workflow_path)))
        if spec.denoise_binding and payload.get("denoise") is not None:
            updates.append(self._binding_update(spec.denoise_binding, float(payload["denoise"]), str(workflow_path)))

        image_path = payload.get("image_path") or payload.get("input_image_path")
        image_binding = spec.image_binding
        if image_binding and image_path:
            image_filename = generator.upload_image(str(image_path))
            updates.append(self._binding_update(image_binding, image_filename, str(workflow_path)))

        seed = payload.get("seed")
        if seed is None and spec.seed_enabled:
            seed = random.randint(1, 999999999)

        return self.adapter.generate_updates(
            workflow=workflow or self.adapter.load_workflow(workflow_path),
            updates_config=updates,
            description=None,
            seed=seed if spec.seed_enabled else None,
            workflow_path=str(workflow_path),
        )


    @staticmethod
    def _resolve_model_overrides(spec: ComfyWorkflowSpec, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw_overrides = payload.get("model_overrides")
        overrides: dict[str, dict[str, Any]] = {}
        if isinstance(raw_overrides, dict):
            overrides.update(
                {
                    str(node_id): dict(value)
                    for node_id, value in raw_overrides.items()
                    if isinstance(value, dict)
                }
            )
        return overrides

    @staticmethod
    def _apply_model_overrides(
        workflow: dict[str, Any],
        overrides: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply model loader class/input overrides to a runtime-only graph."""

        runtime = json.loads(json.dumps(workflow))
        for node_id, override in overrides.items():
            node = runtime.get(str(node_id))
            if not isinstance(node, dict):
                raise ValueError(f"Model override references unknown workflow node {node_id!r}")
            if override.get("class_type"):
                node["class_type"] = str(override["class_type"])
            inputs = override.get("inputs")
            if inputs is not None and not isinstance(inputs, dict):
                raise ValueError(f"Model override inputs for node {node_id!r} must be an object")
            if isinstance(inputs, dict):
                if bool(override.get("replace_inputs")):
                    node["inputs"] = dict(inputs)
                else:
                    node.setdefault("inputs", {}).update(inputs)
        return runtime





    @staticmethod
    def _check_required_node_types(generator: Any, spec: ComfyWorkflowSpec) -> None:
        missing: list[str] = []
        for node_type in spec.required_node_types:
            try:
                info = generator.get_object_info(node_type)
            except Exception:
                info = {}
            if not isinstance(info, dict) or not info.get(node_type):
                missing.append(node_type)
        if missing:
            raise RuntimeError(
                "ComfyUI is missing required node type(s): "
                + ", ".join(missing)
                + ". Install the custom nodes required by the selected workflow before rendering."
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
                steps_binding=NodeBinding(kind="steps", node_type="KSampler", input_key="steps"),
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
                steps_binding=NodeBinding(kind="steps", node_type="KSampler", input_key="steps"),
                denoise_binding=NodeBinding(kind="denoise", node_type="KSampler", input_key="denoise"),
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
