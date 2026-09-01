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
from agentic.assets.minimax_h3 import minimax_h3_model_overrides
from agentic.h3_reference import build_reference_lineage, normalize_reference_manifest
from agentic.runtime.registry import ToolRegistry
from agentic.runtime.h3_modes import validate_h3_payload
from agentic.tools.comfy_adapter import ComfyAdapter


MAX_REFERENCE_IMAGE_SLOTS = 9


def _payload_requires_declared_subject_pair(payload: dict[str, Any]) -> bool:
    context = payload.get("subject_context")
    if not isinstance(context, dict):
        return False
    contract = context.get("interaction_contract")
    return bool(isinstance(contract, dict) and contract.get("required", False))


MAX_REFERENCE_VIDEO_SLOTS = 3


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
    first_frame_binding: NodeBinding | None = None
    last_frame_binding: NodeBinding | None = None
    reference_image_bindings: tuple[NodeBinding, ...] = ()
    reference_video_bindings: tuple[NodeBinding, ...] = ()
    reference_conditioning_node_type: str | None = None
    reference_image_size_binding: NodeBinding | None = None
    seed_enabled: bool = True
    default_payload: dict[str, Any] = field(default_factory=dict)


class ComfyWorkflowToolset:
    DEFAULT_IMAGE_WORKFLOWS = ("krea2_turbo",)
    DEFAULT_REFINE_WORKFLOWS = ("krea2_turbo_img2img",)
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)
    DEFAULT_I2V_WORKFLOWS = ("minimax_h3_lowvram_i2v",)

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
        if "comfy.workflow.reference_to_video" in self.specs:
            tool_registry.register(
                "comfy.render_reference_to_video",
                self._build_handler("comfy.workflow.reference_to_video"),
                "Render a MiniMax H3 reference-image/video-to-video workflow through ComfyUI",
            )
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
        h3_mode = merged_payload.get("h3_mode") or merged_payload.get("generation_type")
        if h3_mode and requested_workflow_name.startswith("minimax_h3_"):
            generic_to_h3 = {
                "anchor_first": "i2va",
                "anchor_first_last": "fl2va",
                "anchor_last": "l2va",
                "reference_bundle": "ref2va",
            }
            validate_h3_payload(generic_to_h3.get(str(h3_mode), str(h3_mode)), merged_payload)
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
        reference_info: dict[str, Any] | None = None
        runtime_workflow = workflow
        model_overrides = self._resolve_model_overrides(spec, merged_payload)
        if model_overrides:
            runtime_workflow = self._apply_model_overrides(runtime_workflow, model_overrides)
            merged_payload["model_overrides"] = model_overrides
        if spec.reference_conditioning_node_type:
            reference_info = self._prepare_reference_payload(merged_payload)
            merged_payload["reference_manifest"] = reference_info["manifest"]
            merged_payload["reference_lineage"] = reference_info["lineage"]
            merged_payload["reference_mode"] = self._reference_mode(reference_info["manifest"])
            self._check_required_nodes(generator, spec, reference_info["manifest"])
            runtime_workflow = self._build_runtime_reference_workflow(
                runtime_workflow,
                reference_info["manifest"],
                merged_payload,
            )
        output_dir = run_dir / spec.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        render_count = max(1, int(merged_payload.get(spec.count_payload_key, 1)))
        saved_files: list[str] = []
        memory_retry_count = 0
        try:
            for run_index in range(render_count):
                iteration_payload = dict(merged_payload)
                if spec.seed_enabled and "seed" not in iteration_payload:
                    iteration_payload["seed"] = random.randint(1, 999999999)
                updates = self._build_updates(
                    spec,
                    workflow_path,
                    iteration_payload,
                    generator,
                    workflow=runtime_workflow if (reference_info or model_overrides) else None,
                )
                run_suffix = spec.file_prefix if render_count == 1 else f"{spec.file_prefix}_{run_index + 1:02d}"
                generate_kwargs = {
                    "workflow_path": str(workflow_path),
                    "updates": updates,
                    "output_dir": str(output_dir),
                    "file_prefix": run_suffix,
                }
                if reference_info or model_overrides:
                    generate_kwargs["workflow"] = runtime_workflow
                try:
                    saved_files.extend(generator.generate(**generate_kwargs))
                except Exception as exc:
                    # Sequential long-video segments can leave a previous
                    # provider graph resident in ComfyUI.  Retry the exact
                    # same recipe once after the server-side unload boundary;
                    # never switch to another workflow as an implicit fallback.
                    if not self._is_memory_error(exc):
                        raise
                    self._release_comfy_memory(generator)
                    memory_retry_count += 1
                    saved_files.extend(generator.generate(**generate_kwargs))
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
            "h3_mode": str(h3_mode or ""),
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

        if spec.reference_conditioning_node_type:
            updates.extend(self._build_reference_updates(spec, payload, workflow_path, generator))

        image_path = payload.get("image_path") or payload.get("input_image_path")
        requested_workflow = str(payload.get("workflow_name") or spec.workflow_name)
        generic_to_h3 = {
            "anchor_first": "i2va",
            "anchor_first_last": "fl2va",
            "anchor_last": "l2va",
            "reference_bundle": "ref2va",
        }
        normalized_h3_mode = generic_to_h3.get(str(payload.get("h3_mode") or ""), str(payload.get("h3_mode") or "")).strip().lower()
        if image_path and spec.name == "comfy.workflow.image_to_video" and requested_workflow.startswith("minimax_h3_"):
            prompt_text = str(payload.get("prompt") or "").lower()
            if str(payload.get("character") or "").strip().lower() == "kirby" or "kirby" in prompt_text:
                assert_kirby_input(
                    image_path,
                    allow_external=bool(payload.get("allow_external_reference", False)),
                    allow_multipanel=requested_workflow == "minimax_h3_ref2va" or normalized_h3_mode in {"ref2va", "reference_to_video", "native_h3_ref2va"},
                    allow_declared_subject_pair=_payload_requires_declared_subject_pair(payload),
                )
        image_binding = spec.image_binding
        if image_path and requested_workflow.endswith("_15s_fl2va_i2v") and image_binding:
            # The FLF graph has two LoadImage nodes.  Bind the opening anchor by
            # its semantic title instead of relying on node insertion order.
            image_binding = NodeBinding(
                kind=image_binding.kind,
                node_type=image_binding.node_type,
                node_index=image_binding.node_index,
                title="native 15s first frame",
                alias=image_binding.alias,
                input_key=image_binding.input_key,
            )
        if image_binding and image_path:
            image_filename = generator.upload_image(str(image_path))
            updates.append(self._binding_update(image_binding, image_filename, str(workflow_path)))

        last_image_path = payload.get("last_image_path") or payload.get("last_frame_path")
        if last_image_path and spec.last_image_binding and requested_workflow.startswith("minimax_h3_"):
            prompt_text = str(payload.get("prompt") or "").lower()
            if str(payload.get("character") or "").strip().lower() == "kirby" or "kirby" in prompt_text:
                assert_kirby_input(
                    last_image_path,
                    allow_external=bool(payload.get("allow_external_reference", False)),
                    allow_multipanel=requested_workflow == "minimax_h3_ref2va" or normalized_h3_mode in {"ref2va", "reference_to_video", "native_h3_ref2va"},
                    allow_declared_subject_pair=_payload_requires_declared_subject_pair(payload),
                )
            last_image_filename = generator.upload_image(str(last_image_path))
            updates.append(self._binding_update(spec.last_image_binding, last_image_filename, str(workflow_path)))
        elif payload.get("use_last_frame") is False and spec.last_frame_binding and requested_workflow.startswith("minimax_h3_"):
            # The 15s manifest contains a last_frame connection by design. Override
            # it with None for first-frame-only runs so ComfyUI cannot reuse the
            # template's placeholder LoadImage node.
            updates.append(self._binding_update(spec.last_frame_binding, None, str(workflow_path)))

        if payload.get("use_first_frame") is False and spec.first_frame_binding and requested_workflow.startswith("minimax_h3_"):
            updates.append(self._binding_update(spec.first_frame_binding, None, str(workflow_path)))

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
    def _reference_mode(references: list[dict[str, Any]]) -> str:
        image_count = sum(str(reference.get("type")) == "image" for reference in references)
        video_count = sum(str(reference.get("type")) == "video" for reference in references)
        parts: list[str] = []
        if image_count:
            parts.append(f"{image_count} image")
        if video_count:
            parts.append(f"{video_count} video")
        return " + ".join(parts)

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
        profile = payload.get("model_profile")
        workflow_name = str(payload.get("workflow_name") or spec.workflow_name)
        if profile and workflow_name.startswith("minimax_h3_"):
            profile_overrides = minimax_h3_model_overrides(
                str(profile),
                reference_to_video=bool(spec.reference_conditioning_node_type),
            )
            for node_id, value in profile_overrides.items():
                merged = dict(overrides.get(node_id, {}))
                merged_inputs = dict(merged.get("inputs") or {})
                merged_inputs.update(dict(value.get("inputs") or {}))
                merged.update(value)
                if merged_inputs:
                    merged["inputs"] = merged_inputs
                overrides[node_id] = merged
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
    def _build_runtime_reference_workflow(
        workflow: dict[str, Any],
        references: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build only the reference loaders that this render actually uses.

        The checked-in workflow is intentionally a zero-reference base graph.
        ComfyUI's H3 node exposes optional auto-grow inputs, so materialising
        unused image/video loaders is unnecessary work and makes the graph
        falsely imply that both reference types are required.
        """

        runtime = json.loads(json.dumps(workflow))
        conditioning_nodes: list[dict[str, Any]] = []
        stale_reference_nodes: set[str] = set()
        reference_prefixes = ("ref_images.", "ref_videos.", "ref_audios.", "ref_video_audios.")

        for node_id, node in list(runtime.items()):
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            title = str(node.get("_meta", {}).get("title") or "").lower()
            if class_type == "MiniMaxH3ReferenceToVideo":
                conditioning_nodes.append(node)
                inputs = node.setdefault("inputs", {})
                for input_name in list(inputs):
                    if input_name.startswith(reference_prefixes):
                        source = inputs.pop(input_name)
                        if isinstance(source, list) and source:
                            stale_reference_nodes.add(str(source[0]))
            if class_type in {"LoadImage", "VHS_LoadVideoPath"} and "h3 reference " in title:
                stale_reference_nodes.add(str(node_id))

        for node_id in stale_reference_nodes:
            runtime.pop(node_id, None)

        next_node_id = max((int(node_id) for node_id in runtime if str(node_id).isdigit()), default=0) + 1

        def allocate_node() -> str:
            nonlocal next_node_id
            while str(next_node_id) in runtime:
                next_node_id += 1
            node_id = str(next_node_id)
            next_node_id += 1
            return node_id

        image_index = 0
        video_index = 0
        image_size = int(payload.get("width") or 608), int(payload.get("height") or 352)
        for reference in references:
            ref_type = str(reference["type"])
            if ref_type == "image":
                image_index += 1
                node_id = allocate_node()
                runtime[node_id] = {
                    "inputs": {"image": "__runtime_reference_image__"},
                    "class_type": "LoadImage",
                    "_meta": {"title": f"H3 reference image {image_index}"},
                }
                input_name = f"ref_images.ref_image_{image_index - 1}"
            else:
                video_index += 1
                node_id = allocate_node()
                runtime[node_id] = {
                    "inputs": {
                        "video": str(reference["path"]),
                        "force_rate": 24.0,
                        "custom_width": image_size[0],
                        "custom_height": image_size[1],
                        "frame_load_cap": int(payload.get("reference_frame_cap") or 0),
                        "skip_first_frames": 0,
                        "select_every_nth": 1,
                    },
                    "class_type": "VHS_LoadVideoPath",
                    "_meta": {"title": f"H3 reference video {video_index}"},
                }
                input_name = f"ref_videos.ref_video_{video_index - 1}"

            for conditioning_node in conditioning_nodes:
                conditioning_node.setdefault("inputs", {})[input_name] = [node_id, 0]

        return runtime

    def _prepare_reference_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        references = normalize_reference_manifest(
            payload.get("reference_manifest"),
            image_paths=payload.get("reference_image_paths"),
            video_paths=payload.get("reference_video_paths"),
            require_files=True,
            max_images=MAX_REFERENCE_IMAGE_SLOTS,
            max_videos=MAX_REFERENCE_VIDEO_SLOTS,
        )
        return {"manifest": references, "lineage": build_reference_lineage(references)}

    def _build_reference_updates(
        self,
        spec: ComfyWorkflowSpec,
        payload: dict[str, Any],
        workflow_path: Path,
        generator: Any,
    ) -> list[dict[str, Any]]:
        references = normalize_reference_manifest(
            payload.get("reference_manifest"),
            require_files=True,
            max_images=len(spec.reference_image_bindings),
            max_videos=len(spec.reference_video_bindings),
        )
        updates: list[dict[str, Any]] = []
        image_index = 0
        video_index = 0
        for reference in references:
            ref_type = str(reference["type"])
            if ref_type == "image":
                binding = spec.reference_image_bindings[image_index]
                uploaded_name = generator.upload_image(str(reference["path"]))
                updates.append(self._binding_update(binding, uploaded_name, str(workflow_path)))
                image_index += 1
            elif ref_type == "video":
                binding = spec.reference_video_bindings[video_index]
                # VHS_LoadVideoPath reads from the same filesystem as ComfyUI;
                # uploading a video through the image endpoint would corrupt the
                # input contract and silently lose frame-rate metadata.
                updates.append(self._binding_update(binding, str(reference["path"]), str(workflow_path)))
                video_index += 1
            else:  # normalize_reference_manifest already rejects this; keep the guard explicit.
                raise ValueError(f"Unsupported H3 reference type: {ref_type}")

        if spec.reference_image_size_binding and payload.get("ref_image_size") is not None:
            updates.append(
                self._binding_update(
                    spec.reference_image_size_binding,
                    str(payload["ref_image_size"]),
                    str(workflow_path),
                )
            )
        return updates

    @staticmethod
    def _check_required_nodes(generator: Any, spec: ComfyWorkflowSpec, references: list[dict[str, Any]]) -> None:
        required = {str(spec.reference_conditioning_node_type)}
        if any(str(reference.get("type")) == "video" for reference in references):
            required.add("VHS_LoadVideoPath")
        missing: list[str] = []
        for node_type in sorted(required):
            try:
                info = generator.get_object_info(node_type)
            except Exception:
                info = {}
            if not isinstance(info, dict) or not info.get(node_type):
                missing.append(node_type)
        if missing:
            raise RuntimeError(
                "ComfyUI is missing required Ref2VA node(s): "
                + ", ".join(missing)
                + ". Update ComfyUI and install ComfyUI-VideoHelperSuite before running this mode."
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
        reference_image_bindings = tuple(
            NodeBinding(kind="reference_image", node_type="LoadImage", title=f"H3 reference image {index}", input_key="image")
            for index in range(1, 10)
        )
        reference_video_bindings = tuple(
            NodeBinding(kind="reference_video", node_type="VHS_LoadVideoPath", title=f"H3 reference video {index}", input_key="video")
            for index in range(1, 4)
        )
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
                first_frame_binding=NodeBinding(kind="first_frame", node_type="MiniMaxH3ImageToVideo", input_key="first_frame") if i2v_workflow.startswith("minimax_h3_") else None,
                last_frame_binding=NodeBinding(kind="last_frame", node_type="MiniMaxH3ImageToVideo", input_key="last_frame") if i2v_workflow.startswith("minimax_h3_") else None,
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
            "comfy.workflow.reference_to_video": ComfyWorkflowSpec(
                name="comfy.workflow.reference_to_video",
                workflow_name="minimax_h3_ref2va",
                output_folder="videos",
                file_prefix="agentic_h3_ref2va",
                count_payload_key="video_count",
                prompt_binding=NodeBinding(kind="prompt", node_type="MiniMaxH3ReferenceToVideo", input_key="prompt"),
                width_binding=NodeBinding(kind="width", node_type="MiniMaxH3ReferenceToVideo", input_key="width"),
                height_binding=NodeBinding(kind="height", node_type="MiniMaxH3ReferenceToVideo", input_key="height"),
                length_binding=NodeBinding(kind="length", node_type="MiniMaxH3ReferenceToVideo", input_key="length"),
                steps_binding=NodeBinding(kind="steps", node_type="BasicScheduler", input_key="steps"),
                reference_image_bindings=reference_image_bindings,
                reference_video_bindings=reference_video_bindings,
                reference_conditioning_node_type="MiniMaxH3ReferenceToVideo",
                reference_image_size_binding=NodeBinding(
                    kind="reference_image_size",
                    node_type="MiniMaxH3ReferenceToVideo",
                    input_key="ref_image_size",
                ),
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
