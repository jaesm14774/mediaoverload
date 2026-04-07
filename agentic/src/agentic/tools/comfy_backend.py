from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import parse, request

import websocket
import yaml


class AgenticComfyCommunicator:
    def __init__(self, host: str | None = None, port: int | None = None, timeout: int = 900) -> None:
        self.host = host or os.environ.get("COMFYUI_HOST", "host.docker.internal")
        self.port = port or int(os.environ.get("COMFYUI_PORT", "8188"))
        self.client_id = str(uuid.uuid4())
        self.server_address = f"{self.host}:{self.port}"
        self.timeout = timeout
        self.ws: websocket.WebSocket | None = None

    def connect_websocket(self) -> None:
        self.ws = websocket.WebSocket()
        self.ws.connect(
            f"ws://{self.server_address}/ws?clientId={self.client_id}",
            ping_interval=20,
            ping_timeout=10,
        )

    def queue_prompt(self, prompt: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode("utf-8")
        req = request.Request(f"http://{self.server_address}/prompt", data=payload)
        return json.loads(request.urlopen(req, timeout=30).read())

    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = False) -> str:
        import mimetypes

        image_bytes = Path(image_path).read_bytes()
        filename = Path(image_path).name
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        boundary = "----WebKitFormBoundary" + str(uuid.uuid4()).replace("-", "")
        parts: list[bytes] = []

        def add_text(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            parts.append(value.encode())
            parts.append(b"\r\n")

        add_text("overwrite", str(overwrite).lower())
        if subfolder:
            add_text("subfolder", subfolder)
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode())
        parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode())
        parts.append(image_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        req = request.Request(
            f"http://{self.server_address}/upload/image",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            response = request.urlopen(req, timeout=60)
            result = json.loads(response.read().decode("utf-8"))
            return str(result.get("name", filename))
        except Exception:
            return filename

    def get_media_file(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        query = parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with request.urlopen(f"http://{self.server_address}/view?{query}", timeout=60) as response:
            return response.read()

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        with request.urlopen(f"http://{self.server_address}/history/{prompt_id}", timeout=30) as response:
            return json.loads(response.read())

    def wait_for_completion(self, prompt_id: str) -> None:
        import time

        started = time.time()
        while True:
            if time.time() - started > self.timeout:
                raise TimeoutError(f"ComfyUI prompt timed out after {self.timeout} seconds: {prompt_id}")
            if not self.ws or not self.ws.connected:
                raise RuntimeError("ComfyUI websocket disconnected")
            try:
                self.ws.settimeout(5.0)
                raw_message = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if isinstance(raw_message, (bytes, bytearray)):
                # ComfyUI may emit binary preview frames over the websocket before
                # the final JSON execution events. These should not be parsed as JSON.
                continue
            message = json.loads(raw_message)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("prompt_id") == prompt_id and data.get("node") is None:
                    return
            if message.get("type") == "execution_error":
                data = message.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    raise RuntimeError(
                        f"ComfyUI execution failed at node {data.get('node_id')}: {data.get('exception_message')}"
                    )

    def identify_all_nodes(self, workflow: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        connections: dict[str, dict[str, Any]] = {}
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            node_inputs = {}
            for input_name, input_value in node_data.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    node_inputs[input_name] = {"source_node": str(input_value[0]), "output_index": input_value[1]}
            connections[node_id] = {
                "inputs": node_inputs,
                "class_type": node_data.get("class_type"),
            }

        node_types: dict[str, list[dict[str, Any]]] = {}
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get("class_type")
            if not class_type:
                continue
            meta = node_data.get("_meta", {})
            title = str(meta.get("title", ""))
            node_info = {
                "id": node_id,
                "data": node_data,
                "connections": connections.get(node_id, {}),
                "metadata": {
                    "title": title,
                    "title_lower": title.lower(),
                    "is_negative": "negative" in title.lower() if class_type in {"PrimitiveString", "CLIPTextEncode"} else False,
                },
            }
            node_types.setdefault(class_type, []).append(node_info)
        return node_types

    @staticmethod
    def update_node_inputs(workflow: dict[str, Any], node_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        if node_id in workflow and isinstance(workflow[node_id], dict):
            workflow[node_id].setdefault("inputs", {}).update(updates)
        return workflow

    def process_workflow(
        self,
        workflow: dict[str, Any],
        updates: list[dict[str, Any]],
        output_path: str,
        file_name: str | None = None,
        auto_close: bool = False,
    ) -> tuple[bool, list[str]]:
        try:
            if not self.ws or not self.ws.connected:
                self.connect_websocket()
            os.makedirs(output_path, exist_ok=True)
            workflow_copy = json.loads(json.dumps(workflow))
            all_nodes = self.identify_all_nodes(workflow_copy)
            for update in updates:
                if update.get("type") == "direct_update":
                    node_id = str(update["node_id"])
                    workflow_copy = self.update_node_inputs(workflow_copy, node_id, dict(update.get("inputs", {})))
                    continue
                node_type = str(update.get("type", ""))
                node_index = int(update.get("node_index", 0))
                matching_nodes = list(all_nodes.get(node_type, []))
                if "is_negative" in update:
                    matching_nodes = [
                        node for node in matching_nodes if node["metadata"].get("is_negative") == update["is_negative"]
                    ]
                if "title" in update:
                    expected = str(update["title"]).lower()
                    matching_nodes = [
                        node for node in matching_nodes if expected in str(node["metadata"].get("title_lower", ""))
                    ]
                if node_index < len(matching_nodes):
                    workflow_copy = self.update_node_inputs(
                        workflow_copy,
                        str(matching_nodes[node_index]["id"]),
                        dict(update.get("inputs", {})),
                    )
            prompt_id = str(self.queue_prompt(workflow_copy)["prompt_id"])
            self.wait_for_completion(prompt_id)
            return self.save_results(prompt_id, output_path, file_name)
        except Exception as exc:
            return False, [str(exc)]
        finally:
            if auto_close and self.ws and self.ws.connected:
                self.ws.close()

    def save_results(self, prompt_id: str, output_path: str, file_name: str | None) -> tuple[bool, list[str]]:
        try:
            history = self.get_history(prompt_id)[prompt_id]
            saved_files: list[str] = []
            for node_output in history.get("outputs", {}).values():
                if not isinstance(node_output, dict):
                    continue
                for key, default_extension in (("images", ".png"), ("gifs", ""), ("videos", "")):
                    for media in node_output.get(key, []):
                        if media.get("type") == "temp":
                            continue
                        media_bytes = self.get_media_file(media["filename"], media["subfolder"], media["type"])
                        extension = Path(media["filename"]).suffix or default_extension
                        base = Path(media["filename"]).stem
                        final_name = media["filename"] if not file_name else f"{base}_{file_name}{extension}"
                        save_path = str(Path(output_path) / final_name)
                        Path(save_path).write_bytes(media_bytes)
                        saved_files.append(save_path)
            return True, saved_files
        except Exception:
            return False, []


class AgenticMediaGenerator:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.communicator = AgenticComfyCommunicator(host, port)
        self.communicator.connect_websocket()

    def generate(self, workflow_path: str, updates: list[dict[str, Any]], output_dir: str, file_prefix: str = "media") -> list[str]:
        workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        success, saved_files = self.communicator.process_workflow(
            workflow=workflow,
            updates=updates,
            output_path=output_dir,
            file_name=file_prefix,
            auto_close=False,
        )
        if not success:
            error = saved_files[0] if saved_files else "Unknown error"
            raise RuntimeError(f"Media generation failed for {workflow_path}: {error}")
        return saved_files

    def upload_image(self, image_path: str) -> str:
        return self.communicator.upload_image(image_path)


class AgenticNodeManager:
    BUILTIN_STRATEGIES = {
        "text": {
            "priority": [
                {"node_type": "PrimitiveString", "input_key": "value", "filter_key": "is_negative"},
                {"node_type": "CLIPTextEncode", "input_key": "text", "filter_key": "is_negative"},
            ]
        },
        "sampler": {
            "priority": [
                {"node_type": "RandomNoise", "input_key": "noise_seed"},
                {"node_type": "KSamplerAdvanced", "input_key": "noise_seed"},
                {"node_type": "KSampler", "input_key": "seed"},
                {"node_type": "MMAudioSampler", "input_key": "seed"},
            ]
        },
    }

    @staticmethod
    def load_workflow_config() -> dict[str, Any]:
        config_path = Path("configs/workflow/workflow_config.yaml")
        if not config_path.exists():
            return {}
        try:
            content = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            return dict(content.get("workflows", {}))
        except Exception:
            return {}

    @classmethod
    def match_workflow_config(cls, workflow_path: str) -> dict[str, Any] | None:
        normalized = workflow_path.replace("\\", "/")
        configs = cls.load_workflow_config()
        if normalized in configs:
            return configs[normalized]
        for key, value in configs.items():
            if normalized.endswith(key) or Path(normalized).name == key:
                return value
        return None

    @classmethod
    def resolve_alias(cls, workflow_path: str, alias_name: str) -> str | None:
        config = cls.match_workflow_config(workflow_path)
        if not config:
            return None
        aliases = config.get("node_aliases", {})
        return str(aliases.get(alias_name)) if alias_name in aliases else None

    @staticmethod
    def get_node_indices(workflow: dict[str, Any], node_type: str, **filters: Any) -> list[int]:
        communicator = AgenticComfyCommunicator()
        matching = communicator.identify_all_nodes(workflow).get(node_type, [])
        if filters:
            filtered = []
            for node in matching:
                metadata = node.get("metadata", {})
                if all(
                    (str(value).lower() in str(metadata.get("title_lower", "")) if key == "title" else metadata.get(key) == value)
                    for key, value in filters.items()
                ):
                    filtered.append(node)
            matching = filtered
        return list(range(len(matching)))

    @staticmethod
    def create_node_update(node_type: str, node_index: int, inputs: dict[str, Any], **additional_params: Any) -> dict[str, Any]:
        return {"type": node_type, "node_index": node_index, "inputs": inputs, **additional_params}

    @classmethod
    def generate_updates(
        cls,
        workflow: dict[str, Any],
        updates_config: list[dict[str, Any]] | None = None,
        description: str | None = None,
        seed: int | None = None,
        use_noise_seed: bool = False,
        exclude_sampler_indices: list[int] | None = None,
        workflow_path: str | None = None,
        **additional_params: Any,
    ) -> list[dict[str, Any]]:
        del workflow_path
        updates: list[dict[str, Any]] = []
        if updates_config:
            updates.extend(cls._generate_custom_updates(workflow, updates_config))
        has_text_update = any(u.get("node_type") in {"PrimitiveString", "CLIPTextEncode"} for u in (updates_config or []))
        if description is not None and not has_text_update:
            updates.extend(cls._generate_builtin_text_updates(workflow, description, **additional_params))
        if seed is not None:
            updates.extend(
                cls._generate_builtin_sampler_updates(
                    workflow,
                    seed,
                    use_noise_seed=use_noise_seed,
                    exclude_indices=exclude_sampler_indices or [],
                )
            )
        return updates

    @classmethod
    def _generate_custom_updates(cls, workflow: dict[str, Any], updates_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for config in updates_config:
            if "node_id" in config:
                results.append({"type": "direct_update", "node_id": config["node_id"], "inputs": dict(config.get("inputs", {}))})
                continue
            node_type = str(config.get("node_type", ""))
            node_index = int(config.get("node_index", 0))
            inputs = dict(config.get("inputs", {}))
            filters = dict(config.get("filter", {}))
            indices = cls.get_node_indices(workflow, node_type, **filters)
            if not indices and node_type == "PrimitiveString" and "value" in inputs:
                # Fallback: workflow uses CLIPTextEncode directly instead of PrimitiveString
                # (e.g. anima_anime.json). Reuse the same title/is_negative filters so
                # positive and negative nodes are matched correctly.
                fallback_indices = cls.get_node_indices(workflow, "CLIPTextEncode", **filters)
                if fallback_indices:
                    clip_index = min(node_index, len(fallback_indices) - 1)
                    results.append(cls.create_node_update("CLIPTextEncode", fallback_indices[clip_index], {"text": inputs["value"]}, **filters))
                continue
            if indices and node_index < len(indices):
                results.append(cls.create_node_update(node_type, indices[node_index], inputs, **filters))
            elif indices:
                for index in indices:
                    results.append(cls.create_node_update(node_type, index, inputs, **filters))
        return results

    @classmethod
    def _generate_builtin_text_updates(cls, workflow: dict[str, Any], description: str, **additional_params: Any) -> list[dict[str, Any]]:
        filter_value = additional_params.get("is_negative", False)
        for priority in cls.BUILTIN_STRATEGIES["text"]["priority"]:
            node_type = priority["node_type"]
            input_key = priority["input_key"]
            filter_key = priority["filter_key"]
            indices = cls.get_node_indices(workflow, node_type, **{filter_key: filter_value})
            if indices:
                return [
                    cls.create_node_update(node_type, index, {input_key: description}, **{filter_key: filter_value})
                    for index in indices
                ]
        return []

    @classmethod
    def _generate_builtin_sampler_updates(
        cls,
        workflow: dict[str, Any],
        seed: int,
        use_noise_seed: bool = False,
        exclude_indices: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        exclude = set(exclude_indices or [])
        priorities = list(cls.BUILTIN_STRATEGIES["sampler"]["priority"])
        if use_noise_seed:
            priorities.sort(key=lambda item: 0 if item["input_key"] == "noise_seed" else 1)
        updates: list[dict[str, Any]] = []
        for priority in priorities:
            node_type = priority["node_type"]
            input_key = priority["input_key"]
            if use_noise_seed and input_key == "seed":
                continue
            indices = [index for index in cls.get_node_indices(workflow, node_type) if index not in exclude]
            if indices:
                updates.extend(
                    cls.create_node_update(node_type, index, {input_key: seed + index}) for index in indices
                )
                if use_noise_seed and input_key == "noise_seed":
                    break
        return updates
