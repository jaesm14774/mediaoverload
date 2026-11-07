from __future__ import annotations

import http.client
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable
from pathlib import Path, PureWindowsPath
from typing import Any, NamedTuple, TypeAlias
from urllib import error, parse, request

import websocket
import yaml

from agentic.tools.comfy_diagnostics import describe_rejection, validate_workflow_graph


class ComfyUploadResult(NamedTuple):
    comfy_path: str
    sha256: str | None


ComfyNodeExecutionInterval: TypeAlias = dict[str, str | float]
ComfyPromptTiming: TypeAlias = dict[
    str,
    float | bool | int | list[str] | list[ComfyNodeExecutionInterval],
]


def get_object_info_for_nodes(
    generator: Any,
    node_types: Iterable[str],
    *,
    tolerate_errors: bool = False,
) -> dict[str, Any]:
    """Fetch targeted node metadata concurrently; optionally treat request errors as missing nodes."""
    ordered_node_types = tuple(sorted(set(node_types)))
    if not ordered_node_types:
        return {}

    def fetch_node_info(node_type: str) -> Any:
        try:
            info = generator.get_object_info(node_type)
        except Exception:
            if not tolerate_errors:
                raise
            return {}
        if tolerate_errors and not isinstance(info, dict):
            return {}
        return info

    with ThreadPoolExecutor(max_workers=min(8, len(ordered_node_types))) as executor:
        node_info: dict[str, Any] = {}
        for info in executor.map(fetch_node_info, ordered_node_types):
            node_info.update(info)
    return node_info


class AgenticComfyCommunicator:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.host = host or os.environ.get("COMFYUI_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("COMFYUI_PORT", "8188"))
        self.client_id = str(uuid.uuid4())
        self.server_address = f"{self.host}:{self.port}"
        self._clock = clock or time.perf_counter
        configured_timeout = os.environ.get("COMFYUI_TIMEOUT_SECONDS", "3600")
        self.timeout = int(timeout if timeout is not None else configured_timeout)
        # A busy ComfyUI (large model swaps, RTX video nodes) can accept a prompt
        # long before its HTTP handler answers; a short limit orphans the job.
        self.submit_timeout = 300.0
        self.ws: websocket.WebSocket | None = None

    def connect_websocket(self) -> None:
        self.ws = websocket.WebSocket()
        try:
            connect_timeout = max(0.1, float(os.environ.get("COMFYUI_CONNECT_TIMEOUT_SECONDS", "30")))
        except ValueError:
            connect_timeout = 30.0
        self.ws.connect(
            f"ws://{self.server_address}/ws?clientId={self.client_id}",
            timeout=connect_timeout,
            ping_interval=20,
            ping_timeout=10,
        )

    def queue_prompt(self, prompt: dict[str, Any], prompt_id: str) -> dict[str, Any]:
        payload = json.dumps(
            {"prompt": prompt, "client_id": self.client_id, "prompt_id": prompt_id}
        ).encode("utf-8")
        req = request.Request(f"http://{self.server_address}/prompt", data=payload)
        try:
            with request.urlopen(req, timeout=self.submit_timeout) as response:
                return json.loads(response.read())
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise RuntimeError(describe_rejection(exc.code, body)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"ComfyUI did not acknowledge /prompt within {self.submit_timeout:g} seconds"
            ) from exc

    def upload_image(
        self,
        image_path: str,
        subfolder: str = "",
        overwrite: bool = False,
        *,
        include_sha256: bool = False,
    ) -> ComfyUploadResult:
        """Upload conditioning media to ComfyUI without buffering the whole file."""
        import mimetypes

        path = Path(image_path)
        filename = path.name
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
        file_header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode()
        prefix = b"".join(parts) + file_header
        suffix = f"\r\n--{boundary}--\r\n".encode()
        content_length = len(prefix) + path.stat().st_size + len(suffix)
        connection = http.client.HTTPConnection(self.host, self.port, timeout=60)
        digest = hashlib.sha256() if include_sha256 else None
        try:
            connection.putrequest("POST", "/upload/image")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            connection.send(prefix)
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    if digest is not None:
                        digest.update(chunk)
                    connection.send(chunk)
            connection.send(suffix)
            response = connection.getresponse()
            response_body = response.read()
            if response.status < 200 or response.status >= 300:
                detail = response_body.decode("utf-8", "replace").strip()[:4000] or "no response body"
                raise RuntimeError(f"ComfyUI /upload/image HTTP {response.status}: {detail}")
            result = json.loads(response_body.decode("utf-8"))
        finally:
            connection.close()
        name = str(result["name"])
        comfy_path = f"{result['subfolder']}/{name}" if result.get("subfolder") else name
        return ComfyUploadResult(
            comfy_path,
            digest.hexdigest() if digest is not None else None,
        )

    def download_media_file(self, filename: str, subfolder: str, folder_type: str, destination: str) -> None:
        """Stream ComfyUI output to disk and publish it only after a complete download."""
        query = parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        target = Path(destination)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        with request.urlopen(f"http://{self.server_address}/view?{query}", timeout=60) as response:
            try:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    def get_history(self, prompt_id: str, *, timeout: float = 30) -> dict[str, Any]:
        with request.urlopen(f"http://{self.server_address}/history/{prompt_id}", timeout=timeout) as response:
            return json.loads(response.read())

    def get_object_info(self, node_type: str | None = None) -> dict[str, Any]:
        url = f"http://{self.server_address}/object_info"
        if node_type:
            url += f"/{parse.quote(node_type, safe='')}"
        with request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())

    def free_memory(self) -> None:
        """Release cached models explicitly during image OOM recovery.

        Successful jobs retain their cache; ComfyUI manages model residency
        when the next workflow needs memory. Cleanup failure must not hide
        the original generation error.
        """
        payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = request.Request(
            f"http://{self.server_address}/free",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                response.read()
        except Exception:
            return

    def cancel_prompt(self, prompt_id: str) -> None:
        """Cancel only this communicator's queued/running Comfy prompt.

        A client-side timeout does not stop ComfyUI by itself.  Leaving the
        graph running can block every later strategy on the shared local GPU.
        Inspect the queue first: delete a pending prompt by id, or interrupt
        only when the exact id is currently running.  An unrelated user's
        prompt is never interrupted.
        """
        target = str(prompt_id or "").strip()
        if not target:
            return
        try:
            with request.urlopen(f"http://{self.server_address}/queue", timeout=10) as response:
                queue = json.loads(response.read())
            running_ids = {
                str(item[1])
                for item in queue.get("queue_running", [])
                if isinstance(item, (list, tuple)) and len(item) > 1
            }
            pending_ids = {
                str(item[1])
                for item in queue.get("queue_pending", [])
                if isinstance(item, (list, tuple)) and len(item) > 1
            }
            if target in pending_ids:
                payload = json.dumps({"delete": [target]}).encode("utf-8")
                req = request.Request(
                    f"http://{self.server_address}/queue",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="DELETE",
                )
                with request.urlopen(req, timeout=10) as response:
                    response.read()
            elif target in running_ids:
                # ComfyUI's interrupt endpoint targets the current running
                # prompt. Queue inspection above proves that it is ours.
                req = request.Request(
                    f"http://{self.server_address}/interrupt",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=10) as response:
                    response.read()
        except Exception:
            # Cleanup must never hide the original generation failure.
            return

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timing_sink: ComfyPromptTiming | None = None,
        workflow_node_types: dict[str, str] | None = None,
    ) -> None:
        deadline = time.monotonic() + max(0.1, float(self.timeout))
        wait_started = self._clock()
        execution_started: float | None = None
        active_node: tuple[str, float] | None = None
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"ComfyUI prompt timed out after {self.timeout} seconds: {prompt_id}")
                history_completed, raw_message = self._receive_prompt_message(
                    prompt_id,
                    remaining,
                )
                if history_completed:
                    self._finish_execution_timing(timing_sink, execution_started)
                    return
                if raw_message is None or isinstance(raw_message, (bytes, bytearray)):
                    continue
                message = json.loads(raw_message)
                execution_started, active_node, completed = self._process_prompt_message(
                    message,
                    prompt_id=prompt_id,
                    wait_started=wait_started,
                    execution_started=execution_started,
                    active_node=active_node,
                    workflow_node_types=workflow_node_types or {},
                    timing_sink=timing_sink,
                )
                if completed:
                    self._finish_execution_timing(timing_sink, execution_started)
                    return
        except BaseException:
            self._finish_execution_timing(timing_sink, execution_started)
            raise

    def _finish_execution_timing(
        self,
        timing_sink: ComfyPromptTiming | None,
        execution_started: float | None,
    ) -> None:
        if timing_sink is not None and execution_started is not None:
            timing_sink["server_execution_seconds"] = max(
                0.0, self._clock() - execution_started
            )

    def _history_reports_prompt_completion(
        self,
        prompt_id: str,
        remaining: float,
    ) -> bool:
        history_response = self.get_history(
            prompt_id,
            timeout=min(5.0, max(0.1, remaining)),
        )
        history = history_response.get(prompt_id, history_response)
        status = history.get("status", {}) if isinstance(history, dict) else {}
        if not isinstance(status, dict) or not status.get("completed"):
            return False
        status_str = str(status.get("status_str") or "").lower()
        if status_str == "success":
            return True
        raise RuntimeError(
            f"ComfyUI execution failed for prompt {prompt_id}: {status_str or 'unknown status'}"
        )

    def _receive_prompt_message(
        self,
        prompt_id: str,
        remaining: float,
    ) -> tuple[bool, str | bytes | None]:
        if not self.ws or not self.ws.connected:
            raise RuntimeError("ComfyUI websocket disconnected")
        try:
            # Some ComfyUI/proxy combinations leave an existing raw socket's
            # timeout unchanged, so constrain both websocket and socket reads.
            read_timeout = min(5.0, max(0.1, remaining))
            self.ws.settimeout(read_timeout)
            raw_socket = getattr(self.ws, "sock", None)
            if raw_socket is not None and hasattr(raw_socket, "settimeout"):
                raw_socket.settimeout(read_timeout)
            return False, self.ws.recv()
        except (websocket.WebSocketTimeoutException, socket.timeout, TimeoutError):
            # The terminal websocket event can be lost after ComfyUI records the
            # prompt as complete. History is authoritative in that case.
            try:
                return self._history_reports_prompt_completion(prompt_id, remaining), None
            except RuntimeError:
                raise
            except Exception:
                # An active prompt may not have a history record yet.
                return False, None

    def _process_prompt_message(
        self,
        message: dict[str, Any],
        *,
        prompt_id: str,
        wait_started: float,
        execution_started: float | None,
        active_node: tuple[str, float] | None,
        workflow_node_types: dict[str, str],
        timing_sink: ComfyPromptTiming | None,
    ) -> tuple[float | None, tuple[str, float] | None, bool]:
        message_type = message.get("type")
        data = message.get("data", {})
        if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
            return execution_started, active_node, False
        if message_type == "execution_cached":
            nodes = data.get("nodes")
            if timing_sink is not None and isinstance(nodes, list):
                timing_sink["cached_node_ids"] = [str(node_id) for node_id in nodes]
                timing_sink["cached_node_count"] = len(nodes)
                timing_sink["cached_nodes_observed"] = True
        elif message_type == "execution_start" and execution_started is None:
            execution_started = self._clock()
            if timing_sink is not None:
                timing_sink["queue_wait_seconds"] = max(
                    0.0, execution_started - wait_started
                )
                timing_sink["execution_start_observed"] = True
        elif message_type == "executing":
            event_at = self._clock()
            self._record_node_execution_interval(
                timing_sink,
                active_node,
                event_at,
                workflow_node_types,
            )
            node_id = data.get("node")
            if node_id is None:
                if timing_sink is not None:
                    timing_sink["node_execution_intervals_complete"] = True
                return execution_started, None, True
            return execution_started, (str(node_id), event_at), False
        elif message_type == "execution_error":
            self._record_node_execution_interval(
                timing_sink,
                active_node,
                self._clock(),
                workflow_node_types,
            )
            raise RuntimeError(
                f"ComfyUI execution failed at node {data.get('node_id')}: {data.get('exception_message')}"
            )
        return execution_started, active_node, False

    @staticmethod
    def _record_node_execution_interval(
        timing_sink: ComfyPromptTiming | None,
        active_node: tuple[str, float] | None,
        completed_at: float,
        workflow_node_types: dict[str, str],
    ) -> None:
        if timing_sink is None or active_node is None:
            return
        intervals = timing_sink.get("node_execution_intervals")
        if not isinstance(intervals, list):
            intervals = []
            timing_sink["node_execution_intervals"] = intervals
        node_id, started_at = active_node
        intervals.append(
            {
                "node_id": node_id,
                "node_type": workflow_node_types.get(node_id, "unknown"),
                "seconds": max(0.0, completed_at - started_at),
            }
        )

    @staticmethod
    def identify_all_nodes(workflow: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        connections: dict[str, dict[str, Any]] = {}
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            node_inputs = {}
            for input_name, input_value in node_data.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    node_inputs[input_name] = {
                        "source_node": str(input_value[0]),
                        "output_index": input_value[1],
                    }
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
            title_lower = title.lower()
            node_info = {
                "id": node_id,
                "data": node_data,
                "connections": connections.get(node_id, {}),
                "metadata": {
                    "title": title,
                    "title_lower": title_lower,
                    "is_negative": (
                        "negative" in title_lower
                        if class_type in {"PrimitiveString", "CLIPTextEncode"}
                        else False
                    ),
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
        timing_sink: list[ComfyPromptTiming] | None = None,
    ) -> tuple[bool, list[str]]:
        # The id is chosen client-side so a lost /prompt response can still be cancelled.
        prompt_id = str(uuid.uuid4())
        total_started = self._clock()
        phase_timings: ComfyPromptTiming = {
            "websocket_connect_seconds": 0.0,
            "workflow_prepare_seconds": 0.0,
            "queue_submit_seconds": 0.0,
            "execution_wait_seconds": 0.0,
            "queue_wait_seconds": 0.0,
            "server_execution_seconds": 0.0,
            "execution_start_observed": False,
            "cached_node_ids": [],
            "cached_node_count": 0,
            "cached_nodes_observed": False,
            "node_execution_intervals": [],
            "node_execution_intervals_complete": False,
            "result_retrieval_seconds": 0.0,
            "total_seconds": 0.0,
        }
        try:
            phase_started = self._clock()
            try:
                if not self.ws or not self.ws.connected:
                    self.connect_websocket()
            finally:
                phase_timings["websocket_connect_seconds"] = (
                    self._clock() - phase_started
                )

            phase_started = self._clock()
            try:
                os.makedirs(output_path, exist_ok=True)
                workflow_copy = self._prepare_workflow_copy(workflow, updates)
                workflow_node_types = {
                    str(node_id): str(node.get("class_type") or "unknown")
                    for node_id, node in workflow_copy.items()
                    if isinstance(node, dict)
                }
            finally:
                phase_timings["workflow_prepare_seconds"] = (
                    self._clock() - phase_started
                )

            phase_started = self._clock()
            try:
                prompt_id = str(self.queue_prompt(workflow_copy, prompt_id)["prompt_id"])
            finally:
                phase_timings["queue_submit_seconds"] = (
                    self._clock() - phase_started
                )

            phase_started = self._clock()
            try:
                self.wait_for_completion(
                    prompt_id,
                    timing_sink=phase_timings,
                    workflow_node_types=workflow_node_types,
                )
            finally:
                phase_timings["execution_wait_seconds"] = (
                    self._clock() - phase_started
                )

            phase_started = self._clock()
            try:
                return self.save_results(prompt_id, output_path, file_name)
            finally:
                phase_timings["result_retrieval_seconds"] = (
                    self._clock() - phase_started
                )
        except Exception as exc:
            if prompt_id:
                self.cancel_prompt(prompt_id)
            return False, [str(exc)]
        finally:
            try:
                if auto_close and self.ws and self.ws.connected:
                    self.ws.close()
            finally:
                phase_timings["total_seconds"] = (
                    self._clock() - total_started
                )
                if timing_sink is not None:
                    timing_sink.append(phase_timings)

    def _prepare_workflow_copy(
        self,
        workflow: dict[str, Any],
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        workflow_copy = json.loads(json.dumps(workflow))
        all_nodes = self.identify_all_nodes(workflow_copy)
        for update in updates:
            workflow_copy = self._apply_workflow_update(
                workflow_copy,
                all_nodes,
                update,
            )
        broken_links = validate_workflow_graph(workflow_copy)
        if broken_links:
            raise ValueError("Workflow has broken links: " + "; ".join(broken_links))
        return workflow_copy

    def _apply_workflow_update(
        self,
        workflow: dict[str, Any],
        all_nodes: dict[str, list[dict[str, Any]]],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        if update.get("type") == "direct_update":
            return self.update_node_inputs(
                workflow,
                str(update["node_id"]),
                dict(update.get("inputs", {})),
            )
        matching_nodes = all_nodes.get(str(update.get("type", "")), [])
        if "is_negative" in update:
            matching_nodes = [
                node
                for node in matching_nodes
                if node["metadata"].get("is_negative") == update["is_negative"]
            ]
        if "title" in update:
            expected = str(update["title"]).lower()
            matching_nodes = [
                node
                for node in matching_nodes
                if expected in str(node["metadata"].get("title_lower", ""))
            ]
        node_index = int(update.get("node_index", 0))
        if node_index < len(matching_nodes):
            return self.update_node_inputs(
                workflow,
                str(matching_nodes[node_index]["id"]),
                dict(update.get("inputs", {})),
            )
        return workflow

    def save_results(self, prompt_id: str, output_path: str, file_name: str | None) -> tuple[bool, list[str]]:
        try:
            history = self.get_history(prompt_id)[prompt_id]
            saved_media, preview_media = self._partition_result_media(history)
            saved_files = self._save_result_media(
                saved_media,
                output_path,
                file_name,
            )
            if not saved_files and preview_media:
                # Keep the preview fallback so downstream image/refine nodes
                # receive a stable local file when ComfyUI has no saved output.
                saved_files = self._save_result_media(
                    preview_media,
                    output_path,
                    file_name,
                    default_media_type="temp",
                )
            if not saved_files:
                raise RuntimeError(f"ComfyUI completed prompt {prompt_id} but returned no media files")
            return True, saved_files
        except Exception as exc:
            return False, [str(exc)]

    @staticmethod
    def _partition_result_media(
        history: dict[str, Any],
    ) -> tuple[list[tuple[dict[str, Any], str]], list[tuple[dict[str, Any], str]]]:
        saved_media: list[tuple[dict[str, Any], str]] = []
        preview_media: list[tuple[dict[str, Any], str]] = []
        for node_output in history.get("outputs", {}).values():
            if not isinstance(node_output, dict):
                continue
            for key, default_extension in (("images", ".png"), ("gifs", ""), ("videos", "")):
                for media in node_output.get(key, []):
                    if not isinstance(media, dict) or not media.get("filename"):
                        continue
                    target = preview_media if media.get("type") == "temp" else saved_media
                    target.append((media, default_extension))
        return saved_media, preview_media

    def _save_result_media(
        self,
        media_entries: Iterable[tuple[dict[str, Any], str]],
        output_path: str,
        file_name: str | None,
        *,
        default_media_type: str = "output",
    ) -> list[str]:
        saved_files: list[str] = []
        for media, default_extension in media_entries:
            filename = str(media["filename"])
            extension = Path(filename).suffix or default_extension
            base = Path(filename).stem
            final_name = filename if not file_name else f"{base}_{file_name}{extension}"
            save_path = self._safe_output_path(output_path, final_name)
            self.download_media_file(
                filename,
                str(media.get("subfolder", "")),
                str(media.get("type", default_media_type)),
                save_path,
            )
            saved_files.append(save_path)
        return saved_files

    @staticmethod
    def _safe_output_path(output_path: str, file_name: str) -> str:
        root = Path(output_path).resolve()
        portable_name = PureWindowsPath(file_name)
        if portable_name.drive or portable_name.root or ".." in portable_name.parts:
            raise ValueError(f"ComfyUI media filename escapes the run output directory: {file_name}")
        candidate = (root / file_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"ComfyUI media filename escapes the run output directory: {file_name}") from exc
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return str(candidate)


class AgenticMediaGenerator:
    def __init__(self, host: str | None = None, port: int | None = None, timeout: int | None = None) -> None:
        self.communicator = AgenticComfyCommunicator(host, port, timeout=timeout)
        self.communicator.connect_websocket()

    def generate(
        self,
        workflow_path: str,
        updates: list[dict[str, Any]],
        output_dir: str,
        file_prefix: str = "media",
        workflow: dict[str, Any] | None = None,
        timing_sink: list[ComfyPromptTiming] | None = None,
    ) -> list[str]:
        workflow = workflow or json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        success, saved_files = self.communicator.process_workflow(
            workflow=workflow,
            updates=updates,
            output_path=output_dir,
            file_name=file_prefix,
            auto_close=False,
            timing_sink=timing_sink,
        )
        if not success:
            error = saved_files[0] if saved_files else "Unknown error"
            raise RuntimeError(f"Media generation failed for {workflow_path}: {error}")
        return saved_files

    def upload_image(
        self,
        image_path: str,
        *,
        include_sha256: bool = False,
    ) -> ComfyUploadResult:
        return self.communicator.upload_image(
            image_path,
            include_sha256=include_sha256,
        )

    def get_object_info(self, node_type: str | None = None) -> dict[str, Any]:
        """Expose ComfyUI node metadata for workflow preflight checks."""
        return self.communicator.get_object_info(node_type)


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
    def _generate_custom_updates(
        cls, workflow: dict[str, Any], updates_config: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                # (e.g. Krea2 workflows). Reuse the same title/is_negative filters so
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
    def _generate_builtin_text_updates(
        cls,
        workflow: dict[str, Any],
        description: str,
        **additional_params: Any,
    ) -> list[dict[str, Any]]:
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
            indices = [
                index
                for index in cls.get_node_indices(workflow, node_type)
                if index not in exclude
            ]
            if indices:
                updates.extend(
                    cls.create_node_update(node_type, index, {input_key: seed + index}) for index in indices
                )
                if use_noise_seed and input_key == "noise_seed":
                    break
        return updates
