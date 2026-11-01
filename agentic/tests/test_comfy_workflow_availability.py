from __future__ import annotations

import socket
from pathlib import Path

import pytest

from agentic.assets.registry import AssetRegistry
from agentic.tools.comfy_workflow_tool import ComfyWorkflowToolset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_user_given_comfyui_is_unavailable_when_an_image_render_is_requested_then_run_directory_is_not_created(
    tmp_path: Path,
) -> None:
    """User Given ComfyUI is unavailable When an image render is requested Then no empty run directory is left behind."""
    run_dir = tmp_path / "run"
    toolset = ComfyWorkflowToolset(
        asset_registry=AssetRegistry(PROJECT_ROOT),
        output_root=tmp_path / "tool-output",
        comfy_host="127.0.0.1",
        comfy_port=_unused_loopback_port(),
    )

    with pytest.raises(RuntimeError, match="ComfyUI is not reachable"):
        toolset.execute(
            "comfy.workflow.text_to_image",
            {"run_dir": str(run_dir), "prompt": "A paper lantern over a quiet canal."},
        )

    assert not run_dir.exists()
