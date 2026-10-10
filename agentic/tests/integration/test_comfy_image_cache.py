from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.request import urlopen

import pytest
from PIL import Image

from agentic.tools.comfy_backend import AgenticMediaGenerator


pytestmark = pytest.mark.integration


def test_user_given_unchanged_image_inputs_when_saving_successive_candidates_then_provider_cache_is_reused(tmp_path: Path) -> None:
    """User Given unchanged image inputs When saving successive candidates
    Then both images are saved and ComfyUI reports reusing the upstream cache.

    This isolates the provider lifecycle with a stock image node. Diffusion
    model speed and GPU residency require their separate measured benchmarks.
    """
    if os.environ.get("AGENTIC_COMFY_INTEGRATION") != "1":
        pytest.skip("Real ComfyUI requires AGENTIC_COMFY_INTEGRATION=1")
    host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    port = int(os.environ.get("COMFYUI_PORT", "8188"))
    base = f"http://{host}:{port}"

    def get(path: str) -> dict:
        with urlopen(base + path, timeout=10) as response:
            return json.load(response)

    def require_idle() -> None:
        queue = get("/queue")
        if queue["queue_running"] or queue["queue_pending"]:
            pytest.skip("ComfyUI occupied; no test prompt queued or interrupted")

    require_idle()
    label = "cache_contract_" + uuid.uuid4().hex
    workflow = {
        "1": {"class_type": "EmptyImage", "inputs": {
            "width": 32, "height": 24, "batch_size": 1, "color": 0x2873AC,
        }},
        "2": {"class_type": "SaveImage", "inputs": {
            "images": ["1", 0], "filename_prefix": "benchmarks/" + label,
        }},
    }
    workflow_path = tmp_path / "image.json"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    generator = AgenticMediaGenerator(host, port, timeout=30)
    saved: list[str] = []
    try:
        for index in range(2):
            require_idle()
            saved.extend(generator.generate(
                str(workflow_path),
                [{"type": "direct_update", "node_id": "2", "inputs": {
                    "filename_prefix": f"benchmarks/{label}_{index}",
                }}],
                str(tmp_path),
            ))
        histories = [item for item in get("/history?max_items=20").values()
                     if item["prompt"][3].get("client_id") == generator.communicator.client_id]
    finally:
        generator.communicator.ws.close()

    assert len(saved) == 2 and len(set(saved)) == 2
    for file in saved:
        with Image.open(file) as image:
            image.load()
            assert image.size == (32, 24)
            assert image.getpixel((0, 0)) == (0x28, 0x73, 0xAC)
    assert len(histories) == 2
    second = max(histories, key=lambda item: item["prompt"][0])
    cached = {node for event, data in second["status"]["messages"]
              if event == "execution_cached" for node in data["nodes"]}
    assert "1" in cached, "A successful image must retain reusable upstream node outputs"
