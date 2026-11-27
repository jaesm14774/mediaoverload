from __future__ import annotations

import threading
import time
from pathlib import Path

from agentic.runtime.registry import ToolRegistry
from agentic.tools.flow_handoff import FlowHandoffToolset, register_flow_handoff_tools


def test_flow_handoff_overrides_render_tools(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_flow_handoff_tools(registry, tmp_path)
    names = registry.list_names()
    assert {"asset.ensure_workflow_ready", "comfy.workflow.text_to_image", "comfy.render_h3"} <= set(names)
    ready = registry.call("asset.ensure_workflow_ready", {"workflow_name": "x"})
    assert ready["workflow_name"] == "x" and ready["asset_status"] == []


def test_failed_marker_raises_instead_of_hanging(tmp_path: Path, monkeypatch) -> None:
    import pytest

    handoff = tmp_path / "handoff"
    monkeypatch.setenv("FLOW_HANDOFF_DIR", str(handoff))
    monkeypatch.setenv("FLOW_HANDOFF_TIMEOUT_SECONDS", "30")

    def operator() -> None:
        for _ in range(100):
            if (handoff / "0001.request.json").exists():
                break
            time.sleep(0.05)
        (handoff / "0001.failed").write_text("flow refused", encoding="utf-8")

    thread = threading.Thread(target=operator)
    thread.start()
    with pytest.raises(RuntimeError, match="flow refused"):
        FlowHandoffToolset(tmp_path).text_to_image({"run_dir": str(tmp_path / "run"), "prompt": "p"})
    thread.join()


def test_sequence_numbers_are_reserved_uniquely(tmp_path: Path) -> None:
    from agentic.tools.flow_handoff import _next_sequence

    assert [_next_sequence(tmp_path) for _ in range(3)] == [1, 2, 3]


def test_render_video_delivers_exact_length_with_stereo_audio(tmp_path: Path, monkeypatch) -> None:
    import shutil
    import subprocess

    import pytest

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe are required")
    handoff = tmp_path / "handoff"
    monkeypatch.setenv("FLOW_HANDOFF_DIR", str(handoff))
    monkeypatch.setenv("FLOW_HANDOFF_TIMEOUT_SECONDS", "60")

    def operator() -> None:
        for _ in range(100):
            if (handoff / "0001.request.json").exists():
                break
            time.sleep(0.05)
        clip = handoff / "0001.out" / "flow.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=5",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-ac", "1", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], check=True)
        (handoff / "0001.done").touch()

    thread = threading.Thread(target=operator)
    thread.start()
    result = FlowHandoffToolset(tmp_path).render_video(
        {"run_dir": str(tmp_path / "run"), "prompt": "p", "width": 640, "height": 360, "length": 124, "fps": 24}
    )
    thread.join()
    target = Path(result["saved_files"][0])
    assert target.is_file()
    from agentic.tools.comfy_h3 import probe_video

    info = probe_video(target)
    assert (info["width"], info["height"], info["length"]) == (640, 360, 124)
    assert info["audio"] is not None and info["audio"]["channels"] == 2


def test_render_video_rejects_unreachable_length_before_any_request(tmp_path: Path, monkeypatch) -> None:
    import pytest

    handoff = tmp_path / "handoff"
    monkeypatch.setenv("FLOW_HANDOFF_DIR", str(handoff))
    with pytest.raises(ValueError):
        FlowHandoffToolset(tmp_path).render_video({"run_dir": str(tmp_path / "run"), "prompt": "p", "length": 900})
    assert not list(handoff.glob("*.request.json"))


def test_text_to_image_blocks_until_operator_signals_done(tmp_path: Path, monkeypatch) -> None:
    handoff = tmp_path / "handoff"
    monkeypatch.setenv("FLOW_HANDOFF_DIR", str(handoff))
    monkeypatch.setenv("FLOW_HANDOFF_TIMEOUT_SECONDS", "30")

    def operator() -> None:
        request = handoff / "0001.request.json"
        for _ in range(100):
            if request.exists():
                break
            time.sleep(0.05)
        out = handoff / "0001.out"
        (out / "a.jpg").write_bytes(b"jpeg-bytes")
        (handoff / "0001.done").touch()

    thread = threading.Thread(target=operator)
    thread.start()
    result = FlowHandoffToolset(tmp_path).text_to_image(
        {"run_dir": str(tmp_path / "run"), "prompt": "p", "width": 640, "height": 360, "image_count": 1}
    )
    thread.join()
    assert result["provider"] == "flow_handoff"
    assert len(result["saved_files"]) == 1 and Path(result["saved_files"][0]).read_bytes() == b"jpeg-bytes"
    assert "16:9" in (handoff / "0001.request.md").read_text(encoding="utf-8")
