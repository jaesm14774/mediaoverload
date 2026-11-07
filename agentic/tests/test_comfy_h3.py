"""H3 input and delivery contracts; real GPU acceptance runs separately."""
from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest
from PIL import Image

from agentic.assets.registry import AssetRegistry
from agentic.runtime.h3_modes import mode_contract
from agentic.tools.comfy_h3 import ComfyH3Toolset, build_h3_workflow, deliver_video

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va", "ref2va"])
def test_user_given_an_h3_mode_when_prepared_then_its_prompt_and_conditioning_contract_are_kept(tmp_path: Path, mode: str):
    """User Given a valid scene and real conditioning images When a Comfy H3 request is prepared Then only the selected mode's anchors or references are connected and the full scene is retained."""
    registry = AssetRegistry(REPO)
    manifest = registry.get_manifest(mode_contract(mode).workflow_name)
    first, last = tmp_path / "first.png", tmp_path / "last.png"
    Image.new("RGB", (608, 352), "green").save(first)
    Image.new("RGB", (608, 352), "blue").save(last)
    payload = {"prompt": "A hero catches a seed.\nWind blows and a bell rings.", "width": 640,
               "height": 360, "length": 120, "seed": 42}
    if mode in {"i2va", "fl2va"}:
        payload["image_path"] = str(first)
    if mode in {"l2va", "fl2va"}:
        payload["last_image_path"] = str(last)
    if mode == "ref2va":
        payload.update(reference_image_paths=[str(first), str(last)], steps=8, model_profile="q2")
    graph, evidence = build_h3_workflow(registry.load_workflow_template(manifest), mode, payload)
    inputs = graph["5"]["inputs"]
    assert inputs["prompt"] == payload["prompt"]
    assert graph["9"]["inputs"]["noise_seed"] == payload["seed"]
    assert evidence["delivery"] == {"width": 640, "height": 360, "length": 120, "fps": 24}
    assert ("first_frame" in inputs) == (mode in {"i2va", "fl2va"})
    assert ("last_frame" in inputs) == (mode in {"l2va", "fl2va"})
    refs = [key for key in inputs if key.startswith("ref_images.")]
    assert len(refs) == (2 if mode == "ref2va" else 0)
    assert all("audio" not in key for key in inputs if key.startswith("ref_"))
    if mode == "ref2va":
        assert "Ref2VA" in graph["1"]["inputs"]["unet_name"]
        assert "Q2_K" in graph["2"]["inputs"]["clip_name"]
        assert len(graph["7"]["inputs"]["sigmas"].split(",")) == payload["steps"] + 1
        assert [ref["path"] for ref in evidence["reference_manifest"]] == [str(first), str(last)]
    else:
        # The distilled model is only valid at its trained 8-step schedule.
        assert "fasth3" in graph["1"]["inputs"]["unet_name"]
        assert graph["7"]["class_type"] == "BasicScheduler"
        assert graph["7"]["inputs"]["steps"] == 8
        assert evidence["steps"] == 8


@pytest.mark.parametrize("mode", ["t2va", "i2va", "fl2va", "l2va"])
def test_user_given_a_distilled_h3_mode_when_a_different_step_count_or_profile_is_requested_then_it_is_rejected(
    tmp_path: Path, mode: str
):
    """User Given the 8-step distilled H3 route When another step count or a GGUF profile is requested Then the request is rejected instead of silently degrading quality."""
    registry = AssetRegistry(REPO)
    template = registry.load_workflow_template(registry.get_manifest(mode_contract(mode).workflow_name))
    frame = tmp_path / "frame.png"
    Image.new("RGB", (608, 352), "green").save(frame)
    base = {"prompt": "A hero jumps.", "image_path": str(frame), "last_image_path": str(frame)}
    base = {key: value for key, value in base.items()
            if (key == "prompt") or (key == "image_path" and mode in {"i2va", "fl2va"})
            or (key == "last_image_path" and mode in {"l2va", "fl2va"})}
    for override in ({"steps": 16}, {"model_profile": "q4"}):
        with pytest.raises(ValueError, match="distilled"):
            build_h3_workflow(template, mode, {**base, **override})
    graph, _ = build_h3_workflow(template, mode, {**base, "steps": 8})
    assert graph["7"]["inputs"]["steps"] == 8


def test_user_given_a_reference_video_when_preparing_ref2va_then_digest_is_deferred_until_upload(tmp_path: Path):
    """User Given a readable reference video When preparing a Ref2VA request Then metadata is ready and hashing can be completed with its upload."""
    video = tmp_path / "motion.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=64x64:rate=24:duration=1",
         "-c:v", "libx264", "-preset", "ultrafast", str(video)],
        check=True,
        capture_output=True,
    )
    registry = AssetRegistry(REPO)
    manifest = registry.get_manifest(mode_contract("ref2va").workflow_name)
    graph, evidence = build_h3_workflow(
        registry.load_workflow_template(manifest),
        "ref2va",
        {"prompt": "A dancer turns slowly.", "reference_video_paths": [str(video)]},
    )

    video_node = next(node for node in graph.values() if node["class_type"] == "LoadVideo")
    file_evidence = next(item for item in evidence["conditioning_files"] if item["role"] == "<Video 1>")
    assert video_node["inputs"]["file"] == video.name
    assert file_evidence["path"] == str(video.resolve())
    assert file_evidence["size_bytes"] == video.stat().st_size
    assert "sha256" not in file_evidence


def test_user_given_a_reference_file_without_video_when_preparing_ref2va_then_it_is_rejected(tmp_path: Path):
    """User Given an audio-only reference file When preparing a Ref2VA request Then the file is rejected before ComfyUI submission."""
    audio = tmp_path / "motion.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "aac", str(audio)],
        check=True,
        capture_output=True,
    )
    registry = AssetRegistry(REPO)
    manifest = registry.get_manifest(mode_contract("ref2va").workflow_name)

    with pytest.raises(ValueError, match="not readable media"):
        build_h3_workflow(
            registry.load_workflow_template(manifest),
            "ref2va",
            {"prompt": "A dancer turns slowly.", "reference_video_paths": [str(audio)]},
        )


@pytest.mark.parametrize("mode,payload", [
    ("i2va", {}), ("fl2va", {}), ("l2va", {}), ("ref2va", {}),
    ("t2va", {"image_path": "forbidden.png"}),
    ("t2va", {"prompt": {"description": "A hero jumps."}}),
    ("ref2va", {"model_profile": "unknown", "reference_image_paths": ["missing.png"]}),
    ("t2va", {"fps": 30}), ("t2va", {"length": 500}),
])
def test_user_given_an_invalid_render_request_when_called_then_it_fails_before_gpu_submission(tmp_path: Path, mode: str, payload: dict):
    """User Given missing conditioning or unsupported settings When the H3 tool is called Then it rejects the request before reaching the GPU service."""
    tool = ComfyH3Toolset(AssetRegistry(REPO), tmp_path, comfy_port=1)
    with pytest.raises(ValueError):
        tool.execute({"workflow_name": mode_contract(mode).workflow_name, "run_dir": str(tmp_path),
                      "prompt": "A hero jumps.", **payload})


def test_user_given_missing_models_when_rendering_then_no_media_is_fabricated(tmp_path: Path):
    """User Given missing model assets When rendering Then an actionable asset error is returned and no video is fabricated."""
    tool = ComfyH3Toolset(AssetRegistry(REPO, asset_root=tmp_path / "missing"), tmp_path, comfy_port=1)
    with pytest.raises(RuntimeError, match="assets missing or corrupt"):
        tool.execute({"workflow_name": mode_contract("t2va").workflow_name,
                      "run_dir": str(tmp_path), "prompt": "A hero jumps."})
    assert not list(tmp_path.rglob("*.mp4"))


def test_user_given_missing_h3_models_when_preflight_fails_then_summary_records_stage_timings(
    tmp_path: Path,
):
    """User Given required H3 models are missing When preflight fails Then the run summary preserves the error and elapsed preflight stages."""
    tool = ComfyH3Toolset(
        AssetRegistry(REPO, asset_root=tmp_path / "missing"),
        tmp_path,
        comfy_port=1,
    )
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="assets missing or corrupt"):
        tool.execute(
            {
                "workflow_name": mode_contract("t2va").workflow_name,
                "run_dir": str(run_dir),
                "prompt": "A hero jumps.",
            }
        )

    summary = json.loads((run_dir / "comfy_h3_summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is False
    assert "assets missing or corrupt" in summary["error"]
    assert {
        "request_validation_seconds",
        "workflow_prepare_seconds",
        "asset_readiness_seconds",
    }.issubset(summary["phase_seconds"])
    assert all(value >= 0 for value in summary["phase_seconds"].values())
    assert summary["wall_seconds"] >= summary["phase_seconds"]["asset_readiness_seconds"]


@pytest.mark.parametrize("video_count", [0, 5, 2.5, True, "invalid"])
def test_user_given_an_invalid_h3_video_count_when_the_tool_is_called_then_it_fails_before_asset_readiness(
    tmp_path: Path,
    video_count: object,
):
    """User Given an invalid H3 candidate count When the tool is called Then it rejects the count before checking model assets."""
    tool = ComfyH3Toolset(
        AssetRegistry(REPO, asset_root=tmp_path / "missing"),
        tmp_path,
        comfy_port=1,
    )
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="video_count must"):
        tool.execute(
            {
                "workflow_name": mode_contract("t2va").workflow_name,
                "run_dir": str(run_dir),
                "prompt": "A hero jumps.",
                "video_count": video_count,
            }
        )

    assert not run_dir.exists()


def test_user_given_an_unaligned_delivery_when_exported_then_both_anchors_canvas_duration_and_audio_are_kept(tmp_path: Path):
    """User Given a real native MP4 with a distinct last frame When exported to the requested grid Then the last anchor, exact canvas, frame count and stereo audio are preserved."""
    source, target = tmp_path / "native.mp4", tmp_path / "delivered.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=608x352:rate=24:duration=5.125",
                    "-f", "lavfi", "-i", "color=c=blue:size=608x352:rate=24:duration=0.04166667",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-t", str(124 / 24),
                    "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)], check=True, capture_output=True)
    requested = {"width": 640, "height": 360, "length": 120, "fps": 24}
    actual = deliver_video(source, target, requested)
    assert {key: actual[key] for key in requested} == requested
    assert actual["audio"]["channels"] == 2
    assert actual["duration_seconds"] == pytest.approx(120 / 24, abs=1 / 24)
    frame = tmp_path / "last.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-vf", "select=eq(n\\,119)",
                    "-frames:v", "1", str(frame)], check=True, capture_output=True)
    red, green, blue = Image.open(frame).convert("RGB").getpixel((320, 180))
    assert blue > red + green
