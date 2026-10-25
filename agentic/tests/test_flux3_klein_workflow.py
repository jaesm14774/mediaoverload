"""Acceptance contract for production FLUX 3 Image Klein routing."""

import json
from pathlib import Path

import yaml

from agentic.assets.registry import AssetRegistry


REPO = Path(__file__).resolve().parents[2]
WORKFLOW_NAME = "flux3_image_klein_img2img"


def test_user_given_a_story_or_edit_route_when_workflows_are_selected_then_krea_flux_and_h3_are_applied_at_their_stages():
    """User Given image and story-transition routes When their workflows are selected Then Krea2 creates source images, FLUX edits changed landing frames, and H3 renders video."""
    workflow_dir = REPO / "configs" / "workflow"
    graph = json.loads((workflow_dir / f"{WORKFLOW_NAME}.json").read_text(encoding="utf-8"))
    registry = AssetRegistry(REPO / "agentic", asset_root=REPO)
    manifest = registry.get_manifest(WORKFLOW_NAME)
    metadata = yaml.safe_load((workflow_dir / "workflow_config.yaml").read_text(encoding="utf-8"))[
        "workflows"
    ][f"{WORKFLOW_NAME}.json"]

    assert "image_refine" in manifest.media_types
    assert "flux3 CLIPLoader support" in manifest.summary
    assert "tested on ComfyUI 0.41.0" in manifest.summary
    assert "experimental" not in manifest.summary.lower()

    nodes = {node["class_type"]: node for node in graph.values()}
    assert {"LoadImage", "UNETLoader", "CLIPLoader", "VAELoader", "TextGenerate", "ReferenceLatent", "KSampler", "SaveImage"} <= set(nodes)
    assert nodes["KSampler"]["inputs"]["steps"] == 16
    assert nodes["KSampler"]["inputs"]["cfg"] == 1.0
    assert nodes["KSampler"]["inputs"]["sampler_name"] == "euler"
    assert nodes["KSampler"]["inputs"]["scheduler"] == "simple"
    assert nodes["TextGenerate"]["inputs"]["use_default_template"] is True
    assert nodes["TextGenerate"]["inputs"]["prompt"][0] in graph
    assert any(
        node["class_type"] == "CLIPLoader" and node["inputs"].get("type") == "flux3"
        for node in graph.values()
    )
    assert sum(node["class_type"] == "ReferenceLatent" for node in graph.values()) >= 2
    assert metadata["node_aliases"]["load_image"] in graph

    expected_assets = {
        "flux3-image-klein-int8-convrot.safetensors",
        "qwen3vl_4b_int8_convrot.safetensors",
        "flux2-vae.safetensors",
        "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
    }
    assert {asset.name for asset in manifest.required_assets} == expected_assets
    assert all(asset.expected_size and asset.expected_size > 0 for asset in manifest.required_assets)
    assert all(asset.source and asset.source.startswith("https://huggingface.co/") for asset in manifest.required_assets)

    routing = yaml.safe_load((REPO / "configs" / "routing.yaml").read_text(encoding="utf-8"))["routing"]
    stages = routing["workflow_stage_candidates"]
    for route in ("text2image2image", "native_h3_fl2va_story", "native_h3_l2va_story"):
        assert stages[route]["refine_workflow_name"] == [WORKFLOW_NAME]
    assert stages["native_h3_fl2va_story"]["video_workflow_name"] == ["minimax_h3_fl2va"]
    assert stages["native_h3_l2va_story"]["video_workflow_name"] == ["minimax_h3_l2va"]
