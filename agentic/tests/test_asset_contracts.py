"""Asset readiness acceptance at CLI and real runtime boundaries."""

import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest
import yaml
from PIL import Image

from agentic.app.main import build_runtime
from agentic.runtime.contracts import GoalRequest


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "entrypoint", [("-m", "agentic.app.main"), ("run_media_interface.py",)]
)
def test_user_given_the_cli_when_viewing_help_then_only_implemented_asset_capabilities_are_advertised(
    entrypoint,
):
    """User Given a generation CLI When viewing help Then it does not promise automatic asset downloads that no runtime can execute."""
    result = subprocess.run(
        [sys.executable, *entrypoint, "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--auto-download-assets" not in result.stdout


def test_user_given_a_missing_model_when_the_goal_runs_then_readiness_explains_installation_and_no_media_is_generated(
    tmp_path,
):
    """User Given an image and a workflow with a missing model When its upscale goal runs Then it fails with an installation action and produces no generated media."""
    workflow_dir = tmp_path / "configs/workflow"
    shutil.copytree(REPO / "configs/workflow", workflow_dir)
    name = "upscale_acceptance"
    (workflow_dir / f"{name}.json").write_text(
        json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}),
        encoding="utf-8",
    )
    (workflow_dir / "workflow_config.yaml").write_text(
        yaml.safe_dump(
            {
                "workflows": {
                    name: {
                        "media_types": ["image_upscale"],
                        "required_assets": [
                            {
                                "name": "acceptance-model.safetensors",
                                "kind": "upscale_model",
                                "target_dir": "models/upscale_models",
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    image = tmp_path / "source.png"
    Image.new("RGB", (320, 320), "green").save(image)
    output = tmp_path / "output"
    planner, runner, _ = build_runtime(
        tmp_path, output_root=output, comfy_root=tmp_path / "empty_models"
    )
    runner.portfolio_memory = None
    result = runner.run(
        planner.build_plan(
            GoalRequest(
                prompt="Enlarge the source image",
                media_type="image_upscale",
                constraints={"workflow_name": name, "input_image_path": str(image)},
            )
        )
    )
    assert result.status == "failed"
    readiness = result.state.node_outputs["asset-check"]
    assert "auto_download" not in readiness
    assert readiness["asset_status"][0]["status"] == "missing"
    assert readiness["asset_status"][0]["action"] == "manual_setup"
    assert any(
        "acceptance-model.safetensors" in log
        for record in result.records
        for log in record.logs
    )
    assert not any(
        path.suffix.lower() in {".png", ".mp4"} for path in output.rglob("*")
    )
