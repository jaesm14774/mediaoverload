"""Installed WanGP runtime and model-file requirements for the H3 catalog."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def wan2gp_config() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    config = yaml.safe_load((repo_root / "configs/wan2gp.yaml").read_text("utf-8"))
    config["root"] = str(Path(os.environ.get("WAN2GP_ROOT") or config["root"]).expanduser().resolve())
    return config


def h3_requirements(mode: str, profile: str = "q4") -> list[dict[str, Any]]:
    root = Path(wan2gp_config()["root"])
    architecture = "ref2va" if mode == "ref2va" else "fl2va"
    definition = root / "finetunes" / f"mediaoverload_h3_{architecture}_{profile}.json"
    files: list[tuple[Path, str, int | None]] = [
        (root / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python"), "runtime", None),
        (root / "shared/api.py", "runtime", None),
        (definition, "model_definition", None),
        (root / "ckpts/MiniMax-H3-video_vae_fp16.safetensors", "video_vae", 5_207_808_496),
        # WanGP requires the original parametrized weights, not the Comfy fused file.
        (root / "ckpts/MiniMax-H3-audio_vae_fp32.safetensors", "audio_vae", 605_429_308),
        (root / "ckpts/minimax_h3/minimax_h3_latent_upscaler_3d_bf16.safetensors", "latent_upscaler", 690_592_992),
    ]
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "vocab.json"):
        files.append((root / "ckpts/Qwen3-VL-32B-Instruct" / name, "tokenizer", None))
    if definition.is_file():
        model = json.loads(definition.read_text("utf-8"))["model"]
        if model["architecture"] != f"minimax_h3_{architecture}_pruned":
            raise ValueError(f"Wrong H3 architecture in {definition}")
        for key, role in (("URLs", "diffusion_model"), ("text_encoder_URLs", "text_encoder")):
            for path in model[key]:
                files.append((Path(path), role, None))
    return [{"name": path.name, "kind": role, "target_dir": str(path.parent), **({"expected_size": size} if size is not None else {})} for path, role, size in files]
