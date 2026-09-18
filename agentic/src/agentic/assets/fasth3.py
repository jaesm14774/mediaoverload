from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FastH3Asset:
    """An official, non-GGUF asset used by the FastH3 benchmark."""

    name: str
    model_dir: str
    source: str
    expected_size: int
    sha256: str
    role: str

    def target_path(self, model_root: Path) -> Path:
        root = model_root.expanduser().resolve()
        target = (root / self.model_dir / self.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"FastH3 asset target escapes model root: {target}")
        return target


FASTH3_ASSETS: tuple[FastH3Asset, ...] = (
    FastH3Asset(
        name="fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        model_dir="diffusion_models",
        source=(
            "https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/resolve/main/"
            "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
        ),
        expected_size=22_128_378_696,
        sha256="0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8",
        role="fasth3_v2_diffusion",
    ),
    FastH3Asset(
        name="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        model_dir="diffusion_models",
        source=(
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        ),
        expected_size=20_970_379_616,
        sha256="e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        role="minimax_h3_native_diffusion",
    ),
    FastH3Asset(
        name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        model_dir="text_encoders",
        source=(
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        ),
        expected_size=27_141_342_152,
        sha256="bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923",
        role="minimax_h3_native_text_encoder",
    ),
    FastH3Asset(
        name="minimax_h3_video_vae_fp16.safetensors",
        model_dir="vae",
        source=(
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            "vae/minimax_h3_video_vae_fp16.safetensors"
        ),
        expected_size=5_207_808_496,
        sha256="7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        role="minimax_h3_video_vae",
    ),
    FastH3Asset(
        name="minimax_h3_audio_vae_fp32.safetensors",
        model_dir="vae",
        source=(
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            "vae/minimax_h3_audio_vae_fp32.safetensors"
        ),
        expected_size=605_254_808,
        sha256="8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        role="minimax_h3_audio_vae",
    ),
)


def get_fast_h3_asset(name: str) -> FastH3Asset:
    normalized = str(name).strip()
    for asset in FASTH3_ASSETS:
        if asset.name == normalized:
            return asset
    choices = ", ".join(asset.name for asset in FASTH3_ASSETS)
    raise KeyError(f"Unknown FastH3 asset {name!r}; choose one of: {choices}")


def inspect_fast_h3_asset(
    asset: FastH3Asset,
    model_root: Path,
    *,
    verify_hash: bool = True,
) -> dict[str, object]:
    target = asset.target_path(model_root)
    part = Path(f"{target}.part")
    if not target.exists():
        return {
            "name": asset.name,
            "role": asset.role,
            "path": str(target),
            "status": "partial" if part.exists() else "missing",
            "size": part.stat().st_size if part.exists() else 0,
            "expected_size": asset.expected_size,
            "source": asset.source,
            "sha256": asset.sha256,
        }
    size = target.stat().st_size
    if size != asset.expected_size:
        status = "corrupt"
    elif not verify_hash:
        status = "size_verified"
    elif _sha256(target) != asset.sha256:
        status = "checksum_mismatch"
    else:
        status = "ready"
    return {
        "name": asset.name,
        "role": asset.role,
        "path": str(target),
        "status": status,
        "size": size,
        "expected_size": asset.expected_size,
        "source": asset.source,
        "sha256": asset.sha256,
    }


def inspect_fast_h3_assets(model_root: Path, *, verify_hash: bool = True) -> list[dict[str, object]]:
    return [inspect_fast_h3_asset(asset, model_root, verify_hash=verify_hash) for asset in FASTH3_ASSETS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
