from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class MiniMaxH3Asset:
    name: str
    target_dir: str
    source: str
    expected_size: int | None
    role: str
    trust: str
    sha256: str | None = None

    def target_path(self, comfy_root: Path) -> Path:
        root = comfy_root.expanduser().resolve()
        path = (root / self.target_dir / self.name).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"H3 asset target escapes ComfyUI root: {path}")
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MiniMaxH3Profile:
    name: str
    workflow_name: str
    summary: str
    assets: tuple[MiniMaxH3Asset, ...]
    width: int
    height: int
    length: int
    steps: int
    sampler: str
    use_spectrum_for_draft: bool
    custom_nodes: tuple[str, ...]
    media_types: tuple[str, ...] = (
        "image_to_video",
        "image_to_video_audio",
        "text2video",
        "long_video",
    )

    @property
    def total_bytes(self) -> int:
        return sum(asset.expected_size or 0 for asset in self.assets)

    @property
    def unknown_size_asset_count(self) -> int:
        return sum(asset.expected_size is None for asset in self.assets)

    @property
    def total_size_human(self) -> str:
        known = _format_bytes(self.total_bytes)
        if self.unknown_size_asset_count:
            return f"at least {known} + {self.unknown_size_asset_count} asset with unknown size"
        return known

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assets"] = [asset.to_dict() for asset in self.assets]
        payload["custom_nodes"] = list(self.custom_nodes)
        payload["total_bytes"] = self.total_bytes
        payload["total_size_human"] = self.total_size_human
        return payload


def _asset(
    name: str,
    target_dir: str,
    source: str,
    expected_size: int | None,
    role: str,
    trust: str,
) -> MiniMaxH3Asset:
    return MiniMaxH3Asset(
        name=name,
        target_dir=target_dir,
        source=source,
        expected_size=expected_size,
        role=role,
        trust=trust,
    )


_VIDEO_VAE = _asset(
    "minimax_h3_video_vae_fp16.safetensors",
    "ComfyUI/models/vae",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
    5_207_808_496,
    "video_vae",
    "official",
)
_AUDIO_VAE = _asset(
    "minimax_h3_audio_vae_fp32.safetensors",
    "ComfyUI/models/vae",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
    605_254_808,
    "audio_vae",
    "official",
)

_H3_FL2VA_Q4 = _asset(
    "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf",
    "ComfyUI/models/unet",
    "https://huggingface.co/molbal/MiniMax-H3-GGUF/resolve/main/minimax_h3_fl2va_pruned_fp8_Q4_0.gguf",
    11_377_542_880,
    "diffusion_model",
    "community",
)
_H3_TEXT_Q4 = _asset(
    "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
    "ComfyUI/models/clip",
    "https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
    14_576_977_888,
    "text_encoder",
    "community",
)
_H3_TEXT_Q2 = _asset(
    "qwen3vl-32B-MiniMax-H3-Q2_K.gguf",
    "ComfyUI/models/clip",
    "https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q2_K.gguf",
    8_487_968_160,
    "text_encoder",
    "community",
)

_H3_REF2VA_Q4 = _asset(
    "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf",
    "ComfyUI/models/unet",
    "https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF/resolve/main/MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf",
    11_564_180_576,
    "ref2va_diffusion_model",
    "community",
)

_NATIVE_DIFFUSION = _asset(
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "ComfyUI/models/diffusion_models",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    20_970_379_616,
    "diffusion_model",
    "official",
)
_NATIVE_TEXT = _asset(
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "ComfyUI/models/text_encoders",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    15_687_142_551,
    "text_encoder",
    "official",
)
_NATIVE_REF2VA_DIFFUSION = _asset(
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "ComfyUI/models/diffusion_models",
    "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    None,
    "diffusion_model",
    "official",
)


PROFILES: dict[str, MiniMaxH3Profile] = {
    "balanced-lowvram": MiniMaxH3Profile(
        name="balanced-lowvram",
        workflow_name="minimax_h3_lowvram_i2v",
        summary=(
            "RTX 4060 8GB default: community Q4 diffusion + Q4 text encoder, "
            "native H3 audio/video nodes, system-RAM offload, 0.2MP and ~5s draft."
        ),
        assets=(_H3_FL2VA_Q4, _H3_TEXT_Q4, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=True,
        custom_nodes=("molbal/ComfyUI-GGUF", "xmarre/ComfyUI-Spectrum-MiniMax-H3"),
    ),
    "ultra-lowvram": MiniMaxH3Profile(
        name="ultra-lowvram",
        workflow_name="minimax_h3_lowvram_i2v",
        summary=(
            "6-8GB emergency profile: same Q4 diffusion with Q2 text encoder; "
            "use only when balanced-lowvram cannot initialize or repeatedly OOMs."
        ),
        assets=(_H3_FL2VA_Q4, _H3_TEXT_Q2, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=True,
        custom_nodes=("molbal/ComfyUI-GGUF", "xmarre/ComfyUI-Spectrum-MiniMax-H3"),
    ),
    "native-quality": MiniMaxH3Profile(
        name="native-quality",
        workflow_name="minimax_h3_native_t2v",
        summary=(
            "Official native H3 quantized set: preferred when fidelity matters more "
            "than download size; still uses the same low-resolution 8GB strategy."
        ),
        assets=(_NATIVE_DIFFUSION, _NATIVE_TEXT, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=True,
        custom_nodes=("xmarre/ComfyUI-Spectrum-MiniMax-H3",),
    ),
    "ref2va-native": MiniMaxH3Profile(
        name="ref2va-native",
        workflow_name="minimax_h3_ref2va",
        summary=(
            "Official MiniMax H3 Ref2VA path for multiple reference images and "
            "reference videos; reference audio is intentionally disabled."
        ),
        assets=(_NATIVE_REF2VA_DIFFUSION, _NATIVE_TEXT, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=False,
        custom_nodes=("Kosinkadink/ComfyUI-VideoHelperSuite",),
        media_types=("native_h3_ref2va", "long_video"),
    ),
    "ref2va-lowvram": MiniMaxH3Profile(
        name="ref2va-lowvram",
        workflow_name="minimax_h3_ref2va",
        summary=(
            "RTX 4060 Ref2VA default: community pruned Q4_K_M GGUF diffusion, "
            "existing Q4 text encoder, native H3 VAEs, and no reference audio."
        ),
        assets=(_H3_REF2VA_Q4, _H3_TEXT_Q4, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=False,
        custom_nodes=("molbal/ComfyUI-GGUF", "Kosinkadink/ComfyUI-VideoHelperSuite"),
        media_types=("native_h3_ref2va", "long_video"),
    ),
    "ref2va-ultra-lowvram": MiniMaxH3Profile(
        name="ref2va-ultra-lowvram",
        workflow_name="minimax_h3_ref2va",
        summary=(
            "RTX 4060 emergency Ref2VA profile: the same Q4 Ref2VA diffusion model "
            "with the Q2 text encoder; reference audio remains disabled."
        ),
        assets=(_H3_REF2VA_Q4, _H3_TEXT_Q2, _VIDEO_VAE, _AUDIO_VAE),
        width=608,
        height=352,
        length=124,
        steps=20,
        sampler="res_multistep",
        use_spectrum_for_draft=False,
        custom_nodes=("molbal/ComfyUI-GGUF", "Kosinkadink/ComfyUI-VideoHelperSuite"),
        media_types=("native_h3_ref2va", "long_video"),
    ),
}


def minimax_h3_model_overrides(profile: str, *, reference_to_video: bool = False) -> dict[str, dict[str, Any]]:
    """Return runtime loader overrides without duplicating a workflow JSON.

    The canonical low-VRAM workflows use GGUF loaders. The official native
    model set uses the corresponding native loader classes, so the override
    includes both ``class_type`` and loader inputs. The caller remains
    responsible for ensuring the selected model files exist in ComfyUI.
    """

    normalized = str(profile or "q4").strip().lower().replace("_", "-")
    aliases = {
        "balanced-lowvram": "q4",
        "ref2va-lowvram": "q4",
        "q4-k-m": "q4",
        "ultra-lowvram": "q2",
        "native-quality": "native",
        "ref2va-native": "native",
        "official": "native",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"q4", "q2"}:
        diffusion = _H3_REF2VA_Q4 if reference_to_video else _H3_FL2VA_Q4
        text_encoder = _H3_TEXT_Q2 if normalized == "q2" else _H3_TEXT_Q4
        return {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "replace_inputs": True,
                "inputs": {"unet_name": diffusion.name},
            },
            "2": {
                "class_type": "CLIPLoaderGGUF",
                "replace_inputs": True,
                "inputs": {"clip_name": text_encoder.name, "type": "minimax"},
            }
        }
    if normalized == "native":
        diffusion = _NATIVE_REF2VA_DIFFUSION if reference_to_video else _NATIVE_DIFFUSION
        return {
            "1": {
                "class_type": "UNETLoader",
                "replace_inputs": True,
                "inputs": {"unet_name": diffusion.name, "weight_dtype": "default"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "replace_inputs": True,
                "inputs": {
                    "clip_name": _NATIVE_TEXT.name,
                    "type": "minimax",
                    "device": "default",
                },
            },
        }
    raise ValueError(f"Unknown MiniMax H3 model profile: {profile!r}. Choose q4, q2, or native.")


def get_profile(name: str = "balanced-lowvram") -> MiniMaxH3Profile:
    normalized = str(name or "balanced-lowvram").strip().lower()
    try:
        return PROFILES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown MiniMax H3 profile {name!r}; choose one of: {choices}") from exc


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.2f} GiB"


def inspect_asset(asset: MiniMaxH3Asset, comfy_root: Path) -> dict[str, Any]:
    path = asset.target_path(comfy_root)
    part_path = Path(f"{path}.part")
    if not path.exists():
        return {
            "name": asset.name,
            "path": str(path),
            "status": "partial" if part_path.exists() else "missing",
            "size": part_path.stat().st_size if part_path.exists() else 0,
            "expected_size": asset.expected_size,
            "expected_size_human": _format_bytes(asset.expected_size) if asset.expected_size else None,
            "source": asset.source,
            "trust": asset.trust,
        }
    actual_size = path.stat().st_size
    if asset.expected_size is not None and actual_size != asset.expected_size:
        status = "corrupt"
    elif asset.sha256 and _sha256(path) != asset.sha256:
        status = "checksum_mismatch"
    else:
        status = "ready"
    return {
        "name": asset.name,
        "path": str(path),
        "status": status,
        "size": actual_size,
        "expected_size": asset.expected_size,
        "expected_size_human": _format_bytes(asset.expected_size) if asset.expected_size else None,
        "source": asset.source,
        "trust": asset.trust,
    }


def inspect_profile(profile: MiniMaxH3Profile | str, comfy_root: Path) -> dict[str, Any]:
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    assets = [inspect_asset(asset, comfy_root) for asset in resolved.assets]
    return {
        "profile": resolved.name,
        "workflow_name": resolved.workflow_name,
        "summary": resolved.summary,
        "comfy_root": str(comfy_root.expanduser().resolve()),
        "total_bytes": resolved.total_bytes,
        "total_size_human": resolved.total_size_human,
        "ready": all(item["status"] == "ready" for item in assets),
        "assets": assets,
        "custom_nodes": list(resolved.custom_nodes),
    }


def download_profile(
    profile: MiniMaxH3Profile | str,
    comfy_root: Path,
    *,
    dry_run: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    root = comfy_root.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for asset in resolved.assets:
        target = asset.target_path(root)
        status = inspect_asset(asset, root)
        if status["status"] == "ready":
            results.append({**status, "action": "reuse"})
            continue
        if dry_run:
            results.append({**status, "action": "download"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part_path = Path(f"{target}.part")
        written = _download_asset(asset, part_path, chunk_size=chunk_size)
        if asset.expected_size is not None and written != asset.expected_size:
            raise RuntimeError(
                f"Incomplete MiniMax H3 download for {asset.name}: "
                f"{written} bytes, expected {asset.expected_size}. The .part file is resumable."
            )
        if asset.sha256 and _sha256(part_path) != asset.sha256:
            raise RuntimeError(f"SHA256 mismatch for MiniMax H3 asset {asset.name}; keeping .part for inspection.")
        part_path.replace(target)
        results.append({**inspect_asset(asset, root), "action": "downloaded"})
    return {
        "profile": resolved.name,
        "comfy_root": str(root),
        "dry_run": dry_run,
        "total_bytes": resolved.total_bytes,
        "total_size_human": resolved.total_size_human,
        "ready": all(item["status"] == "ready" for item in results),
        "assets": results,
    }


def _download_asset(asset: MiniMaxH3Asset, part_path: Path, *, chunk_size: int) -> int:
    """Download a large H3 asset with Windows curl resume/retry when available."""
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        try:
            download_timeout = max(1, int(os.environ.get("MINIMAX_H3_DOWNLOAD_TIMEOUT_SECONDS", "1800")))
        except ValueError:
            download_timeout = 1800
        command = [
            curl,
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--max-time",
            str(download_timeout),
            "--continue-at",
            "-",
            "--output",
            str(part_path),
            asset.source,
        ]
        try:
            completed = subprocess.run(command, check=False, timeout=download_timeout + 30)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out downloading {asset.name} after {download_timeout} seconds; "
                "the .part file is resumable."
            ) from exc
        if completed.returncode:
            raise RuntimeError(
                f"Failed to download {asset.name} with curl (exit {completed.returncode}); "
                "the .part file is resumable."
            )
        return part_path.stat().st_size if part_path.exists() else 0

    existing = part_path.stat().st_size if part_path.exists() else 0
    request_headers: dict[str, str] = {}
    if existing:
        request_headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(asset.source, headers=request_headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {asset.name}: HTTP {exc.code}") from exc
    response_status = getattr(response, "status", None)
    if existing and response_status == 200:
        # Some mirrors ignore Range. Restart the partial file rather than corrupt it.
        existing = 0
    mode = "ab" if existing else "wb"
    written = existing
    with response, part_path.open(mode) as output:
        while True:
            block = response.read(chunk_size)
            if not block:
                break
            output.write(block)
            written += len(block)
    return written


def manifest_asset_requirements(profile: MiniMaxH3Profile | str) -> list[dict[str, Any]]:
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    return [
        {
            "name": asset.name,
            "kind": asset.role,
            "target_dir": asset.target_dir,
            "source": asset.source,
            "expected_size": asset.expected_size,
            "sha256": asset.sha256,
        }
        for asset in resolved.assets
    ]


def profile_manifest(profile: MiniMaxH3Profile | str) -> dict[str, Any]:
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    return {
        "name": resolved.workflow_name,
        "profile": resolved.name,
        "summary": resolved.summary,
        "media_types": list(resolved.media_types),
        "required_assets": manifest_asset_requirements(resolved),
        "recommended_defaults": {
            "width": resolved.width,
            "height": resolved.height,
            "length": resolved.length,
            "frame_rate": 24,
            "steps": resolved.steps,
            "sampler": resolved.sampler,
            "use_spectrum_for_draft": resolved.use_spectrum_for_draft,
        },
        "custom_nodes": list(resolved.custom_nodes),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profiles_as_json() -> str:
    return json.dumps({name: profile.to_dict() for name, profile in PROFILES.items()}, indent=2, ensure_ascii=False)


def iter_assets(profile: MiniMaxH3Profile | str) -> Iterable[MiniMaxH3Asset]:
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    return iter(resolved.assets)
