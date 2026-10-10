"""ComfyUI model-file requirements for the existing MiniMax H3 profiles."""
from __future__ import annotations

from typing import Any

DIFFUSION_MODELS = {
    "fl2va": ("minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", 11_377_542_880),
    "ref2va": ("MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf", 11_564_180_576),
}
TEXT_ENCODERS = {
    "q4": ("qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14_576_977_888),
    "q2": ("qwen3vl-32B-MiniMax-H3-Q2_K.gguf", 8_487_968_160),
}


def h3_requirements(mode: str, profile: str = "q4") -> list[dict[str, Any]]:
    if profile not in TEXT_ENCODERS:
        raise ValueError(f"Unsupported H3 model_profile: {profile}; choose q4 or q2")
    architecture = "ref2va" if mode == "ref2va" else "fl2va"
    models = [
        (*DIFFUSION_MODELS[architecture], "diffusion_model", "unet"),
        (*TEXT_ENCODERS[profile], "text_encoder", "clip"),
        ("minimax_h3_video_vae_fp16.safetensors", 5_207_808_496, "video_vae", "vae"),
        ("minimax_h3_audio_vae_fp32.safetensors", 605_254_808, "audio_vae", "vae"),
    ]
    return [{"name": name, "kind": kind, "target_dir": f"ComfyUI/models/{directory}",
             "expected_size": size} for name, size, kind, directory in models]
