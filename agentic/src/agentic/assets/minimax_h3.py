"""ComfyUI model-file requirements for the MiniMax H3 routes.

T2VA, I2VA, FL2VA and L2VA run the FastVideo FastH3 8-step distillation, which is only valid at
its trained step count. Ref2VA has no distilled checkpoint and keeps the Q4 GGUF recipe with a
selectable text-encoder profile.
"""
from __future__ import annotations

from typing import Any

FASTH3_STEPS = 8
FASTH3_DIFFUSION = ("fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", 22_128_378_696)
FASTH3_TEXT_ENCODER = ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15_687_142_551)

REF2VA_DIFFUSION = ("MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf", 11_564_180_576)
REF2VA_TEXT_ENCODERS = {
    "q4": ("qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14_576_977_888),
    "q2": ("qwen3vl-32B-MiniMax-H3-Q2_K.gguf", 8_487_968_160),
}
VIDEO_VAE = ("minimax_h3_video_vae_fp16.safetensors", 5_207_808_496)
AUDIO_VAE = ("minimax_h3_audio_vae_fp32.safetensors", 605_254_808)


def uses_fasth3(mode: str) -> bool:
    return mode != "ref2va"


def h3_requirements(mode: str, profile: str = "q4") -> list[dict[str, Any]]:
    """Model files for one H3 mode; `profile` selects the Ref2VA text encoder only."""
    if uses_fasth3(mode):
        models = [
            (*FASTH3_DIFFUSION, "diffusion_model", "diffusion_models"),
            (*FASTH3_TEXT_ENCODER, "text_encoder", "text_encoders"),
        ]
    else:
        if profile not in REF2VA_TEXT_ENCODERS:
            raise ValueError(f"Unsupported H3 model_profile: {profile}; choose q4 or q2")
        models = [
            (*REF2VA_DIFFUSION, "diffusion_model", "unet"),
            (*REF2VA_TEXT_ENCODERS[profile], "text_encoder", "clip"),
        ]
    models += [(*VIDEO_VAE, "video_vae", "vae"), (*AUDIO_VAE, "audio_vae", "vae")]
    return [
        {
            "name": name,
            "kind": kind,
            "target_dir": f"ComfyUI/models/{directory}",
            "expected_size": size,
        }
        for name, size, kind, directory in models
    ]
