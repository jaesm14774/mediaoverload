"""ComfyUI model-file requirements for the MiniMax H3 routes.

Every route is a few-step distillation, which is only valid at its trained step count. T2VA, I2VA,
FL2VA and L2VA run the MiniMax H3 FL2VA base patched by the 2-step pi-H3 DWD adapter, which predicts
the pixels directly so no video VAE decode runs. Ref2VA runs the Q4 GGUF checkpoint patched by the
lightx2v 8-step turbo LoRA, with a selectable text-encoder profile.
"""
from __future__ import annotations

from typing import Any

PIH3_STEPS = 2
REF2VA_STEPS = 8
PIH3_DIFFUSION = ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", 20_970_379_616)
PIH3_ADAPTER = ("pi_h3_2step_dwd_adapter_int8_convrot.safetensors", 2_192_176_240)
H3_TEXT_ENCODER = ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15_687_142_551)

REF2VA_DIFFUSION = ("MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf", 11_564_180_576)
REF2VA_TEXT_ENCODERS = {
    "q4": ("qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", 14_576_977_888),
    "q2": ("qwen3vl-32B-MiniMax-H3-Q2_K.gguf", 8_487_968_160),
}
REF2VA_TURBO_LORA = ("minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", 1_956_193_000)
VIDEO_VAE = ("minimax_h3_video_vae_fp16.safetensors", 5_207_808_496)
AUDIO_VAE = ("minimax_h3_audio_vae_fp32.safetensors", 605_254_808)


def uses_pih3(mode: str) -> bool:
    return mode != "ref2va"


def h3_steps(mode: str) -> int:
    return PIH3_STEPS if uses_pih3(mode) else REF2VA_STEPS


def h3_requirements(mode: str, profile: str = "q4") -> list[dict[str, Any]]:
    """Model files for one H3 mode; `profile` selects the Ref2VA text encoder only."""
    if uses_pih3(mode):
        models = [
            (*PIH3_DIFFUSION, "diffusion_model", "diffusion_models"),
            (*PIH3_ADAPTER, "lora", "loras"),
            (*H3_TEXT_ENCODER, "text_encoder", "text_encoders"),
        ]
    else:
        if profile not in REF2VA_TEXT_ENCODERS:
            raise ValueError(f"Unsupported H3 model_profile: {profile}; choose q4 or q2")
        models = [
            (*REF2VA_DIFFUSION, "diffusion_model", "unet"),
            (*REF2VA_TEXT_ENCODERS[profile], "text_encoder", "clip"),
            (*REF2VA_TURBO_LORA, "lora", "loras"),
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
