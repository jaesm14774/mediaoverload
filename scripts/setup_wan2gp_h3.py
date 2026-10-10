"""Register existing local H3 GGUF files with the isolated WanGP installation."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True, help="Directory containing models/")
    args = parser.parse_args()
    for architecture, diffusion in (
        ("fl2va", "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf"),
        ("ref2va", "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf"),
    ):
        for profile, text, variant in (
            ("q4", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", "gguf_q4_k_m"),
            ("q2", "qwen3vl-32B-MiniMax-H3-Q2_K.gguf", "gguf_q2_k"),
        ):
            model_path = args.comfy_root / "models/unet" / diffusion
            text_path = args.comfy_root / "models/clip" / text
            for path in (model_path, text_path, args.wan_root / "ckpts/MiniMax-H3-video_vae_fp16.safetensors", args.wan_root / "ckpts/MiniMax-H3-audio_vae_fp32.safetensors"):
                if not path.is_file():
                    raise FileNotFoundError(path)
            name = f"mediaoverload_h3_{architecture}_{profile}"
            definition = {"model": {"name": f"MediaOverload H3 {architecture.upper()} {profile.upper()}", "description": "Local H3 workflow for MediaOverload. Powered by WanGP.", "architecture": f"minimax_h3_{architecture}_pruned", "URLs": [str(model_path.resolve())], "text_encoder_variant": variant, "text_encoder_URLs": [str(text_path.resolve())], "video_vae_file": "MiniMax-H3-video_vae_fp16.safetensors", "audio_vae_file": "MiniMax-H3-audio_vae_fp32.safetensors"}}
            target = args.wan_root / "finetunes" / f"{name}.json"
            target.write_text(json.dumps(definition, indent=2), encoding="utf-8")
            print(target)


if __name__ == "__main__":
    main()
