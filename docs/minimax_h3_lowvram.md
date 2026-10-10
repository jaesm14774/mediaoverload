# Local H3 on RTX 4060 8GB

The production H3 video backend is **Powered by WanGP**, using Profile 4, Sage2 and the existing pruned Q4 GGUF diffusion models. See [current workflows, isolated installation and delivery contract](wan2gp_h3.md).

Krea images still use ComfyUI. H3 generation no longer requires the Comfy H3 custom nodes or its old workflow graphs. Both backends share the existing diffusion / text-encoder files through local WanGP model definitions; WanGP uses the original audio-VAE format.

Register the already installed files with `scripts/setup_wan2gp_h3.py`. `scripts/setup_minimax_h3.py` remains a shared model-file inventory/downloader; it does not establish WanGP readiness. The workflow asset check validates the isolated WanGP runtime and its actual required files.

The fixed 2026-10-08 cold T2VA comparison is documented in [the matched benchmark](research/wan2gp_local_benchmark_2026-10-08.md): ComfyUI 902.976 seconds, WanGP 199.482 seconds, one 608x352 / 124-frame case. New strategy runs are separate evidence and do not inherit that speed ratio automatically.
