# MiniMax H3 RTX 4060 validation policy

The executable Ref2VA default is `minimax_h3_ref2va.json`. It uses the
community `MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` diffusion model through
`UnetLoaderGGUF`, the existing Q4 Qwen text encoder, and the official H3
video/audio VAEs. Reference audio is intentionally not connected.

All H3 model assets are rooted at the configured D: or E: ComfyUI install.
The current validation root is `D:\ComfyUI_windows_portable`. A model becomes
usable only after exact expected-byte-size validation. `.part`, `.prefix`, and
range files are resumable download artifacts and must never be selected by a
workflow.

The official native model set is selectable at runtime through
`model_profile: native` on the canonical `minimax_h3_ref2va.json` workflow. It
is not the RTX 4060 default because it requires substantially larger native
diffusion and text-encoder assets.
