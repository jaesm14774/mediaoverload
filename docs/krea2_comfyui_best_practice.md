# Krea 2 Turbo on this machine

## Supported route

Krea 2 Turbo creates still images. Image-assisted video routes use Krea to
create six reviewable candidates, then pass the approved image to WanGP
MiniMax H3. The prompt-only H3 T2VA route remains separate.

```text
user prompt -> Krea 2 Turbo -> six still candidates -> human review
                                                     |
                                                     v
                                    WanGP H3 -> video + native audio
```

## Production image recipes

Both `configs/workflow/krea2_turbo.json` and
`configs/workflow/krea2_turbo_img2img.json` now use:

- Stock ComfyUI `UNETLoader` with `Krea2-Turbo-W4A4-noLowRank.safetensors`.
- Native ConvRot W4A4 quantized operations and ComfyUI dynamic VRAM loading.
- `CLIPLoaderGGUF` with `Qwen3VL-4B-Instruct-Q4_K_M.gguf`, type `krea2`.
- `qwen_image_vae.safetensors`.
- 8 steps, Euler/simple, CFG 1, zeroed negative conditioning.
- T2I denoise 1; continuity img2img denoise 0.25.

The W4A4 checkpoint is the community quantization of Krea 2 Turbo. This
installation already has ComfyUI 0.30 and Torch 2.11.0+cu130 with native
`convrot_w4a4` support. No extra node or package was installed. The GGUF
plugin remains necessary for the text encoder; the diffusion model no longer
uses `UnetLoaderGGUF`. There is no automatic fallback to the old diffusion
checkpoint.

Character names and the resolved scene prompt pass through `positive_prompt`
without a hidden appearance description or prompt enhancement. Human image
review remains the quality gate. The img2img recipe is a local continuity
adaptation, rather than an official Krea img2img template.

## Verified model assets

```text
D:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\Krea2-Turbo-W4A4-noLowRank.safetensors
D:\ComfyUI_windows_portable\ComfyUI\models\clip\Qwen3VL-4B-Instruct-Q4_K_M.gguf
D:\ComfyUI_windows_portable\ComfyUI\models\vae\qwen_image_vae.safetensors
```

The diffusion asset is 8,057,520,008 bytes, SHA-256
`f16df7a51f632d0743fb5402d47609ec0572a0fbcda9a4a2bf971f177bde101f`.
The workflow asset manifest uses the same filename, size, hash and source.
All three assets are ready on the D: runtime.

## Cache and video handoff

Successful image jobs retain ComfyUI models between candidates. Do not call
`/free` before or after every image: `free_memory: true` resets ComfyUI node
caches as well as unloading models. Image OOM recovery may explicitly release
the cache and retry the same recipe once.

The WanGP provider checks that ComfyUI is idle, then calls `/free` with
`unload_models: true` and `free_memory: true` once before loading H3. This is
the boundary where image weights must leave the 8 GB GPU. The default balanced
H3 video profile remains 608x352.

## Measured acceptance on RTX 4060 8 GB

The previous GGUF diffusion graph took 211.6–1841.8 seconds per 512x640 image
in the investigated six-candidate batch. A cache-retained GGUF experiment
stopped at the 240-second budget before its first image completed. Subsequent
warm GGUF images were not measured; this does not establish their latency or
rule out a significant cache benefit.

The replacement uses the same production prompt, seed sequence, 8 steps and
text encoder/VAE. Native W4A4 produced:

- 512x640: 15.6 and 18.7 seconds with existing server caches.
- 1024x576 after explicitly clearing ComfyUI models/node caches: 49.3 seconds;
  the next image took 17.5 seconds.
- Production six-candidate tool entry after clearing ComfyUI cache: 107.6 seconds total; each image 9.9–46.9 seconds, six valid 512x640 PNGs, no OOM retry.
- Production continuity-tool entry: 18.9 seconds, valid 512x640 PNG.

These are measured image results, not an all-character quality guarantee or a
full video/publishing rerun. Clearing ComfyUI does not clear Windows file
caches or restart the process. Bandana Waddle Dee's face, body, feet, bandana
and spear passed visual inspection; FLUX.2 Klein was fast but failed this
character contract. Keep human review before video generation.

See [the investigation and reproducible evidence](research/krea2_latency_2026-10-09.md).

## Other image routes

Krea remains the first candidate for image-assisted stages. Existing Anima,
Nova, hybrid Nova/Z-Image and character-specific candidates remain separately
selectable when their assets are ready. Pure `z_image` and
`z_image_i2i_anime` workflows remain retired.

## Primary references

- [Official Krea inference and Turbo settings](https://github.com/krea-ai/krea-2)
- [Official ComfyUI Krea guide](https://docs.comfy.org/tutorials/image/krea/krea-2)
- [Native W4A4 implementation](https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI)
- [W4A4 checkpoint](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)
- [Krea Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE)
