# Krea 2 Turbo on this machine

## Supported route

Krea 2 Turbo creates still images. Image-assisted video routes use Krea to
create six reviewable candidates, then pass the approved image to ComfyUI
MiniMax H3. The prompt-only H3 T2VA route remains separate.

```text
user prompt -> Krea 2 Turbo -> six still candidates -> human review
                                                     |
                                                     v
                                    ComfyUI H3 -> video + native audio
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

The W4A4 checkpoint is the community quantization of Krea 2 Turbo. The
ComfyUI installation must support native `convrot_w4a4` operations. Verify its
installed ComfyUI/Torch versions and GPU support separately. The GGUF
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
The manifest provides these expected assets; validate the current installation before generation.

## Cache and video handoff

Successful image jobs retain ComfyUI models between candidates. Do not call
`/free` before or after every image: `free_memory: true` resets ComfyUI node
caches as well as unloading models. Image OOM recovery may explicitly release
the cache and retry the same recipe once.

H3 uses the same ComfyUI queue and model manager as image generation. ComfyUI
loads the video workflow's weights and releases other models as needed;
there is no second renderer or cross-process GPU handoff. The default balanced
H3 video profile remains 608x352.

## Latency diagnosis

Measure the current workload from ComfyUI queue state, image jobs, model loading and run logs. Candidate count, cache reuse, CPU offload and GPU contention are separate causes. A previous single-image timing does not predict a new six-candidate run.

Compare the same canvas, steps, prompt, conditioning and delivered outputs. Keep human review of identity and visual quality; a fast model is only useful when it meets those contracts.

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
