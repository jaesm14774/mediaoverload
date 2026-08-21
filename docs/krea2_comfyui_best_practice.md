# Krea 2 Turbo on this machine

## Decision

Krea 2 Turbo is a text-to-image model, not a native text-to-video model. The
official Krea 2 release describes Turbo as an 8-step distilled image
checkpoint. The official Krea video model is Krea Realtime 14B, which has a
different runtime profile and is not a practical local target for this
machine's Windows RTX 4060 8GB setup.

The supported local architecture is therefore:

```text
name-only / user prompt
        |
        v
Krea 2 Turbo T2I -> six reviewable still candidates -> approved first frame
        |
        v
MiniMax H3 I2V -> video + native H3 audio
```

The pure `native_h3_t2v_story` route remains prompt-only H3 T2V. Its optional
image stage is not connected as video conditioning. This keeps the distinction
between a real T2V route and a Krea2-first-frame I2V route explicit.

## Local best practice

The repository uses the local low-VRAM graph in
`configs/workflow/krea2_turbo.json`:

- `UnetLoaderGGUF` with `krea2_turbo_bf16-Q4_0.gguf`
- `CLIPLoaderGGUF` with `Qwen3VL-4B-Instruct-Q4_K_M.gguf` and `type: krea2`
- `qwen_image_vae.safetensors`
- 8 steps, Euler, simple scheduler, CFG 1, denoise 1.0
- `ConditioningZeroOut` for the negative conditioning path
- 1024x576 as the first 16:9 smoke-test size for the 8GB profile
- prompt enhancement intentionally disabled

The repository also contains `configs/workflow/krea2_turbo_img2img.json` for
continuity repair. It uses the same Krea2 model family and denoise 0.25. This
is a local img2img adaptation of the official Krea2 T2I graph, not a claim that
the official Krea2 template is an img2img workflow.

The strict Kirby semantic test must pass the original name and scene prompt
through `positive_prompt` without adding a hidden appearance description. The
default graph placeholder is only a manual example; runtime prompt binding is
the source of truth. The prompt enhancer is not part of the active graph,
because it would make it impossible to tell whether Krea2 understood a proper
character name by itself.

## Model files

Place these files in the local ComfyUI installation before a live render:

```text
D:\ComfyUI_windows_portable\ComfyUI\models\unet\krea2_turbo_bf16-Q4_0.gguf
D:\ComfyUI_windows_portable\ComfyUI\models\clip\Qwen3VL-4B-Instruct-Q4_K_M.gguf
D:\ComfyUI_windows_portable\ComfyUI\models\vae\qwen_image_vae.safetensors
```

The maintained local `ComfyUI-GGUF` fork must be installed because the graph
uses `UnetLoaderGGUF` and `CLIPLoaderGGUF`. The official FP8 ComfyUI template
uses `krea2_turbo_fp8_scaled.safetensors` and the official Qwen3-VL text
encoder; that path is not selected here because its memory requirement is a
poor fit for an 8GB GPU. Q4_0 is the first practical point, not a quality
guarantee: final identity scores must be measured after the first local
render.

Do not jump directly to 1344x768. First validate one 1024x576 image, then
raise resolution only if the model loads and the Kirby identity score remains
acceptable. The H3 hand-off still targets the existing 608x352 balanced-lowvram
video profile.

Before handing the approved frame to H3, call ComfyUI's `/free` endpoint with
`unload_models: true` and `free_memory: true`. Krea2 Q4_0 can occupy most of an
8GB GPU, so keeping its weights cached while H3 loads causes a real text
encoder OOM. The repository's ComfyUI adapter already performs this lifecycle
boundary before and after generation; direct API smoke tests must do it too.

## Retired routes

The active routing candidates now put `krea2_turbo` first for every stage that
needs a text-to-image opening/reference, while retaining Anima, Nova, Nova +
Z-Image Anime, Z-Image + Nova, and the Kirby-specific image workflows as
additional candidates when their assets are available. The pure `z_image.json`
and `z_image_i2i_anime.json` workflows are retired; the hybrid Nova/Z-Image
workflows are not pure Z-Image and remain available.

## Verification boundary

The local ComfyUI server and the Krea2 node classes were inspected, and the
three Krea2 assets are now installed and verified on the D: ComfyUI runtime.
Strict live tests already produced valid Kirby and Waddle Dee images, and a
Krea2-to-MiniMax H3 I2V smoke test completed. The remaining comparison gate is
to render each retained image candidate only when its own assets are present:

1. confirm the candidate's model files appear in ComfyUI's model dropdowns;
2. render the candidate at 1024x576 with the exact prompt `Kirby` plus a
   simple scene phrase;
3. score the output against a correct Kirby reference;
4. only after passing the image gate, pass the reviewed image to
   `minimax_h3_lowvram_i2v`.

## Primary references

- [Krea 2 official repository](https://github.com/krea-ai/krea-2)
- [Krea 2 Turbo model card](https://huggingface.co/krea/Krea-2-Turbo)
- [Official ComfyUI Krea 2 guide](https://docs.comfy.org/tutorials/image/krea/krea-2)
- [Official Krea 2 Turbo ComfyUI template](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_krea2_turbo_t2i.json)
- [Krea 2 prompting guide](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md)
- [Krea Realtime 14B repository](https://github.com/krea-ai/realtime-video)
- [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE)
