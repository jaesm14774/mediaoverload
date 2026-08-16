# Agentic Runtime

`agentic/` is the media runtime for explicit planners, execution graphs, skills, tools, memories, and an asset registry. It gives AI agents a composable surface for designing and refining workflows without coupling generation logic to a service hierarchy.

## Design Principles
- Goal driven: each run starts from an abstract goal plus constraints, not a hard-coded generation strategy.
- Skill native: work is composed from independent skills registered in a shared registry so they can be reordered, retried, or swapped.
- Tool aware: ComfyUI, media processing, model downloads, and publishing endpoints are exposed as tools instead of hidden behind bespoke services.
- Asset aware: workflows and models ship with manifests so the runtime can reason about missing dependencies and auto-provision them.
- Self-reflective: the runner records state, metrics, and feedback after every node to drive retries and future decisions.

## Layout
- `src/agentic/app/`: CLI entrypoint plus runtime bootstrap helpers.
- `src/agentic/runtime/`: planners, execution graph contracts, runner, registries.
- `src/agentic/skills/`: skill implementations, including agent-facing planning/media primitives and workflow-specific chains.
- `src/agentic/tools/`: built-in tools that wrap ComfyUI, audio, media, and evaluation utilities.
- `src/agentic/assets/`: workflow manifest and asset registry helpers.
- `src/agentic/memory/`: run memory buffers and portfolio/experience storage.
- repo-level `configs/workflow/`: machine-readable ComfyUI workflow templates.
- repo-level `configs/storyboards/`: reusable storyboard contracts and presets.
- `examples/`: runnable goal definitions for smoke tests.

## Quickstart
```bash
cd agentic
python -m pip install -e .
agentic --goal "kirby explores a surreal city" \
        --media-type long_video \
        --duration-seconds 45 \
        --execute
```

The CLI prints the generated plan first, then executes it through the registered skill/tool graph. Production paths use the configured ComfyUI/FFmpeg/TTS adapters, while local adapters support smoke tests and offline development.

For a real local output demo that does not depend on ComfyUI, run:

```bash
agentic --goal "a robot chef preparing noodles in a rainy night market" \
        --media-type storyboard \
        --style "graphic novel travel poster" \
        --execute
```

That workflow writes real `PNG` frames plus summary files into `agentic/output/`.

For a real ComfyUI image generation run:

```bash
agentic --goal "kirby in a neon ramen alley" \
        --media-type image \
        --style "anime key visual" \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188
```

This uses the existing `z_image.json` graph in the repo and writes actual ComfyUI outputs into `agentic/output/`. ComfyUI must already be running and reachable.

For the `text2img2video` chain:

```bash
agentic --goal "kirby jogging through a rainy neon ramen alley at night" \
        --media-type text2img2video \
        --style "anime key visual" \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188
```

That chain composes `text2img -> upscale -> image-to-video -> gif preview` through the shared runtime.

For the real `long_video` chain:

```bash
agentic --goal "kirby explores a surreal city at night" \
        --media-type long_video \
        --duration-seconds 20 \
        --style "anime cinematic travel film" \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188 \
        --comfy-root D:\ComfyUI_windows_portable
```

This chain now uses the new runtime to compose `segment prompt -> recipe-specific anchors/references -> I2V segments -> tail/continuation or transition -> concat -> gif preview`. The selected mix can use first, first+last, last, or reference-bundle conditioning per segment; it is not a single fixed first-keyframe path. Add `--use-tts` if you also want per-segment narration generation and final mux.

Additional runtime primitives are available:

```bash
agentic --goal "refine this portrait" \
        --media-type image_refine \
        --input-image C:\path\to\input.png \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188

agentic --goal "animate this still" \
        --media-type image_to_video \
        --input-image C:\path\to\input.png \
        --text "soft rainy ambience" \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188

agentic --goal "robot chef in a rainy alley" \
        --media-type text2video \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188

agentic --goal "refine character art from scratch" \
        --media-type text2img2img \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188
```

Routing and workflow selection live in the repo-level `configs/routing.yaml` and
the workflow manifests under `configs/workflow/`; there is no separate routing
map required by the runtime entry point.

## Agentic Skill Surface

The runtime is no longer limited to workflow-specific wrappers. It now exposes agent-facing skills that can be recomposed by planners:

- Goal and prompt skills: `agent.goal.expand`, `agent.prompt.compose`, `agent.story.segment`, `agent.segment.prepare`, `agent.sticker.expressions`
- Media skills: `media.ensure_workflow`, `media.image.refine`, `media.image.upscale`, `media.image.animate`
- Audio/video packaging skills: `media.audio.narrate`, `media.audio.concat`, `media.video.concat`, `media.video.merge_audio`, `media.video.gif_preview`, `media.video.extract_last_frame`

Those skills are the default building blocks for composed capabilities instead of hard-coded strategy flow.

## Validation Snapshot

The following paths are covered by the checked-in runtime tests and local-adapter validation:

- `image`
- `image_refine`
- `image_upscale`
- `image_to_video`
- `text2img2video`
- `video_narrate`
- `long_video` (minimal chain without claiming full `--use-tts` validation yet)

The current development order should keep validating small primitives first, then reuse them inside more complex chains such as `long_video`.

## Roadmap Snapshot
The current runtime already covers planner + execution graph + shared ComfyUI
asset tooling + long-video/storyboard skills + prompt/review state + social
publishing tools. Further work should extend those shared contracts and skill
registrations rather than create another workflow-specific orchestration path.
