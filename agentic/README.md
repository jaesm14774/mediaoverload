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

That workflow writes real `PNG` frames plus summary files into `output/`.

For a real ComfyUI image generation run:

```bash
agentic --goal "kirby in a neon ramen alley" \
        --media-type image \
        --style "anime key visual" \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188
```

This uses the active `krea2_turbo.json` graph in the repo and writes actual ComfyUI outputs into `output/`. ComfyUI must already be running and reachable.

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

This chain now uses the default long-video contract to compose `segment prompt -> planned first/last story-state anchors -> H3 FL2VA segments -> concat -> trim -> technical QA/contact sheet`. The default is four roughly five-second segments for a 20-second output, with pure prompt-only T2V available only through an explicit recipe override. Add `--use-tts` if you also want per-segment narration generation and final mux.

For agent-controlled editing of generated images or video segments:

```bash
agentic --goal "turn these generated shots into a clean vertical reel" \
        --media-type image_sequence_edit \
        --duration-seconds 15 \
        --edit-input C:\\path\\to\\shot-01.mp4 \
        --edit-input C:\\path\\to\\shot-02.mp4 \
        --edit-input C:\\path\\to\\shot-03.mp4 \
        --edit-profile xfade_clean_v1 \
        --edit-transition-duration 0.10 \
        --execute \
        --output-dir C:\\path\\to\\edit-output
```

`image_sequence_edit` is the provider-neutral, OpenCut-inspired timeline surface. It records an ordered `EditPlan`, normalizes images/video to one canvas, adds bounded still-image motion, renders deterministic transitions, preserves/creates a stereo audio lane, emits an edit manifest and contact sheet, then runs technical QA and a GIF preview. Still-only input defaults to `motion_cut_v1`: bounded zoom/pan with clean hard cuts. Video-only input defaults to `baseline_concat`; transition profiles remain explicit opt-ins until they pass the independent repeat gate. Add `--edit-creative-review` to enable a blocking vision-LLM loop (up to four deterministic candidates); each rejected candidate keeps its rendered MP4, contact sheet, join evidence and review JSON, while only the best passing candidate is materialized to the requested output. The loop can change transition grammar and bounded pan/zoom/drift motion for generated video segments, so a static source clip can receive a controlled editorial treatment. If the vision review is unavailable or uncertain, the edit fails closed.

For example, run the creative gate explicitly when a clean default edit is not enough:

```bash
agentic --goal "make the generated shots playful, readable and rhythmically varied" \
        --media-type image_sequence_edit \
        --duration-seconds 15 \
        --edit-input C:\\path\\to\\shot-01.mp4 \
        --edit-input C:\\path\\to\\shot-02.mp4 \
        --edit-input C:\\path\\to\\shot-03.mp4 \
        --edit-input-root C:\\path\\to \
        --edit-creative-review \
        --edit-creative-review-max-attempts 4 \
        --execute \
        --output-dir C:\\path\\to\\edit-output
```

`editorial_kinetic_v1` also enables this review automatically. Technical QA alone is never a creative acceptance gate.

For a declarative JSON plan, use the standalone editor entrypoint:

```bash
agentic-edit --edit-plan C:\\path\\to\\edit-plan.json \
             --output C:\\path\\to\\edit-output\\edited.mp4
```

The plan contract contains ordered `clips`, an empty `transitions` list for hard-cut profiles or one transition per boundary for transition profiles, `output_width`, `output_height`, `fps`, optional `target_duration_seconds`, `profile`, and deterministic `variant_seed`. Input files must be under the repository, configured output root, `AGENTIC_ALLOWED_MEDIA_ROOTS`/`AGENTIC_ALLOWED_IMAGE_ROOTS`, or an explicit `--input-root`. Add `--creative-review --creative-review-max-attempts 4` to the standalone command to run the same blocking loop. Creative-review receipts include the exact candidate plans, LLM backend, scores, issues, next-change recommendation and selected output; no second LLM rewrite or social dispatch is implied.

The same compositor can be inserted into a non-TTS `long_video` plan with `--longvideo-edit-profile xfade_clean_v1`. TTS long-video runs remain on the existing audio-owned packaging route until a separate audio-aware edit contract is validated.

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
- Audio/video packaging skills: `media.audio.narrate`, `media.audio.concat`, `media.video.concat`, `media.video.compose_timeline`, `media.video.merge_audio`, `media.video.gif_preview`, `media.video.extract_last_frame`

Those skills are the default building blocks for composed capabilities instead of hard-coded strategy flow.

## Validation Snapshot

The following paths are covered by the checked-in runtime tests and local-adapter validation:

- `image`
- `image_refine`
- `image_upscale`
- `image_to_video`
- `text2img2video`
- `video_narrate`
- `long_video` (default 20-second planned-anchor path with final technical QA; TTS remains optional)
- `image_sequence_edit` (ordered image/video timeline, deterministic transitions, optional vision-LLM creative loop, manifest/contact sheet, technical QA)

The current development order should keep validating small primitives first, then reuse them inside more complex chains such as `long_video`.

## Roadmap Snapshot
The current runtime already covers planner + execution graph + shared ComfyUI
asset tooling + long-video/storyboard skills + prompt/review state + social
publishing tools. Further work should extend those shared contracts and skill
registrations rather than create another workflow-specific orchestration path.
