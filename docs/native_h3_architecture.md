# Native MiniMax H3 story recipe

`native_h3_story` is the first-class MiniMax H3 generation type. It uses the
same automation surface as every other media strategy; generation starts at
`run_media_interface.py` or the scheduler and reaches ComfyUI through the
agentic runtime.

## Current short-video contract

The production Kirby native-H3 recipes use a single 15-second clip with five
causal beats: hook, promise, escalation, reversal, and payoff. The story is
generated from the current news item; the repository preset supplies only the
character/style/continuity contract. Beat boundaries may move slightly, but
they must remain contiguous from 0 to 15 seconds and the final beat must solve
the original objective.

There is also `configs/storyboards/kirby_native_20s.yaml` for experiments and
future continuation workflows. Direct H3 rendering is intentionally capped at
362 frames (~15 seconds): the local ComfyUI H3 node documents 124-362 frames as
the trained range. A 20-second direct request is rejected before submission;
the system does not silently shorten it or substitute a fallback.

`native_h3_t2v_story` is the direct text-to-video route. It calls the existing
`comfy.workflow.text_to_video` tool and has no keyframe/image nodes.
`native_h3_story` is the image-to-video route and uses the existing keyframe
and identity-gate nodes before calling `comfy.workflow.image_to_video`. Both
routes share the same news-grounded story validator and QA/package nodes.

## Runtime path

```text
schedule / run_media_interface.py
  -> agentic.app.character_workflow.run_character_workflow
  -> routing.yaml generation_type + workflow candidates
  -> TaskPlanner._build_native_h3_story_plan
  -> SkillRegistry / WorkflowRunner
  -> ComfyWorkflowToolset
  -> ComfyUI API graph
```

The native graph is:

```text
storyboard prompt
   -> opening keyframe  ─┐
   -> ending keyframe   ─┼-> character identity gate
                        └-> MiniMax H3 first-frame + last-frame I2V
                              -> ffprobe duration/stream QA
                              -> contact sheet + GIF + packaged video
```

The prompt is assembled by `longvideo.prepare_native_h3_story` from the
configured storyboard and the current run's creative brief. The LLM is only
used by the existing route/prompt layer when enabled; the story preset, state
changes, first/last frame contracts, and QA rules remain reproducible and
visible in the plan manifest.

## Kirby configuration

The reusable recipe lives in:

- `configs/characters/kirby.yaml` — profile, storyboard, 608x352, 362 frames,
  16 steps, and workflow names.
- `configs/storyboards/kirby_native_15s.yaml` — one 15-second causal arc with
  hook, escalation, and payoff; it is one H3 clip, not stitched 5-second clips.
- `configs/workflow/minimax_h3_lowvram_15s_fl2va_i2v.json` — visible ComfyUI
  API graph with first and last frame LoadImage bindings.

To make another scheduled story, copy the storyboard preset, keep
`story_spine`, `native_shots`, `opening_keyframe_prompt`, and
`ending_keyframe_prompt`, then point a character or routing override at it.
No generator code changes are required.

## Scheduler setup

Set these values in `media_overload.env` (the existing scheduler already reads
them):

```dotenv
SCHEDULER_CHARACTER=kirby
SCHEDULER_PREFERRED_GENERATION_TYPE=native_h3_story
SCHEDULER_RUN_IMMEDIATELY=false
SCHEDULER_COMFY_HOST=127.0.0.1
SCHEDULER_COMFY_PORT=8188
SCHEDULER_COMFY_ROOT=D:/ComfyUI_windows_portable
```

For a mixed content calendar, leave
`SCHEDULER_PREFERRED_GENERATION_TYPE` empty. The existing router will choose
from the configured generation candidates; Kirby's `native_h3_story` weight is
currently higher than the segmented long-video recipe.

## OpenRouter free model pool

The scheduler uses the checked-in `configs/openrouter_models.yaml` snapshot. It
does not query the OpenRouter catalog during normal runs, so a catalog outage
cannot interrupt a scheduled story. Text and vision have separate fixed pools;
the rotating adapter randomizes the first candidate for every request and then
tries the remaining candidates if the selected route fails.

The current snapshot keeps the largest general-purpose/open multimodal models
that passed repeated MediaOverload JSON smoke tests. The 550B Ultra model is
included: its first weak prompt produced invalid JSON, but the stricter JSON
contract and repair pass made it usable. Models are assigned a request mode in
the YAML file: `structured` sends `response_format`, `prompt_only` relies on
the strict prompt plus the shared parser, and `reasoning_off` disables hidden
reasoning when it otherwise consumes the answer budget.

The Nano 12B VL model is intentionally vision-only in the default snapshot:
its image route is useful, while its text route took about two minutes in the
low-cost healthcheck and is not allowed to slow down story planning.

`AGENTIC_OPENROUTER_DISCOVER_MODELS=true` is an explicit diagnostic opt-in only;
it is not used by the scheduler configuration. Keep
`AGENTIC_OPENROUTER_MAX_*_MODELS_PER_CALL=0` to allow the complete fixed pool to
participate in rotation. The legacy Gemini fallback is disabled by default.

The model pool is checked in as configuration and is used directly by the
runtime. Diagnostics are intentionally kept out of the production generation
path.

## Outputs

Each run stores the generated video plus:

- `native_h3` plan and prompt lineage in the normal agentic run result;
- GIF preview and first/last continuity frame paths.

The native H3 technical QA node is currently an explicit no-op: it returns
`passed: true` after confirming that the render produced a video path. It does
not run ffprobe, duration/audio heuristics, or contact-sheet generation. The
final Discord human review is the authoritative visual/story decision.

## Automated social publishing

Publishing stays on the same goal/plan/skill/tool path. There is no separate
uploader script. Use the formal repo entry point after the ComfyUI server and
the scheduler dependencies are ready:

```powershell
python run_media_interface.py `
  --character kirby `
  --generation-type native_h3_story `
  --comfy-root D:\ComfyUI_windows_portable `
  --publish-mode safe_poc `
  --publish-platform youtube `
  --publish-platform facebook `
  --publish-platform instagram_graph
```

`safe_poc` is deliberately non-public: YouTube uses `private`, Facebook uses
the Reels API with `DRAFT`, and Instagram creates a video container without
calling `media_publish`. The story generation and QA gates still run normally;
if generation, QA, authentication, upload, or container creation fails, the
run is failed and the error is returned per platform. No fixed storyboard or
silent fallback is used.

An expired `IG_GRAPH_ACCESS_TOKEN` is a real credential failure, not a reason
to publish through another Instagram path; refresh the token in the character's
`instagram_graph.env` and rerun the same command.

Use `--publish-mode live` only after the safe run is verified. Instagram video
publishing requires a publicly reachable video URL in the configured Graph API
adapter; a local `D:\` path cannot be fetched by Meta. Facebook Reels are padded
to a 9:16 canvas by the existing FFmpeg adapter when the source is not already
vertical. X is disabled in Kirby's checked-in config because the current X API
uses pay-per-use credits; it is not part of the free safe POC.

## Mandatory six-candidate opening-frame review

Kirby's production `native_h3_story` recipe treats the opening frame as a
hard human gate. The same news-grounded story is generated once, then the
opening image workflow renders six low-cost candidates in one batch. Discord
receives all six attachments in order; the reviewer must select exactly one or
reject the batch. Missing Discord configuration, timeout, API failure, and
reject are blocking outcomes. The runtime never silently selects candidate 1.

After approval, the selected image is immutable: it is passed directly to the
MiniMax H3 I2V workflow without identity img2img refinement or automatic
opening-frame regeneration. The ending keyframe is generated separately, and
the final publish/review plan requires a Discord decision before safe POC
publishing.

The formal command remains the repository entry point:

```powershell
python run_media_interface.py `
  --character kirby `
  --generation-type native_h3_story `
  --comfy-host 127.0.0.1 `
  --comfy-port 8188 `
  --comfy-root D:\ComfyUI_windows_portable `
  --publish-mode safe_poc `
  --publish-platform youtube `
  --publish-platform facebook `
  --publish-platform instagram_graph
```
