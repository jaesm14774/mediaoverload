> H3 generation backend: **Powered by WanGP**. See [current workflows and runtime contracts](wan2gp_h3.md).

# Native MiniMax H3 story recipe

The five Native H3 modes are first-class MiniMax H3 generation types:
`native_h3_story`, `native_h3_t2v_story`, `native_h3_fl2va_story`,
`native_h3_l2va_story`, and `native_h3_ref2va`. The composed
`text2image2native_h3_ref2va` route adds an explicit candidate-image stage.
They use the same automation surface as every other media strategy; generation starts at
`run_media_interface.py` or the scheduler and reaches WanGP through the
agentic runtime.

## Native H3 generation

The production Kirby native-H3 recipes use a single 15-second clip. The preset
supplies renderer settings and a character reference; the LLM follows the
current user prompt and selected style. Shot count and creative fields are
optional, and the application assigns render timing to any returned shots.
News context is retained for factual grounding and traceability. Post-render
inspection records technical evidence for Discord, which owns creative review.

The production route uses `configs/storyboards/native_h3_15s.yaml` for its
duration and renderer settings. Direct H3 rendering is capped
at 362 frames (~15 seconds): the configured WanGP H3 routes support up to 362 delivered frames.
A 20-second direct request is rejected before
submission; the system does not silently shorten it or substitute a fallback.

`native_h3_t2v_story` is the direct text-to-video route. It calls the existing
`wan2gp.render_h3` tool and has no keyframe/image nodes.
`native_h3_story` is the first-frame image-to-video route; `native_h3_fl2va_story`
adds a reviewed landing frame, `native_h3_l2va_story` keeps only the reviewed
landing frame, and `native_h3_ref2va` consumes either a validated manifest or
the six-candidate Discord reference review when its manifest is empty. All modes
share the same minimal render-contract normalization and inspection/package nodes.

## Runtime path

```text
schedule / run_media_interface.py
  -> agentic.app.character_workflow.run_character_workflow
  -> routing.yaml generation_type + workflow candidates
  -> TaskPlanner dispatches the selected Native H3 route
  -> SkillRegistry / WorkflowRunner
  -> Wan2GPToolset / wan2gp.render_h3
  -> isolated worker / official WanGP Python API
```

The native graph is:

```text
selected Native H3 route
   -> T2V: prompt -> MiniMax H3 T2V
   -> I2VA: six opening candidates -> Discord -> first-frame I2V
   -> FL2VA: opening + landing candidates -> Discord -> first+last-frame I2V
   -> L2VA: six landing candidates -> Discord -> last-frame I2V
   -> Ref2VA: valid manifest OR six T2I candidates -> Discord -> Ref2VA
   -> media inspection evidence for Discord
   -> contact sheet + GIF + packaged video
```

`longvideo.prepare_native_h3_story` delegates news selection and LLM story
generation to `agentic/src/agentic/runtime/story_service.py`, then formats the
resolved prompt for rendering. Optional story fields and `news_trace` remain
available in the plan manifest. The publish stage receives a compact story/news
context rather than the full production prompt.

## Publishable 30/45-second story assembly

Use the `text2longvideo` route for a long story. The checked-in
`text2longvideo` profile turns 30 seconds into roughly six
five-second H3 clips, or 45 seconds into roughly nine clips. Segment prompts
follow the current user brief and do not require a fixed number of narrative
beats. I2V carries the actual rendered tail into the next clip; the selected
conditioning workflow handles frame transitions. If approved
reference images or videos are configured, only the opening clip uses Ref2VA
for optional visual references, after which the real tail continues the story.

This is the production route currently selected for the local RTX 4060. Each
segment starts from the previous rendered tail, and the selected conditioning
workflow is used before the final package is assembled.

The route extracts a tail frame and runs technical QA for every segment, then
writes `publish_ready_longvideo.json` beside the normal long-video summary.
Subjective story quality remains a human review decision; a live social post
is complete only when the platform receipt confirms the requested state.

Example for a 30-second publish candidate:

```powershell
python run_media_interface.py `
  --character kirby `
  --generation-type text2longvideo `
  --duration-seconds 30 `
  --prompt "Kirby follows a glowing seed through a windy meadow, loses it to the storm, teams up with Waddle Dee to rescue it, and plants it in a warm clearing where it blooms" `
  --publish-mode safe_poc `
  --publish-platform youtube `
  --publish-platform facebook `
  --publish-platform instagram_graph
```

Change only `--duration-seconds` to `45` for the nine-segment version. The
Kirby short-form 2x speed transform is disabled for this profile, so the
requested long duration is preserved. `safe_poc` first produces private,
draft, or container-only platform artifacts; use `live` only after final
approval and the public URL/credentials required by each platform are ready.

## Kirby configuration

The reusable recipe lives in:

- `configs/characters/kirby.yaml` — profile, render recipe, 608x352, 362 frames,
  16 steps, and workflow names.
- `configs/storyboards/native_h3_15s.yaml` — native H3 duration and story input
  settings; the current prompt supplies the creative direction.
- `configs/workflow/wan2gp_h3_fl2va.json` — WanGP workflow metadata;
  the renderer maps first and last images to `image_start` / `image_end` with `SE`.

To make another scheduled story, set its character profile, renderer settings,
and prompt in the character or routing configuration.

## Scheduler setup

Set these values in `media_overload.env` (the existing scheduler already reads
them):

```dotenv
SCHEDULER_CHARACTER=kirby
SCHEDULER_PREFERRED_GENERATION_TYPE=
SCHEDULER_RUN_IMMEDIATELY=false
SCHEDULER_COMFY_HOST=127.0.0.1
SCHEDULER_COMFY_PORT=8188
SCHEDULER_COMFY_ROOT=D:/ComfyUI_windows_portable
```

Leave `SCHEDULER_PREFERRED_GENERATION_TYPE` empty for the mixed content
calendar. The scheduler passes an RNG into the character workflow. The runtime
first samples the configured `character.group_name` from the active weighted
rows in `anime.anime_roles`, then samples the generation strategy; the LLM
writes the story for the already-selected character and strategy.

## Group character selection

The checked-in `configs/characters/kirby.yaml` uses `group_name: Kirby`. The
current CLI and scheduler both use this same configuration, so no separate
group command is required:

```powershell
python run_media_interface.py --character kirby --generation-type text2img --no-review --no-publish
```

At the start of each run, `CharacterGroupSelectionService` queries
`anime.anime_roles` with the current schema and keeps only rows where
`group_name` matches, `status = 1`, `weight > 0`, and `role_name_en` is not
empty. It performs one weighted random choice using the workflow RNG. The
selected role name, role description, keywords, full eligible candidate list,
weights, and selection source are passed into the prompt/storyboard contract.

The evidence is available in both `run_manifest.json` as `character_selection`
and `events.jsonl` as `character.group.selected`. A missing MySQL configuration
or an empty eligible set is a hard selection failure recorded as
`character.group.selection_failed`; the runtime does not silently fall back to
the configured base name.

## OpenRouter free model pool

The scheduler uses the checked-in `configs/openrouter_models.yaml` snapshot. It
does not query the OpenRouter catalog during normal runs, so a catalog outage
cannot interrupt a scheduled story. Text and vision have separate fixed pools;
the rotating adapter randomizes the first candidate for every request and then
tries the remaining candidates if the selected route fails.

The current snapshot keeps the largest general-purpose/open multimodal models
that passed repeated MediaOverload JSON smoke tests. Models are assigned a
request mode in the YAML file: `structured` sends `response_format`,
`prompt_only` relies on the prompt plus the shared parser, and `reasoning_off`
disables hidden reasoning when it otherwise consumes the answer budget. Native
H3 story generation explicitly uses prompt-only JSON for every model, even if
the catalog entry says `structured`; this avoids making a free-plan provider
implement the old nested response schema. The application still normalizes and
checks the small render/safety contract locally.

The Nano 12B VL model is intentionally vision-only in the default snapshot:
its image route is useful, while its text route took about two minutes in the
low-cost healthcheck and is not allowed to slow down story planning.

`AGENTIC_OPENROUTER_DISCOVER_MODELS=true` is an explicit diagnostic opt-in only;
it is not used by the scheduler configuration. Keep
`AGENTIC_OPENROUTER_MAX_*_MODELS_PER_CALL=0` to allow the complete fixed pool to
participate in rotation. Gemini rotation is disabled by default.

The model pool is checked in as configuration and is used directly by the
runtime. Diagnostics are intentionally kept out of the production generation
path.

## Outputs

Each run stores the generated video plus:

- `native_h3` plan and prompt lineage in the normal agentic run result;
- GIF preview and first/last continuity frame paths.

The canonical debug record is `logs/runs/<run_id>/`. It contains
`lifecycle.log`, `events.jsonl`, `llm/*.json` with every request/response and
repair attempt, `nodes/*.json` with node outputs, workflow result JSON, and
`run_manifest.json`. `logs/agentic_portfolio.jsonl` remains a compact
cross-run memory and is not the source of truth for prompt debugging.

The native H3 inspection node calls `media.video_qa` only to create a contact
sheet and record observable media metadata for Discord. Its result is not an
automatic DQ gate: duration, dimensions, frame rate, audio, and subject counts
never filter or block a candidate before or after human review. Missing files,
failed rendering, and tool failures remain real system errors.

No machine gate judges story, rhythm, identity, cuteness, composition, or
prompt similarity. Discord owns those decisions. Approved media and captions
are reused without a second editorial gate.

## Automated social publishing

Publishing stays on the same goal/plan/skill/tool path. There is no separate
uploader script. Use the formal repo entry point after the ComfyUI image server,
isolated WanGP installation and scheduler dependencies are ready:

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
receives the usable attachments in order; the reviewer must select exactly one
or reject the batch. The review asks for a visible cute hit or expression,
motion already starting in the first second, one Kirby, one simple prop, and a
single readable gag. Static portal gazes, abstract lore, duplicate characters,
and multi-character conflict are reject signals. The displayed Asset range is
derived from the attachments actually delivered, not hard-coded to six.
Missing Discord configuration, timeout, API failure, and reject are blocking
outcomes. The runtime never silently selects candidate 1.

After approval, the selected image is immutable: it is passed directly to the
MiniMax H3 I2V workflow without identity img2img refinement or automatic
opening-frame regeneration. The ending keyframe is generated only when
`use_last_frame: true`; otherwise I2VA submits only the opening image with `S`.
The final publish/review plan requires a Discord decision before safe
POC publishing.

`--no-review` is an explicit bypass: it does not send an interactive Discord
review message and selects the configured single opening candidate
automatically. This mode still runs deterministic media QA. When publishing is
attempted, the runtime sends a separate concise Discord run-status notification
and records the notification receipt; interactive review delivery records the
channel, message, attachment count, and decision session separately.

Publication state is based on verified platform receipts rather than HTTP
success alone. A run can therefore be `partially_published` or `staged` when
one platform is public but another is private, draft, or failed. The checked-in
YouTube credentials currently request `private`, so a live run is not treated
as publicly complete until the YouTube receipt reports public visibility.

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
