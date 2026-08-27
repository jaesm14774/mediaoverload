# Native MiniMax H3 story recipe

The five Native H3 modes are first-class MiniMax H3 generation types:
`native_h3_story`, `native_h3_t2v_story`, `native_h3_fl2va_story`,
`native_h3_l2va_story`, and `native_h3_ref2va`. The composed
`text2image2native_h3_ref2va` route adds an explicit candidate-image stage.
They use the same automation surface as every other media strategy; generation starts at
`run_media_interface.py` or the scheduler and reaches ComfyUI through the
agentic runtime.

## Current short-video contract

The production Kirby native-H3 recipes use a single 15-second clip with three
causal beats: hook, escalation, and payoff. The repository preset supplies the
character, identity, safety, continuity, and timing contract; the LLM supplies
the current story. Creative metadata such as `gag_card`, `story_spine`, and
`news_trace` is optional and is retained as observability/context when present.
The pre-render contract only requires a JSON story with the expected number of
shots, visible actions, contiguous timing, and safe visual prompt text. Missing
titles, camera directions, state changes, keyframes, or audio are filled from
the generated actions and the stable preset before prompt composition.

News grounding and story quality are recorded as advisory scores at this stage,
not as free-model generation gates. Beat boundaries must remain contiguous from
0 to 15 seconds; post-render technical QA and any recipe-enabled semantic QA
remain the authoritative checks for the actual media.

The production route uses `configs/storyboards/kirby_native_15s.yaml` as its
identity and continuity contract. Direct H3 rendering is intentionally capped
at 362 frames (~15 seconds): the local ComfyUI H3 node documents 124-362 frames
as the trained range. A 20-second direct request is rejected before
submission; the system does not silently shorten it or substitute a fallback.

`native_h3_t2v_story` is the direct text-to-video route. It calls the existing
`comfy.workflow.text_to_video` tool and has no keyframe/image nodes.
`native_h3_story` is the first-frame image-to-video route; `native_h3_fl2va_story`
adds a reviewed landing frame, `native_h3_l2va_story` keeps only the reviewed
landing frame, and `native_h3_ref2va` consumes either a validated manifest or
the six-candidate Discord reference gate when its manifest is empty. All modes
share the same minimal render-contract normalization and QA/package nodes.

## Runtime path

```text
schedule / run_media_interface.py
  -> agentic.app.character_workflow.run_character_workflow
  -> routing.yaml generation_type + workflow candidates
  -> TaskPlanner dispatches the selected Native H3 route
  -> SkillRegistry / WorkflowRunner
  -> ComfyWorkflowToolset
  -> ComfyUI API graph
```

The native graph is:

```text
selected Native H3 route
   -> T2V: prompt -> MiniMax H3 T2V
   -> I2VA: six opening candidates -> Discord -> first-frame I2V
   -> FL2VA: opening + landing candidates -> Discord -> first+last-frame I2V
   -> L2VA: six landing candidates -> Discord -> last-frame I2V
   -> Ref2VA: valid manifest OR six T2I candidates -> Discord -> Ref2VA
   -> shared technical QA + sampled-frame semantic QA
   -> contact sheet + GIF + packaged video
```

`longvideo.prepare_native_h3_story` delegates news selection and LLM story
generation to `agentic/src/agentic/runtime/story_service.py`, then formats the
resolved storyboard into the render prompt. The storyboard rules, `news_trace`,
state changes, first/last frame contracts, and QA rules remain reproducible and
visible in the plan manifest. The publish stage receives a compact story/news
context rather than the full production prompt.

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
implement the old nested semantic schema. The application still normalizes and
checks the small render/safety contract locally.

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

The canonical debug record is `agentic/logs/runs/<run_id>/`. It contains
`lifecycle.log`, `events.jsonl`, `llm/*.json` with every request/response and
repair attempt, `nodes/*.json` with node outputs, workflow result JSON, and
`run_manifest.json`. `agentic/logs/agentic_portfolio.jsonl` remains a compact
cross-run memory and is not the source of truth for prompt debugging.

The native H3 QA node delegates technical checks to the shared
`media.video_qa` tool. It records file/stream/dimension/duration checks, audio
warnings, and a duration-aware contact sheet in the run directory. When the
recipe enables `semantic_qa_required`, the shared Prompt Engine sends that
contact sheet to the configured vision model and records the full request and
response under `agentic/logs/runs/<run_id>/llm/`. The semantic result is
advisory evidence for human review and never blocks the run. Technical media QA
remains the hard pre-publication check, and Discord approval remains the
authority for subjective story and visual quality.

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
`use_last_frame: true`; otherwise the Comfy last-frame input is explicitly
cleared. The final publish/review plan requires a Discord decision before safe
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
