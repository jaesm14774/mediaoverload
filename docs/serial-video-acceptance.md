# Serial video acceptance and implementation plan

操作入口、MySQL 表關係與可複製 CLI 範例見 [serial-video.md](serial-video.md)。

## Acceptance scenarios

### User Given eligible old series and default 0.5/0.5 weights When serial auto-next selects a mode Then it starts new-series onboarding for a roll below 0.5 and continues an eligible series for a roll at or above 0.5

### User Given no eligible old series When serial auto-next selects a mode Then it starts new-series story onboarding with 100% probability regardless of configured weights

### User Given a character profile sets series-selection weights When serial auto-next runs with eligible old series Then it uses that profile's configured new and continue proportions

### User Given a new series is selected When its story, character reference, and episode outline reach human review Then each review happens in Discord and an accepted decision continues the same serial run through video generation

### User Given a creator starts serial video with a selected character When the series story is drafted Then that selected character is the protagonist and directly drives the story

### User Given no complexity is specified When the series story and episode outline are drafted Then the result is a simple short drama with one clear goal, one main obstacle, a small cast, and an easy-to-follow resolution

### User Given the creator selects cinematic story complexity When the series story and episode outline are drafted Then the model may use layered stakes and character arcs while keeping the same selected protagonist

### User Given Discord review is unavailable or times out When serial onboarding reaches a review gate Then the run blocks without accepting the draft or starting video generation

### User Given a new serial story draft with the default inspiration mode When it is generated Then one safe traceable news article may inspire one small fictional problem and its source details are saved beside the treatment for human review

### User Given random news is unavailable When a new story draft starts Then the request stops with an actionable message and does not silently generate an ungrounded story

### User Given a custom premise and explicit news opt-out When a story draft starts Then the LLM uses the premise without news context and the saved inspiration mode is none

### User Given a creator edited the saved story Markdown When they accept that plan Then the exact reviewed text, reviewer decision, and approved SHA are saved while the production series remains unactivated until character approval

### User Given a story treatment is requested When the text model generates it Then MediaOverload keeps the readable text directly and does not request JSON schema or structured-output validation

### Later episodes preserve their new story action through H3 prompt preparation

User Given a later-episode segment plan requires a new visible event and its Ref2VA opening image depicts the prior episode's ending, When the vision prompt writer returns a scene focused only on the old pose or prop, Then the final H3 prompt still gives the current segment's action and visible outcome priority and treats the old image as an opening continuity point rather than the target composition.

### A creator can inspect the serial ledger and trace each delivered episode

User Given a serial series has run through multiple episodes, When the creator opens the dedicated MySQL database or follows the documented read-only queries, Then they can see episode versions, target duration, planned segments, render attempts, QA and human decisions, reference assets, publication receipts, final paths, and hashes linked by their IDs.

### Serial records use the dedicated MySQL database

User Given the configured MySQL service is available, When serial mode starts, Then it writes all series, episode, plan, segment, render, review, reference, and publication records to the dedicated `mediaoverload_serial` MySQL database and does not create or read a SQLite serial ledger.

### Final episode acceptance dispatches the approved public copy

User Given an episode has passed final video QA and the creator has reviewed its caption and hashtags in Discord, When the creator accepts the final MP4, Then MediaOverload dispatches that approved copy to the enabled platforms and stores a receipt or failure outcome for each platform in MySQL.

### Final QA records the ready episode before publication review

User Given every segment has current technical QA, human acceptance, and valid tail lineage, When final video QA passes, Then MySQL stores the assembled MP4 path and SHA and marks the episode ready before Discord publication review begins.

### Approved serial copy stays free of forced episode labels

User Given the creator approves a serial episode caption and hashtags in Discord, When MediaOverload prepares platform captions, Then it dispatches the approved copy without adding an episode number, topic heading, or workflow metadata.

### Final publication review contains only publishable copy

User Given a serial episode is ready for publication, When MediaOverload sends its final review to Discord, Then the review card contains the caption and hashtags that will be published, with no separate episode, topic, strategy, or workflow description.

### Episode duration controls segment count and final runtime

User Given the creator requests a total duration longer than ten seconds, When MediaOverload plans the episode, Then it retains the requested total duration, divides the story into H3-sized segments, and records both the total target duration and each segment duration.

### Ordinary daily generation stays on its current route

User Given the normal daily workflow is invoked without serial mode or a series ID, When MediaOverload builds and runs its plan, Then it uses the existing strategy selection, review policy, and output contract without reading or writing serial state.

### Serial video can be selected by normal weighted routing

User Given serial video is enabled for the character and has a positive generation weight, When normal strategy routing selects `serial_video`, Then the existing serial runner generates and reviews the episode instead of starting the ordinary media workflow.

### Serial mode remains explicitly selectable

User Given a creator runs `python run_media_interface.py --character kirby --generation-type serial_video`, When the Kirby YAML enables serial video, Then the existing serial runner starts with its configured defaults.

### A new series records fixed identity references

User Given a new series ID, a fixed character, and one or more readable identity images, When the creator starts its first episode, Then MediaOverload persists the series and reference paths with SHA-256 values and plans only rendered video segments.

### Registered identity references are available to H3 prompt preparation

User Given serial mode registers an identity image outside the output directory, When the H3 prompt is prepared, Then vision validation can read that image while retaining all caller-configured allowed roots.

### Same-episode continuation uses the approved rendered tail

User Given a segment has passed technical QA and its human review accepted a specific MP4 SHA, When the next segment is rendered, Then its I2VA first-frame input is the extracted final frame of that exact MP4 version.

User Given an H3 segment was duration-normalized and accepted before a later review timed out, When the episode resumes, Then the original render node reuses its generated MP4 and the trim node reuses the exact accepted normalized MP4 so final assembly keeps the accepted prefix.

### Duration-normalized clips retain their reviewed lineage

User Given H3 renders a longer clip than the segment duration and MediaOverload trims it, When QA, human review, and tail extraction run, Then all three use the trimmed MP4 and the ledger retains the original render as its source.

### A new episode uses Ref2VA without locking the prior composition

User Given the previous episode has a current accepted final segment and the series identity references still match their recorded SHA values, When the next episode starts, Then its first Ref2VA render receives the prior episode's accepted ending frame and the fixed character references as a reference bundle, while later segments use I2VA tails.

### Review, QA, and asset changes fail closed

User Given a human review rejects or times out, technical QA fails, an input asset changes, or a reviewed video no longer matches its recorded SHA, When continuation or assembly is requested, Then no tail is accepted, no downstream segment is rendered from that state, and no final episode is marked ready.

User Given the final assembled MP4 passed Discord review, When its bytes change before social dispatch, Then the runtime blocks dispatch and records the publication attempt as failed.

### Segment review timeout is an explicit operational limit

User Given a creator configures a Discord review timeout, When a segment review takes longer than that limit, Then the run records a blocked review and does not continue from the unapproved video.

User Given a segment review timed out and the creator later accepts that exact QA-passed MP4, When the creator resumes the same episode, Then the accepted render and its recorded story plan are reused, its actual tail frame is extracted, and only unfinished segments render.

User Given Discord review timed out, When the creator records an accepted or rejected decision for a specific current MP4 path, Then the ledger binds that decision to the exact QA-passed video SHA before any resume or rerender can proceed.

User Given the serial story planner completes, When its segment plan is stored, Then the exact beats used by the first render remain available to every resumed segment and later episode.

### Regeneration invalidates downstream lineage

User Given an earlier segment is rendered again, When its active render version changes, Then its previous review is stale and every later segment and episode conditioned on that version becomes invalid until regenerated and reviewed again.

### Delivery contains only accepted dynamic clips

User Given every planned segment has a current accepted MP4 and valid lineage, When the episode is assembled, Then the delivered file is a playable video assembled from those MP4s and its stored SHA matches the delivered artifact.

### Final episode QA gates the processed output without rewriting segment QA

User Given the planned segments have their own accepted render versions, When final QA inspects the assembled and postprocessed episode, Then it gates episode delivery while leaving every segment review bound to its own MP4 SHA.

### News is optional inspiration

User Given a serial story prompt has no relevant news context, When its episode and segment plans are generated, Then story planning proceeds from the user's series context without requiring a news selection.

### Automatic episode selection balances a fresh series and a continuation

User Given at least one active series has a ready latest episode, When the MediaOverload automatic-next command makes its 50/50 choice, Then one branch starts a fresh series and the other randomly selects an eligible series and creates its next episode from that series' latest approved ending.

### The existing media interface selects serial generation from character YAML

User Given the creator selects `serial_video` in `run_media_interface.py` and the character YAML enables the serial strategy, When the interface runs, Then it reads duration, style, output directory, identity references, review timeout, enabled platforms, and episode count from that YAML and enters the existing serial runner without starting the ordinary character workflow.

### A failed episode resumes before selecting another automatic episode

User Given automatic serial generation has a current blocked episode with a persisted plan and segments, When the creator starts the serial strategy again, Then MediaOverload resumes that same episode before applying the 50/50 fresh-or-continuation choice and reuses its accepted segment prefix.

### A first-segment failure can restart from the first segment

User Given the current episode failed before any segment was accepted but its plan and segment records are persisted, When automatic serial generation retries, Then it reruns the saved first segment without requiring a previously accepted segment.

### A Discord failure stops a multi-episode interface run

User Given the creator asks the serial strategy to run multiple episodes, When a Discord review fails or times out, Then the current episode is recorded as blocked, the interface stops before starting another episode, and the next invocation resumes from that failed episode's accepted checkpoint.

### Completed and discarded series leave the continuation pool

User Given the creator marks a series as completed or discarded after reviewing an episode, When the next automatic selection runs, Then its lifecycle status is saved in MySQL and the series is not eligible for continuation.

User Given a series is completed or discarded, When the creator explicitly requests episode 1 again, Then serial generation rejects the request and leaves the terminal series closed.

### Continuing remains the default episode outcome

User Given a final episode review is open, When the creator accepts the episode without choosing a terminal series action, Then the episode can publish and the series remains active for continuation.

### Discarding a series does not publish its episode

User Given a creator judges the generated episode too boring to keep, When they choose Discard series on the final review, Then MySQL marks the series discarded and the workflow does not dispatch the episode to any platform.

User Given a creator discarded a series, When an operator later requests publication for one of its episodes, Then the runtime refuses to open a new publication attempt for that series.

### Repeated automatic runs stop at an incomplete delivery

User Given the creator asks the existing media interface to generate multiple episodes, When a run times out, fails, or lacks a reported-success receipt from any enabled platform, Then the interface exits before starting another episode.

User Given a publication attempt has a reported-success receipt for one platform and failed or missing receipts for others, When the creator retries the episode, Then the next review and dispatch include only platforms without any reported-success receipt for that episode.

## Repository findings and decisions

- The serial workflow is based on `32f9073af6c53b0860181e29860ed92811703214`. This review started with the serial implementation in the shared worktree; those changes were preserved, reviewed, and included in the feature branch.
- The configured MySQL service is the formal persistence layer for serial-series records. Use the existing PyMySQL dependency and the dedicated `mediaoverload_serial` database; leave the Anime and crawler schemas untouched. Runtime media and intermediate artifacts remain under the configured output root.
- The formal continuation route is `TaskPlanner` → `build_long_video` → `LongVideoSkills.render_segment_video` → `comfy.render_h3`. Existing workflow manifests define `reference_bundle`/Ref2VA and `anchor_first`/I2VA, and the H3 tool builds the corresponding ComfyUI graph and records conditioning files. Serial review nodes run `media.video_qa`, mandatory Discord segment acceptance, and `media.extract_last_frame` before later segments.
- Serial planning will require `serial_mode=true`, a nonempty `series_id`, an explicit fixed character and saved identity references. It will require every-segment human review, disable probe auto-selection, and choose Ref2VA for the episode's first segment followed only by I2VA. Normal generation keeps the existing planner behavior when serial mode is absent.
- Persist series, episode versions and target durations, planned segments, render attempts, QA/review records, reference assets, and per-platform publication outcomes in normalized MySQL tables with foreign keys. Bind reviews and references to SHA-256; validate hashes before using a reference, extracting a tail, continuing, assembling, or publishing. A changed active render invalidates later segments and any episode using its tail.
- Keep story state as structured JSON derived from the previous accepted episode's planned beats and ending. Pass it as optional creative context to the existing long-video story planner. The serial route does not require news.
- Assemble through the existing long-video video editor only after per-segment gates; persist the final path and SHA only after checking its lineage and playable video stream.

## Reference repository review

1. [ai-comic-factory](https://github.com/jbilcke-hf/ai-comic-factory) separates LLM story creation and image rendering behind configurable providers. Its README says the project is not a single self-contained deployment, and GitHub marks it archived. Borrow the stage separation as a planning idea; do not adopt its provider stack or treat its comic outputs as video evidence.
2. [AI-Comic-Generator](https://github.com/Dapeng960208/AI-Comic-Generator) uses structured JSON for story, characters, and storyboard, project-scoped persistence, character references, and previous-panel context. Borrow explicit structured planning and fixed identity references. Its FastAPI/Vue/SQLite/SQLModel image application does not fit the existing Python workflow runner or H3 rendering boundary.
3. [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) combines long-range image consistency with a separate motion predictor and describes a two-stage video path. Its model is not the configured H3 ComfyUI workflow. Do not transplant it or infer H3 continuity quality from its repository examples.
4. [AI-Comic-Crafter](https://github.com/Balajee2k/AI-Comic-Crafter) describes curated styles, session memory, character references, and targeted panel edits. Borrow the idea of storing stable visual identity separately from per-scene prompts; do not adopt its Streamlit/remote image services or panel-rendering architecture.
5. [comic-maker](https://github.com/AskAillex/comic-maker) has a clear local story → editable JSON → ComfyUI image panels → assembly sequence and repeats character descriptions. Borrow the explicit staged workflow and inspectable plan. Its image panels, seed-family strategy, and CBZ assembly do not satisfy video continuity.
6. [Comic_Studio](https://github.com/CallmeTruong/Comic_Studio) uses staged planning, storyboard validation, rendered-page vision QA, and bounded retries. Borrow validation and QA as separate gates. Do not add LangGraph or use comic-page review as evidence for rendered video acceptance.
7. [MangaNinjia](https://github.com/ali-vilab/MangaNinjia) is a reference-following line-art colorization implementation with a non-commercial license. It is image-specific and not a source of video continuation evidence; do not integrate its model or code.

These repositories inform planning and reference handling only. Continuity claims require an actual ComfyUI/GPU render and visual inspection of the resulting videos.

## Running a serial series

Start a series with a fixed character image and a stable output directory:

```powershell
$env:PYTHONPATH = "agentic/src"
python -m agentic.app.main `
  --goal "Kirby finds a fallen star seed in a quiet garden, plants it, and protects it as a tiny star flower blooms." `
  --media-type long_video `
  --duration-seconds 30 `
  --style "watercolor storybook animation" `
  --serial-mode `
  --series-id kirby-continuity `
  --character Kirby `
  --serial-character-reference "D:\ComfyUI_windows_portable\ComfyUI\input\krea2_kirby_identity.png" `
  --output-dir "agentic/output/kirby-continuity" `
  --execute
```

Use the same `--output-dir` and MySQL database for later episodes so the series ledger stays linked. The character reference can be omitted after the first episode; if supplied again, its path and SHA-256 must still match. Pass `--serial-episode-number` to re-render an existing episode number, which invalidates that episode's later lineage. To resume after a review timeout, inspect the generated MP4 and record the exact decision and path, then rerun the original command with `--serial-resume-episode-number` instead of `--serial-episode-number`:

```powershell
python -m agentic.app.main `
  --goal "Kirby finds a fallen star seed in a quiet garden, plants it, and protects it as a tiny star flower blooms." `
  --media-type long_video `
  --duration-seconds 30 `
  --style "watercolor storybook animation" `
  --serial-mode `
  --series-id kirby-continuity `
  --character Kirby `
  --serial-resume-episode-number 1 `
  --output-dir "agentic/output/kirby-continuity" `
  --execute
```

The resume command reuses the persisted plan and accepted MP4s. If a review timed out, record a decision first with `--serial-review-episode-number`, `--serial-review-segment-number`, `--serial-review-decision accepted|rejected`, and `--serial-reviewed-video <exact MP4 path>`. Accepted reviews must identify the same file bytes shown to the creator. A rejected segment stays blocked and is rendered as a new attempt when the creator resumes; previously accepted prefix segments remain bound to their approved MP4s. Use `--serial-episode-number` when the creator wants a full new episode version, which invalidates that episode's later lineage.

## Validation boundary

Local tests can prove the CLI contract, MySQL state transitions, lineage invalidation, planner workflow selection, fail-closed review behavior, and video-only assembly contract. They cannot prove H3 visual continuity or that social platforms accepted a publication. Those require a fresh ComfyUI/GPU run, inspection of generated clips and transition frames, a recorded human review decision, and provider delivery receipts.
# User Given an approved series story When a character card is drafted Then the creator can specify preferences or let the model invent the character, and the draft waits for human approval
# User Given a character card draft When the creator approves its text and selected reference image Then the approved card and exact reference image SHA are saved before production series creation
# User Given an approved story and character When an episode outline is drafted Then it connects the episode to the series arc and waits for human approval before video generation
# User Given an approved episode outline When serial generation starts Then the approved outline, story version, character card and reference image are the inputs to the existing segment review and accepted-tail flow
