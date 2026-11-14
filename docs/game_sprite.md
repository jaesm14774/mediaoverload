# Game Sprite Mode

`game_sprite` is an additive MediaOverload generation mode for game-ready
motion assets. It does not replace or modify the existing image, sticker,
story, Native H3, audio, review, publishing, or scheduler contracts.

## What it produces

The mode passes the requested motion to image and video prompt generation and
packages the rendered result as:

- a 4x4 RGBA atlas with exactly 16 frames, preserving each frame's generated background;
- individual RGBA PNG frames;
- a looping GIF when the generated motion is periodic;
- a lossless animated WebP;
- `manifest.json` with frame geometry, timing, background handling, prompt
  lineage, and deterministic QA results.

The 4x4 grid is only the export format. The creative motion is not limited to
fixed states or a closed action vocabulary. Each LLM request receives a small,
ephemeral sample from the `game_sprite` strategy context in
`configs/routing.yaml`. These
notes are abstract inspiration only: they are not character abilities, a
moveset, a history, or a uniqueness database. The LLM may ignore them and
freely decide the motion.

The system records optional motion notes when the prompt model supplies them.
They do not need a fixed beat count or causal structure, and the system does
not rewrite them through an identity or scene repair pass. Packaging preserves
the generated background instead of requiring a color key, but the keyframe
prompt asks for a plain backdrop because the cell is read at thumbnail size.

The reference pack is sampled per request and recorded in the motion plan for
lineage. Nothing is written back to the strategy config, and no character is
permanently associated with any reference.

## Explicit route

Use the direct agentic route when the mode should be fixed:

```bash
agentic --goal "Kirby folds a glowing leaf into a tiny glider and returns" \
        --media-type game_sprite \
        --character Kirby \
        --style "bright toy game art" \
        --duration-seconds 5 \
        --execute \
        --comfy-host 127.0.0.1 \
        --comfy-port 8188
```

The Kirby character workflow can also select the mode explicitly with
`preferred_generation_type=game_sprite`. In that path, the selected
`generation.creative_profile.canvas` sets the run's source-image and video
canvas. `generation.game_sprite` supplies the Krea/H3 workflows, timing, and
atlas cell settings. The mode uses Krea for one clean source image,
MiniMax H3 I2V for the continuous clip, then samples that clip into 16 frames
in one decode pass. The manifest records the exact interval-midpoint sample
times under `sampling.timestamps_seconds`.

## Packaging

The keyframe is rendered on a flat green backdrop and the clip keeps it. The
packager cuts the backdrop out of every frame (green must lead red and blue,
which tolerates the shade drift video models add), crops all frames to one
shared box so the scale never changes, downsamples with premultiplied box
resampling, and quantizes every frame to one 32-colour palette. The atlas and
frames are transparent PNGs; `animation.gif` and `animation.webp` are
enlarged with nearest-neighbour to about 384 pixels so the pixels stay crisp.
A frame with no green backdrop is kept whole instead of being erased.

## Render budget

The atlas keeps 16 frames at 128x128, so the H3 clip is sized for that, not for
a finished video. Measured runs spent 12 to 58 minutes in the H3 stage and
under two minutes everywhere else, because the clip was 10 seconds
(243 frames) at the full keyframe canvas.

- `generation.game_sprite.duration_seconds` sets the clip length (default 5,
  which H3 rounds up to 124 frames). `game_sprite` is exempt from the
  ten-second minimum that applies to other video routes.
- `generation.game_sprite.video_max_edge` (default 384) caps the clip canvas.
  The clip keeps the keyframe's aspect ratio, aligned to 32 pixels, and never
  drops below H3's 256-pixel minimum edge. The Krea keyframe still renders at
  the full run canvas.
- The route has no editorial-pacing or NVIDIA upscale/frame-generation node.
  Pacing exists to keep audio in sync and the upscale feeds 128x128 cells, so
  both cost minutes and VRAM swaps for no visible gain.

Lower `duration_seconds` or `video_max_edge` for speed; raise them if motion
or detail in the 128x128 cells looks too coarse. Review a rendered atlas after
changing either value.

## Random route selection

Kirby's `generation.generation_type_weights` now contains `game_sprite: 1`.
When a character workflow is invoked without a preferred generation type and
provides its normal RNG, the existing weighted selector may choose
`game_sprite`. The global `routing.strategy_candidates` list also exposes the
new mode to the existing route planner. Existing weights are unchanged; the
new mode is simply an additional candidate.

To make a run deterministic, pass the explicit generation type. To let the
existing scheduler or character workflow choose, omit the preference and keep
the configured RNG/history path. Every selected run records
`source_generation_type`, `routing_selection_source`, selected workflows, and
the sprite manifest in its run evidence.

## Route isolation

`game_sprite` reads only its own `generation.game_sprite` recipe for Krea/H3
and packaging values. It does not inherit the `native_h3_story` recipe, so
Native H3 story fields such as `require_human_review: true`, 15-second story
timing, and required audio remain unchanged and do not leak into this asset
route.

The adapter preserves sampled backgrounds and existing source transparency,
computes one shared alpha bounding box across the frames, and fits them into one
stable cell size. It rejects fully transparent source frames. The exported
atlas contains exactly 16 sampled frames. These checks are deterministic;
visual review of the resulting atlas is still recommended for art quality,
silhouette readability, and game feel.

## One character per keyframe

A run once produced a five-panel storyboard with five cropped copies of the
character, and the atlas sampled that. The cause was the prompt, which asked
the model to stage five phases and mentioned the atlas and a ten-second clip.
Now:

- The model is told to write one still frame with exactly one character, at
  most one prop, and one visual motif from a crowded news brief. It is not told
  about the atlas, frame count, fps, or clip length.
- The compiled keyframe prompt starts with the one-character, no-sheet,
  plain-backdrop contract and the pixel-art rendering contract. The selected
  style comes last and lends only its palette, never its medium, composition,
  or exclusions, so a painterly style cannot override the game-art look. A
  style that defines no palette adds nothing; the sprite keeps its default
  colours.
- The video prompt is one short paragraph plus a compact choreography
  contract, because the H3 text encoder reads every word.

These are prompt contracts, not guarantees. Krea can still ignore them, so
review the keyframe before trusting the atlas.

## Skill-effect motion

`game_sprite` treats the character, theme, and style as references for one
playable skill animation. A quiet or static brief supplies its mood and visual
design; it does not leave the character idle. The video prompt asks for a
single action with visible anticipation, charge, release, impact or effect
peak, and recovery. The character's pose and the effect must change together:
the VFX originates from a visible source, travels or expands with the action,
then resolves with a contact cue and follow-through. A brief held impact pose
can sell force; long idle holds and camera-only movement cannot substitute for
the skill animation. The prompt also keeps the character large and readable in
the keyframe, with room for the effect to move.

**User Given** a game-sprite request whose scene or style is quiet or static,
**When** the motion plan and clip are generated, **Then** the scene is re-staged
as one readable skill effect with distinct preparation, charge, release,
impact or peak, and recovery phases in the 16-frame atlas; each phase changes
the character pose or visible effect, and the requested identity and visual
style remain recognizable. A still character with only camera or background
movement does not satisfy this scenario.

This is a generation prompt contract, not proof of a provider's output. The
manifest and deterministic packaging QA do not establish that the skill is
readable or sufficiently animated; review the resulting atlas and animation.

## Output checks

The packager checks frame count, configured cell and atlas dimensions, binary
alpha, and nonempty frames. Subject size, identity, motion, background, and
appearance remain creative review decisions. A pixel-area ratio is not a proxy
for character quality.
