# Game Sprite Mode

`game_sprite` is an additive MediaOverload generation mode for game-ready
motion assets. It does not replace or modify the existing image, sticker,
story, Native H3, audio, review, publishing, or scheduler contracts.

## What it produces

The mode passes the requested motion to image and video prompt generation and
packages the rendered result as:

- a transparent 4x4 atlas with exactly 16 frames;
- individual RGBA PNG frames;
- a transparent looping GIF when the generated motion is periodic;
- a lossless animated WebP;
- `manifest.json` with frame geometry, timing, chroma-key settings, prompt
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
not rewrite them through an identity or scene repair pass. The renderer still
uses a uniform chroma-key background and a 4x4 atlas because those are required
to extract the exported frames.

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
`preferred_generation_type=game_sprite`. In that path, Kirby YAML contributes
the resolved character profile, visual style, Krea/H3 workflow names, and
sprite technical parameters. The mode uses Krea for one clean source image,
MiniMax H3 I2V for the continuous clip, then samples that clip into 16 frames.

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

Each run chooses a flat chroma background from a palette. The adapter removes the
recorded per-run chroma color, computes a shared alpha bounding box across all
frames, fits the subject into one stable cell size, and rejects empty or
motionless outputs. The H3 source clip is now configured for 8 seconds; the
exported atlas still contains exactly 16 sampled frames. These checks are
deterministic; visual review of the resulting atlas is still recommended for
art quality, silhouette readability, and game feel.

## Output checks

The packager checks frame count, configured cell and atlas dimensions, binary
alpha, nonempty frames, and per-frame edge-contact evidence. Edge contact is
reported for review rather than used to reject a render. Subject size, identity,
motion, and appearance remain creative review decisions. A pixel-area ratio
is not a proxy for character quality.
