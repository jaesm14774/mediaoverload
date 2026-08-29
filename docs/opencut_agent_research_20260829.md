# OpenCut for agent-controlled automatic editing — research note

## Decision

OpenCut is a useful design reference, but the current upstream repository is not yet the right runtime dependency for unattended MediaOverload editing. The rewrite advertises an Editor API, plugin-first architecture, MCP server and headless/batch mode, while explicitly pointing users to `opencut-classic` for the version available today. The current rewrite editor route is still a placeholder and the API currently exposes health/echo endpoints rather than an editing contract.

The practical choice is therefore a provider-neutral timeline contract inside MediaOverload, with an adapter boundary that can target OpenCut later if its headless API becomes stable. The first implementation uses FFmpeg as the deterministic compiler so it can run in the existing agent workflow, CI and local GPU-independent packaging path.

## What upstream contributes

- The rewrite's direction is aligned with this use case: [OpenCut README](https://raw.githubusercontent.com/OpenCut-app/OpenCut/main/README.md) names third-party plugins, MCP and headless automation as planned capabilities.
- The current rewrite is not an automation-ready editor surface: [`apps/web/src/routes/editor.tsx`](https://raw.githubusercontent.com/OpenCut-app/OpenCut/main/apps/web/src/routes/editor.tsx) renders “Coming soon”, while [`apps/api/src/index.ts`](https://raw.githubusercontent.com/OpenCut-app/OpenCut/main/apps/api/src/index.ts) only defines health and echo routes.
- Classic has the right conceptual model for future mapping: scenes contain main/overlay video tracks and audio tracks; visual elements carry start time, duration, trim boundaries, source duration, params, animations and effects. See the [classic timeline types](https://raw.githubusercontent.com/OpenCut-app/OpenCut-classic/main/apps/web/src/timeline/types.ts).
- Classic's export path is frame-oriented and can include audio through Mediabunny's canvas and audio sources, as shown by the [classic scene exporter](https://raw.githubusercontent.com/OpenCut-app/OpenCut-classic/main/apps/web/src/services/renderer/scene-exporter.ts).
- A community [OpenCut MCP adapter](https://github.com/kenimo49/opencut-mcp) demonstrates agent operations such as adding media/tracks, inserting clips, trimming, moving, splitting, deleting and exporting. Its implementation operates through a live browser session and `window.__editor`; it is useful as an interaction proof, not a stable server-side render API. The adapter also documents an internal `window.__opencut` conversion hook for timeline ticks, which is a maintenance risk.

## Fit for MediaOverload

| Requirement | OpenCut today | MediaOverload implementation |
|---|---|---|
| Agent chooses ordered generated media | Classic browser state can do it; rewrite API is not ready | `EditPlan.clips` with explicit source paths and labels |
| Reproducible rendering | Browser export depends on UI/runtime state | FFmpeg normalization, explicit profile, seed, probes and source hashes |
| More variability | Classic has effects/animations, but agent contract is not yet stable | allowlisted transitions plus bounded still/video pan, zoom and drift variants |
| Visual quality loop | Planned MCP/headless direction, not current upstream contract | local vision-LLM reviews contact sheet and join frames; fail-closed |
| Batch/CI operation | Future rewrite goal; classic MCP requires browser session | unique run directories, atomic materialization, no source mutation |
| Human review boundary | Browser editor is suitable for inspection | candidate artifacts and review receipts remain available; technical QA does not replace creative approval |

## Implemented boundary

`image_sequence_edit` now has four layers:

1. `EditPlan`: ordered clips, bounded motion, transition names, canvas/fps/duration and a deterministic seed.
2. `OpenCutEditAdapter`: validates roots and symlinks, bounds the render budget, hash-checks and stages approved sources, normalizes mixed images/video, supplies a stereo audio lane, renders with FFmpeg and writes source/output evidence.
3. `media.video.compose_timeline` plus `media.materialize_edit`: agent-facing composition and safe final selection; rejected candidates are never copied over the requested final.
4. Vision creative review: the local vision model reads the full contact sheet and join-adjacent frames, returns a strict structured response and a next-change recommendation, and can trigger up to four bounded retries. Technical QA must pass before visual review and materialization. The selected candidate is recorded in `creative_review.json`.

This is intentionally not a claim that OpenCut's web editor or community MCP has been embedded. If the upstream rewrite later exposes a versioned headless scene/export API, the adapter can map `EditPlan` to its scene/track/element model without changing the agent skill or the creative-review receipt.

## Risks and next boundary

- Do not use browser MCP as the only production renderer: it couples the agent to a running UI, private editor globals and browser timing.
- Do not treat transition count as creativity. The review prompt penalizes ghosting, black flashes, split-screen artifacts, frozen holds and effects that dominate the story.
- Generated video segments can already contain motion; bounded post-camera motion is only a retry variant and must be visually reviewed. It is not a substitute for better generation when the source action itself is weak.
- Audio remains deliberately conservative. This edit layer preserves or synthesizes a stereo lane and can crossfade segment audio, but TTS long-video remains on its existing audio-owned route until a separate audio-aware timeline contract is validated.
- A future OpenCut integration should start with an export-only adapter and a manifest round trip, then add browser preview or live scene mutation only after the headless contract, cancellation semantics and output receipts are stable.
