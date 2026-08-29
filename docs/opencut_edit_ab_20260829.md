# OpenCut-inspired editing iteration log — 2026-08-29

## Scope

This experiment evaluates the new provider-neutral edit layer against three real generated H3 segment videos already present in the repository. It does not claim that the upstream OpenCut web editor was embedded or that any social platform was published to.

The source order was fixed for every run:

1. `Kirby_H3_draft_00004__agentic_i2v.mp4`
2. `Kirby_H3_draft_00005__agentic_i2v.mp4`
3. `Kirby_H3_draft_00006__agentic_i2v.mp4`

Each iteration rendered an actual MP4, extracted a contact sheet and boundary frames, and was judged for:

- semantic continuity at each segment join;
- ghosting, black flashes, split-screen or other transition artifacts;
- rhythm and perceived intentionality;
- whether the effect supports the story rather than becoming the subject;
- technical duration, dimensions, frame rate and audio continuity.

## Iteration decisions

| Run | Candidate | Observation | Decision | Next change |
|---|---|---|---|---|
| 01 | `baseline_concat` / hard cut | Clean and honest baseline, but the bridge → heart and heart → bird joins feel like stacked source files. No transition grammar. | Keep as baseline only. | Add a restrained transition. |
| 02 | `xfade_clean_v1`, 0.18 s fade | Smoother rhythm, but overlap is long enough to produce visible double exposure around both joins. | Reject as too soft/ghosted. | Shorten overlap to 0.10 s. |
| 03 | `editorial_kinetic_v1`, seed 1 | Wipes create hard vertical splits and compete with the visual story; variety is not automatically interest. | Reject and do not make default. | Test a chapter dip. |
| 04 | `chapter_dip_v1`, 0.10 s fade-to-black | The black dip reads as a rendering gap because there is no beat, title card or intentional blackout to motivate it. | Reject. | Return to clean fade with shorter overlap. |
| 05 | `xfade_clean_v1`, 0.10 s fade | Keeps the causal progression readable, removes the hard-cut discontinuity, and avoids the longer ghosting and black flash seen in prior candidates. | Accept as current default. | Keep editorial variants opt-in and visually reviewed. |

## Evidence

The five render directories are under `output/opencut_edit_ab_20260829/run_01_baseline` through `run_05_short_fade`. Every manifest reports a 15.00-second, 576×1024, 24fps H.264/AAC output with stereo audio. `run_05_short_fade/final.edit_manifest.json` is the historical human-reviewed default candidate; it is not a substitute for the stricter creative-review gate below.

The image-only smoke test also rendered three generated character stills with bounded `slow_zoom_in` motion and the same 0.10-second fade profile. The final-code runtime agent workflow then executed the same edit graph through `image_sequence_edit_v1`, passed technical QA, emitted a GIF preview and persisted its summary under `output/opencut_edit_ab_20260829/runtime_agent_workflow_final`.

## Runtime vision-review loop

The first runtime LLM review changed the quality conclusion. The older human acceptance of the 0.10-second fade was overturned after the local Qwen vision model inspected actual join frames and found double exposure. The loop then tested shorter fades, hard cuts, editorial wipes and bounded camera motion; it never publishes a rejected candidate.

| Runtime | Candidate sequence | Result |
|---|---|---|
| `runtime_creative_review_final_v2` | 0.10 s fade → 0.07 s fade → hard cut | All rejected: fades ghosted; hard cut lacked enough editorial motion. |
| `runtime_creative_review_final_v3` | 0.10 s fade → 0.07 s fade → hard cut → wipe variants | All rejected: clean wipes still produced distracting split-screen geometry. |
| `runtime_creative_editorial_seed3` | `smoothleft`/`circleopen` at 0.10 s | Rejected: the LLM saw double exposure at both joins. |
| `runtime_creative_editorial_seed3_short` | `smoothleft`/`circleopen` at 0.04 s | Rejected: joins were clean, but the middle beat remained a static hold. |
| `runtime_creative_review_final_v4` | fade → shorter fade → hard cut plus `pan_right`/`drift_up`/`drift_down` | Accepted in the first complete-motion run by local Qwen vision, score 90. A later evidence-complete rerun was used as the final result. |
| `runtime_creative_review_final_v5` | fade → shorter fade → hard cut plus `pan_right`/`drift_up`/`drift_down` with hard-cut join frames | Accepted by local Qwen vision, score 95; full before/join/after evidence was supplied for both cuts. |
| `runtime_creative_review_final_v6` | Same bounded motion search, with pre-materialize technical QA and staged source snapshots | Accepted by local Qwen vision, score 95 after 3 reviews; this is the latest post-security-patch receipt. |
| `runtime_creative_review_final_v7` | Same post-security-patch loop with an explicit 15.0-second target | Accepted by local Qwen vision, score 92 after 3 reviews; this is the current exact-duration artifact. |

The latest accepted runtime artifact is `output/opencut_edit_ab_20260829/runtime_creative_review_final_v7/final.mp4`. Its final plan is hard-cut video with `slow_zoom_in`, `pan_left`, and `slow_zoom_out`; the generated `final.join_check.jpg` confirms the two joins switch cleanly without crossfade ghosting, and the candidate review evidence contains opening, before/join/after frames at 5.167s and 10.334s, plus ending. The run receipt records three actual candidate reviews, the selected score of 92, and an exact 15.0-second output. v5 and v6 remain historical accepted runs; v7 is the current duration-constrained evidence after the security hardening.

## Current contract

- Still-only `image_sequence_edit` input defaults to `motion_cut_v1`, which combines bounded zoom/pan motion with clean hard cuts. Video-only input defaults to `baseline_concat`; `xfade_clean_v1`, `chapter_dip_v1` and `editorial_kinetic_v1` remain explicit opt-ins.
- `motion_cut_v1` passed an independent production-style image matrix 3/3 with OpenRouter creative scores 92/92/92 (minimum 92, population stddev 0). The artifacts are under `output/edit_strategy_benchmark_20260829_prod_v7_motion_cut_images`.
- `editorial_kinetic_v1` automatically enables the blocking frame-level creative review loop; a direct adapter render still marks an unreviewed editorial result as unreviewed and is not a publish receipt.
- The agent can choose ordered images or video segments through `EditPlan`.
- Images receive bounded deterministic motion. Creative-review retries can also add bounded pan/zoom/drift motion to video segments when the LLM identifies a static hold; the original source files are never mutated.
- Edit plans are bounded before FFmpeg: clip count, canvas dimensions, FPS, per-clip/source-start/total duration, transition duration, metadata size and estimated render work are capped. Approved sources are hash-checked and staged into a private temporary snapshot before FFmpeg opens them.
- All normalized clips have a stereo audio lane so the composition remains muxable even when the source is a still image or muted video.
- The edit manifest records source hashes, probes, the exact plan, output probe and render metrics; each agent run writes to a unique timestamp/UUID directory to avoid cross-run clobbering.
- Creative review is fail-closed: unavailable/uncertain/invalid-schema vision results and all-rejected candidate sets return `status=failed` and preserve candidate evidence without materializing a final. Technical QA runs before candidate materialization as well as in the downstream graph.
- The edit layer is available as an explicit `longvideo_edit_profile` for non-TTS long-video packaging. TTS long-video remains on the existing audio-owned route until a separate audio-aware timeline contract is validated.

## Independent production-style matrix

`scripts/run_edit_strategy_benchmark.py` invokes the real `image_sequence_edit_v1` graph in isolated trial directories. It does not replace the production planner and does not publish media. Every trial renders an MP4 through the production compositor, runs production technical QA, extracts join/sample evidence, and sends the fixed candidate to the existing OpenRouter vision reviewer. A strategy is `stable_for_fixture` only when all required repeats pass; one good render is never promotion evidence.

On the three generated H3 videos, all 12 trials passed technical QA, but no strategy passed the 3/3 creative gate. The source evidence also showed that the round-3 origami video contains a mid-clip yellow/purple artifact, so that source issue is not attributed to a transition. On the three generated keyframe PNGs, the initial profiles were unstable: `baseline_concat` 2/3, `xfade_clean_v1` 0/3, `chapter_dip_v1` 1/3, and `editorial_kinetic_v1` 0/3. The one-lever `motion_cut_v1` follow-up passed 3/3 and was then verified through the default production CLI route with a successful `image_sequence_edit_v1` run, technical QA, GIF preview and persisted summary.
