# Reference-video remix workflow

MediaOverload now supports a reference video as a pre-production evidence
source for its existing Native H3 and segmented long-video graphs.

The workflow is intentionally a remix workflow, not a source-video copier:

1. FFmpeg/ffprobe measures the local video or downloads a URL through `yt-dlp`.
2. The runtime writes a `reference_video_brief.json`, uniformly sampled
   keyframes, and a labelled contact sheet under the configured output root.
3. Existing story planning receives the structural brief and keyframes as
   visual evidence.
4. The generated story must keep the selected character and episode objective,
   while borrowing only pacing, framing, motion grammar, and escalation shape.
5. Existing H3 rendering, QA, Discord review, and publication boundaries remain
   authoritative.

## CLI

```powershell
python -m agentic.app.main `
  --goal "Kirby turns a tiny weather warning into a physical gag" `
  --media-type native_h3_story `
  --duration-seconds 15 `
  --style "tactile pastel 2D anime" `
  --character Kirby `
  --reference-video "D:\references\clip.mp4" `
  --reference-keyframes 12 `
  --execute
```

For a URL, `yt-dlp` must be installed and available on `PATH`. The analyzer
fails explicitly when it is not available; it does not silently replace a URL
with a different source. The current `standard` mode uses up to 12 keyframes
and a conservative scene-change threshold. `deep` uses up to 20 keyframes and
a more sensitive threshold.

The output brief is structural by design. Semantic interpretation is performed
by the existing vision-capable story model over the extracted keyframes when a
configured LLM is available. This keeps the evidence inspectable and avoids
presenting a filename or FFmpeg heuristic as a claim about the source video's
characters or plot.

## Contracts and boundaries

- `configs/reference_video_brief.schema.json` describes the persisted brief.
- `agentic/src/agentic/runtime/reference_video.py` owns source resolution,
  probe, scene sampling, keyframes, contact sheet, and remix guidance.
- `reference-video-analysis` is a graph node, not a second orchestrator.
- `reference_video_source` is different from `reference_video_paths`: the
  former is a style/pacing reference; the latter remains a generation
  conditioning bundle for Ref2VA/long-video recipes.
- The source's recognizable assets, logos, readable text, plot, and location
  are explicitly out of scope for copying.
- No social platform receipt is implied by reference analysis or media QA.

OpenMontage's reference-video ideas informed this contract, but its source is
not embedded here. Its AGPL license must be handled separately if code is ever
copied; this implementation uses the existing MediaOverload runtime and
FFmpeg boundary instead.
