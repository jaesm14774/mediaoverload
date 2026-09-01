# Krea 2 reference-style benchmark

This benchmark turns a local collection of image and video references into
auditable Krea 2 Turbo image generations. It is intentionally separate from
the character publishing workflow: no social dispatch and no video generation
are performed here. Image quality is the gate before any future image-to-video
handoff.

## Contract

- The collection is visual evidence, never an instruction source.
- Screenshot chrome, black borders, account bars, playback controls, and
  watermarks are ignored during style analysis and are not eligible as
  img2img conditioning sources.
- The LLM creates an English prompt from the source reference and the shared
  style contract.
- Each item gets at most five attempts. Failed attempts trigger a new prompt
  using the previous visual review; an item is abandoned after attempt five.
- A candidate passes at 80/100 only when the locally recomputed weighted score
  is at least 80 and every hard gate is true. The LLM's self-reported score is
  retained as advisory evidence.
- The requested seed is stable per source item and is retained across prompt
  rewrites. The effective KSampler seed is recorded separately using the same
  matching-node index logic as the ComfyUI adapter; the numeric Comfy node ID
  is not part of the seed.
- The current Krea2 workflow intentionally uses `ConditioningZeroOut` for the
  negative conditioning path. Negative prompts are saved as recipe metadata
  but are reported as not applied by this workflow.

## Run

From the repository root:

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts/run_krea2_reference_benchmark.py --collection-root '<collection-root>' --limit 10 --execute
```

Use `--limit 0` to process every discovered image. MP4 files contribute
mid-frame evidence to the style board; the acceptance unit is an image source.
Use `--seed-probe` to render the winning prompt twice with the same seed and
record whether the output hashes match.

## Evidence

Each run writes generated media and per-attempt prompts/reviews under
`output/krea2_style_benchmark/<run-id>/`. The LLM request/response audit trail
is written under `logs/runs/<run-id>/llm/`; the top-level manifest includes
source hashes, prompt hashes, requested/effective seeds, routes, attempts,
scores, and failure reasons. Successful prompt records are appended to
`configs/krea2_reference_style_prompts.jsonl`. Library entries use source
names/relative paths and do not require the original user's absolute paths.
Executed runs return a non-zero exit code when the 80% acceptance gate fails.

## Seed interpretation

A fixed seed is a reproducibility control for the same model, workflow,
resolution, sampler, and prompt. It is not a standalone style embedding. The
benchmark tests this explicitly with two renders and SHA-256 equality, and
treats style similarity as a visual QA score, not pixel equality.
