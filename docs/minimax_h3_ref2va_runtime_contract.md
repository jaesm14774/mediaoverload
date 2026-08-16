# MiniMax H3 Ref2VA runtime contract

The Ref2VA base workflow is intentionally empty of reference loaders. The Python binding creates only the loader nodes required by the current run:

- `1..N` accepted images become `LoadImage` nodes wired to `ref_images.ref_image_*`.
- `1..N` accepted videos become `VHS_LoadVideoPath` nodes wired to `ref_videos.ref_video_*`.
- Unused image/video slots are not filled with placeholders and are not sent as `None` conditioning entries.
- Reference audio is never connected. The workflow still decodes H3's generated audio output.

`minimax_h3_ref2va.json` is the single canonical Ref2VA workflow. Model selection is runtime data, not another workflow file:

- `model_profile: q4` keeps the RTX 4060 GGUF loaders.
- `model_profile: q2` keeps the Q4 Ref2VA diffusion model but swaps only the text encoder to Q2_K for OOM fallback.
- `model_profile: native` patches the diffusion/text loader classes and model names in memory before queueing.
- `model_overrides` can explicitly patch node IDs `1` and `2` with `class_type` and `inputs` when an API caller needs a custom model set.

Typical API payload:

```json
{
  "workflow_name": "minimax_h3_ref2va",
  "model_profile": "q4",
  "reference_manifest": [{"path": "D:/.../identity.png", "type": "image"}]
}
```

This low-level binding receives a validated manifest. At planner level,
`native_h3_ref2va` with an empty configured manifest first creates six image
candidates and waits for Discord selection; the selected assets are then
normalized into the manifest shown above. `text2image2native_h3_ref2va` names
the same candidate stage explicitly.

The native profile still requires its model files to exist in the configured D: or E: ComfyUI model directories; selecting it does not download anything implicitly.

These are valid and distinct modes: image-only, video-only, and explicitly mixed image+video. On an RTX 4060, start with one identity image; add a second image only when it contributes a different view or controlled appearance detail. Add a reference video only when motion, camera, or timing is actually needed. Sending duplicate image and video content increases VRAM and runtime without adding a useful constraint.

For Discord or other human review flows, Ref2VA candidates pass through `publish.media.ingest` and `review.assets.select` with `review_all_candidates: true`. The validator consumes only the selected assets and builds the final reference manifest. The ordinary image-to-video route does not use this multi-reference behavior: it resolves the first approved image only.

The effective limits are configurable (`reference_max_images` up to 9 and `reference_max_videos` up to 3), but the runtime never assumes that all limits are populated. `reference_frame_cap` may be used to cap an expensive reference video. Keep all model, cache, and output paths on D: or E: for the current local setup; the repository remains on C:.
