# Wan2GP H3 acceptance

- User Given a text-only H3 story When rendering Then WanGP FL2VA receives no image conditioning and returns a playable video with native audio.
- User Given a scene prompt containing paragraph breaks When submitting one video Then the entire prompt remains one generation task instead of being split into separate prompts.
- User Given a retained opening frame and prepared multi-paragraph scene When the actual native-story render skill runs Then it forwards the full scene to the registered WanGP renderer and produces real stereo audio/video. A one-step smoke case verifies this integration without making a quality claim.
- User Given an approved opening image When using I2VA Then the image reaches `image_start` with flag `S`; no end image or appearance-reference input is substituted.
- User Given approved opening and ending images When using FL2VA Then both images reach `image_start` / `image_end` with flag `SE`.
- User Given only an approved ending image When using L2VA Then `image_end` has flag `E` and no opening image is added.
- User Given ordered image or video references When using Ref2VA Then the dedicated Ref2VA checkpoint receives `image_refs` / `video_guide*`, keeps their labels and order, and generates its own audio.
- User Given a missing required image or an invalid reference bundle When rendering Then validation fails before starting GPU generation; no other route runs instead.
- User Given a corrupt image or video file When supplying conditioning Then the media is rejected before GPU submission.
- User Given a requested canvas and frame count When H3 requires aligned dimensions and 17n+5 frames Then the generation uses sufficient aligned frames and the delivered MP4 preserves the requested dimensions, FPS, frame count and audio duration. Tail-conditioned routes preserve the last generated frame when resampling.
- User Given successive clips in one runtime When rendering Then one isolated WanGP process initializes once, records effective settings, conditioning file hashes, events and timings, and releases its models after each job so Krea / reviews can use the GPU.
- User Given any existing H3 generation strategy When planning and rendering Then the registered video tool and workflow are WanGP; obsolete Comfy H3 routes are unavailable.
- User Given a real WanGP MP4 When the planned native-story suffix applies 2x playback and media QA Then QA uses the delivered duration divided by two and the output preserves 24 FPS and stereo audio.
- User Given a retained 15-second L2VA run whose historical prompt fails the current scene-only contract When replaying the historical benchmark through the official API Then the exact original prompt, ending image, width, height, frame count, steps, text-encoder profile and recorded seed reach WanGP and produce real native audio/video; production prompt validation remains enforced.

GPU acceptance runs must use real files and ffprobe. Unit validation proves input contracts, not semantic fidelity. No social publishing or human-review decision is part of this backend acceptance.
