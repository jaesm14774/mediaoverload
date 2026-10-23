# NVIDIA VFX post-process acceptance

## User acceptance scenarios

### Final video is enlarged, interpolated, and kept in sync

User
Given a finished SDR MP4 with a constant frame rate and width and height divisible by 4, and ComfyUI has the NVIDIA VFX nodes on an RTX 40-series or newer GPU
When the video enters final post-processing
Then each source frame is upscaled 2x with NVIDIA ULTRA before 2x MEDIUM frame generation, the output is 2x the width and height and frame rate, the final endpoint is held for one output frame so the source duration is retained, and the original audio stream is preserved.

### Long clips fit the configured GPU memory

User
Given a finished video longer than one processing chunk
When the NVIDIA VFX post-process runs
Then it processes overlapping chunks of at most 32 source frames, joins each shared endpoint once, and produces one continuous video with no repeated boundary frame.

### AAC packet timing does not block final processing

User
Given a speed-adjusted video whose AAC audio endpoint differs from its video endpoint by no more than one source frame
When the video canvas is normalized and NVIDIA VFX starts
Then canvas normalization preserves the existing AAC stream timing, the VFX preflight accepts the one-frame difference, and audio/video timing errors larger than the supported tolerance are still rejected.

### Unsupported runtime is reported clearly

User
Given ComfyUI is unavailable, an NVIDIA VFX node is missing, or the GPU cannot run the NVIDIA effect
When final post-processing starts
Then the workflow fails with an actionable error and retains the source video as the upstream artifact.

### Matroska duration metadata can fall back to the container duration

User
Given a valid Matroska or WebM video whose video or audio stream duration is `N/A` while the container duration is available
When the video is inspected for NVIDIA VFX post-processing
Then the tool uses the available container duration for the video and treats an unavailable audio stream duration as unknown without rejecting the source.

### Final checks inspect the post-processed video

User
Given a video generation workflow reaches technical QA, preview, review, or packaging
When NVIDIA VFX post-processing is enabled
Then those final stages use the upscaled and interpolated file and its expected dimensions and frame rate.

### Game-sprite packaging samples the enhanced animation

User
Given a game-sprite goal with NVIDIA VFX post-processing enabled and an H3 output whose dimensions are divisible by 4
When the generated animation enters sprite packaging
Then NVIDIA VFX upscales and interpolates the animation before the packager samples frames for the atlas, while an explicit `enable_nvidia_vfx_postprocess: false` setting sends the original animation directly to packaging.
