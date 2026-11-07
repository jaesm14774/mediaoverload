# NVIDIA VFX post-process acceptance

## User acceptance scenarios

### Final video is enlarged, interpolated, and kept in sync

User
Given a finished SDR MP4 with a constant frame rate and width and height divisible by 8, and ComfyUI has the NVIDIA VFX nodes on an RTX 40-series or newer GPU
When the video enters final post-processing
Then each source frame is upscaled 3x with NVIDIA ULTRA before 2x MEDIUM frame generation, the output is 3x the width and height and 2x the frame rate, the final endpoint is held for one output frame so the source duration is retained, and the original audio stream is preserved.

### A 16:9 canvas is delivered as full HD

User
Given a finished 640x360 SDR MP4
When it completes final post-processing
Then the delivered video is 1920x1080.

### Sizes that cannot scale to exact dimensions are rejected early

User
Given a source whose width or height is not divisible by 8
When final post-processing starts
Then it is rejected before ComfyUI is contacted, because NVIDIA VSR rounds output dimensions to a multiple of 8 and the 3x result would not be exact.

### Final output contains every expected frame

User
Given ComfyUI has returned each VFX chunk and the chunks are joined
When final video validation completes
Then one full audio/video decode reports exactly `source frame count × frame-rate multiplier` frames, the output retains the source duration and audio, and has no missing or repeated chunk boundary frame.

### The final endpoint adds exactly one output frame

User
Given the joined NVIDIA VFX chunks contain `(source frame count - 1) × frame-rate multiplier + 1` frames
When final assembly holds the last frame for exactly one output frame
Then the output has `source frame count × frame-rate multiplier` frames and passes final frame-count validation.

### Silent source videos remain valid

User
Given a valid video-only MP4 without an audio stream
When source validation counts decoded frames
Then it fully decodes the video, reports the exact frame count, and does not require an audio stream.

### Long clips use larger overlapping batches

User
Given a finished video longer than 128 source frames and an RTX GPU with enough memory for its input dimensions
When the NVIDIA VFX post-process runs
Then it processes overlapping chunks of at most 128 source frames, joins each shared endpoint once, and produces one continuous video with no repeated boundary frame.

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

### Game-sprite packaging samples the raw animation

User
Given a game-sprite goal
When its plan is built
Then the packager samples the atlas frames straight from the rendered H3 clip, with no NVIDIA VFX or editorial-pacing node in between.
