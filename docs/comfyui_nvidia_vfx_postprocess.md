# NVIDIA VFX video post-process

## ComfyUI setup

The runtime uses the ComfyUI NVIDIA VFX nodes for the final video pass. The Windows portable setup used on this machine is:

```powershell
cd D:\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes.git
git -C ComfyUI_Nvidia_VFX_Nodes checkout 38d790ded00650681fa3645362efe28c8560cf40
cd ..\..
python_embeded\python.exe -m pip install wheel-stub
python_embeded\python.exe -m pip install --no-build-isolation nvidia-vfx==0.2.0.0 --index-url https://pypi.nvidia.com
```

Restart ComfyUI after installation. NVIDIA's frame generation requires an RTX 40-series or newer GPU; the installed `nvidia-vfx` package must be installed in the same Python environment that starts ComfyUI. This machine was set up with `nvidia-vfx 0.2.0.0`, NVIDIA node revision `38d790ded00650681fa3645362efe28c8560cf40`, CUDA-enabled PyTorch cu130, and an RTX 4060.

The project does not install NVIDIA's SDK into the agent Python environment. The SDK runs inside ComfyUI, which owns the GPU and its memory.

## Processing order

The final video path runs:

1. Read at most 32 source frames through VideoHelperSuite, overlapping adjacent chunks by one frame.
2. Upscale each chunk 2x with `RTXVideoSuperResolution`, `ULTRA`, strength `1.0`, and 8-bit RGB.
3. Generate 2x the frame rate with `RTXVideoFrameGeneration`, `MEDIUM`, with automatic shot-change detection.
4. Join the chunks, remove each repeated boundary frame, hold the final endpoint for one output frame, and mux the source audio back onto the result. Canvas normalization copies an existing AAC stream and bounds the output to the video stream duration, avoiding a second AAC encode and its packet-end drift.

For a 640x360, 24 FPS source, the expected output is 1280x720 at 48 FPS with the same duration and audio channels. NVIDIA's node emits `(N - 1) * multiplier + 1` frames for `N` input frames. Holding the final frame once restores the `N * multiplier` frame count needed to keep the original duration at the doubled frame rate.

Source width and height must each be divisible by 4. NVIDIA VSR rounds scaled output dimensions to the nearest multiple of 8; this input constraint keeps the output dimensions at exactly 2x and lets final QA use the same expected dimensions. The tool rejects other sizes before uploading frames to ComfyUI.

HDR is not enabled. The project outputs SDR video, and the current node setup uses the 8-bit RGB path.

## Runtime behavior

NVIDIA post-processing is enabled by default for generated video plans, including short videos, native H3, long videos, and motion comics. Set `enable_nvidia_vfx_postprocess: false` in goal constraints to skip it explicitly. If ComfyUI or a required node is missing, the workflow fails with an installation/runtime error and keeps the upstream file available for diagnosis.

The output summary records source and result media metadata, selected node settings, chunk boundaries, processing time, and any error. QA, GIF preview, final review, and packaging consume the post-processed file.

ComfyUI retains uploaded source videos and per-chunk output files in its input and output directories. Apply the server's retention policy to keep repeated long-video runs from accumulating files.

## Validation performed

On 2026-10-16, a 12-frame clip with AAC audio completed end to end in 7.4 seconds and produced 512x288, 48 FPS, 24 frames, and a 0.5-second result with its audio intact. A real 640x360 H3 render with 362 frames and stereo AAC completed in 12 chunks in 82.1 seconds on the RTX 4060. Its output probe reported 1280x720, 48 FPS, 724 frames, 15.083 seconds, and the original two audio channels and duration. Three matched scene frames were visually checked at the beginning, middle, and end; the scene composition and subjects remained intact after upscaling.

## Validation

Acceptance scenarios: [NVIDIA VFX post-process acceptance](comfyui_nvidia_vfx_postprocess_acceptance.md).

Run the focused planning contract tests with:

```powershell
python -m pytest agentic/tests/test_nvidia_vfx_planning.py -q
```
