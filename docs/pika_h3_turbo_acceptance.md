User Given the registered Pika H3 Flex Turbo FL2VA workflow and its native 480p canvas, When a T2VA request is prepared with a prompt and seed, Then the graph uses the Q4 H3 base, the four-step Pika adapter, `res_multistep`, and an 832x480 canvas.

User Given ComfyUI is running and all workflow assets are present, When three Pika-only requests use the approved prompts and seeds, Then they produce three 832x480 videos with 124 frames, 24 fps, and stereo audio without queuing the original workflow.

User Given the configured model root does not contain the Pika adapter and `COMFYUI_ASSET_EXTRA_ROOT` points to another ComfyUI model root, When asset readiness is checked, Then the adapter is found under that root's `models/loras` directory without a workstation path in the shared workflow configuration.

The contract test covers graph construction. The three-video scenario requires ComfyUI and a GPU; its locally verified run and artifacts are recorded in [the review](pika_h3_turbo_review.md).
