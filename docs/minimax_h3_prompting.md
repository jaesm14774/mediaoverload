# MiniMax H3 prompt behavior

The local H3 path passes the requested brief, selected style, and optional character reference to the prompt composer. It does not require a fixed number of story beats, actions, camera moves, sound effects, or character counts.

The composer preserves optional details supplied by the user or generated story, such as scene notes, shot descriptions, and audio direction. Missing optional details stay missing. For image-to-video, the supplied frame is identified as the starting frame because that is part of the renderer input.

Runtime checks cover renderer inputs such as supported workflow, duration, dimensions, and frame sources. Visual quality and creative choices remain available for human review.

For `native_h3_fl2va_story`, opening and ending candidates are independent text-to-image illustrations of their respective story states. The ending never uses the approved opening as an image-to-image input. Both use the same character and selected illustration style, while the final pose, expression, prop placement, and camera follow the last story beat. Identical opening and final state descriptions are rejected before ending generation. Human review must still verify that the rendered images show the intended change; different text or files alone do not prove a visible story outcome.

## Code entry points

- General prompt builders: `agentic/src/agentic/runtime/prompting.py`
- Native H3 prompt composer: `agentic/src/agentic/minimax_prompting.py`
- Native H3 storyboard normalization: `agentic/src/agentic/storyboard.py`
- Native H3 story generation: `agentic/src/agentic/runtime/prompt_tasks/native_h3.py`
- H3 execution skill: `agentic/src/agentic/skills/longvideo.py`
- ComfyUI renderer: `agentic/src/agentic/tools/comfy_h3.py`
