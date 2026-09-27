# MiniMax H3 prompt behavior

The local H3 path passes the requested brief, selected style, and optional character reference to the prompt composer. It does not require a fixed number of story beats, actions, camera moves, sound effects, or character counts.

The composer preserves optional details supplied by the user or generated story, such as scene notes, shot descriptions, and audio direction. Missing optional details stay missing. For image-to-video, the supplied frame is identified as the starting frame because that is part of the renderer input.

Runtime checks cover renderer inputs such as supported workflow, duration, dimensions, and frame sources. Visual quality and creative choices remain available for human review.

## Code entry points

- General prompt builders: `agentic/src/agentic/runtime/prompting.py`
- Native H3 prompt composer: `agentic/src/agentic/minimax_prompting.py`
- Native H3 storyboard normalization: `agentic/src/agentic/storyboard.py`
- Native H3 story generation: `agentic/src/agentic/runtime/llm_engine.py`
- ComfyUI execution: `agentic/src/agentic/skills/longvideo.py`
