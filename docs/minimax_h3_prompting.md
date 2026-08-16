# MiniMax H3 prompt contract

This repository uses a local prompt composer that mirrors the structure shown in MiniMax's H3 documentation. It does not call the hosted H3-Context-IR endpoint, so scheduled runs remain inside the existing `run_media_interface.py` and ComfyUI workflow architecture.

## Prompt order

Every video-capable prompt is organized as:

1. Subject identity and continuity
2. Scene and story intent
3. One primary physical action
4. Environment and causal obstacle
5. Camera movement attached to that action
6. Style, lighting, and quality
7. Timestamped shot handoff and end state
8. `overall_soundscape` and `non_diegetic_music`

Native H3 uses the same order as one `integrated_multimodal_description` with `[Shot N / SHOT N | time]` entries. The five-beat Kirby story contract is hook, promise, escalation, reversal, and payoff.

## Text-to-video vs image-to-video

- Text-to-video establishes Kirby inside the first moving action. It must not begin as a posed character sheet.
- Image-to-video treats the first frame as authoritative. Its prompt describes how that exact image starts moving and evolves; it must not redraw a competing opening composition.
- Both modes preserve one protagonist, one readable geography, and a visible state handoff between beats.

## Code entry points

- Shared composer: `agentic/src/agentic/minimax_prompting.py`
- General prompt builders: `agentic/src/agentic/runtime/prompting.py`
- Native H3 storyboard rules/formatter: `agentic/src/agentic/storyboard.py`
- Native H3 news/LLM orchestration: `agentic/src/agentic/runtime/story_service.py`
- LLM story/repair instructions: `agentic/src/agentic/runtime/llm_engine.py`
- ComfyUI execution: `agentic/src/agentic/skills/longvideo.py`

## Official references

- [MiniMax H3 video generation guide](https://platform.minimax.io/docs/guides/video-generation)
- [MiniMax H3 Context-IR API](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir)
- [MiniMax H3 announcement](https://minimaxi.com/blog/minimax-h3)

The official guide documents H3's multimodal input modes, 4–15 second duration range, prompt length limit, and native audio output. The Context-IR example demonstrates timestamped shots plus separate soundscape and music descriptions; the local composer follows that shape while keeping the workflow offline/local.
