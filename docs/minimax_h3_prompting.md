# MiniMax H3 prompt contract

MiniMax H3 uses different prompt contracts for base video modes and Ref2VA. The runtime formats the prompt for the selected mode immediately before submission to ComfyUI.

## Base video modes

T2VA, I2VA, FL2VA, and L2VA use these three fields in order:

1. `integrated_multimodal_description`
2. `overall_soundscape`
3. `non_diegetic_music`

Image alignment is explicit in the scene description. I2VA aligns `<Picture 1>` with the start. FL2VA aligns `<Picture 1>` with the start and `<Picture 2>` with the end. L2VA aligns `<Picture 1>` with the end. The scene itself should describe a visible action and its result; a single H3 segment should read as one continuous shot.

## Ref2VA

Ref2VA uses six sections: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music`. Each `<Picture N>` or `<Video N>` label follows the ordered reference manifest. A continuation image is named as the starting frame; identity and style references stay in subject definitions.

## Long-video image path

The first segment prompt is a scene draft used to create candidate keyframes. After the first-frame and optional last-frame anchors have been selected, a vision-model call receives those exact files and the draft scene. It writes the final H3 scene plus separate audio directions. The video render depends on that result, then the mode formatter adds the correct image alignment and field labels.

This order lets the prompt writer inspect the actual image while keeping keyframe creation independent from the final H3 wording. Subsequent segments use the rendered tail as their first-frame image, so their vision prompt can preserve the appearance and setting H3 actually produced.

Describe motion positively as an ordered chain: what moves from the first moment, its target and path, the next connected action, and the visible resulting state. Use two or three concrete action verbs. A state description plus negative rules such as “do not hold” did not produce reliable movement in the paired H3 renders; an ordered sword-to-chart action did.

## Single-focus action continuity

### Acceptance scenarios

- User Given a multi-segment action story with one lead subject, a required supporting pursuer, and connected street changes, When the segments and H3 scene prompts are prepared, Then the lead remains the sole visual focus, the pursuer keeps its supporting role and relative position, and each setting change follows the route or a motivated cut without reversing screen direction.
- User Given a brief that explicitly requests an ensemble or co-leads, When the video prompts are prepared, Then the requested subjects can share focus while their stated identities and relationships remain consistent.
- User Given a long video requires human review of every segment, When a segment is rendered and its technical inspection completes, Then Discord acceptance happens before its observed tail can condition the next segment.

Unless the brief requests an ensemble or co-leads, video planning selects one primary visual focus. Other necessary subjects stay subordinate to it; they retain their named identity, action role, and stated lead/follow or pursuer/fleeing relationship across segments. Vary camera distance, angle, and setting to create energy without reversing the action axis or making a supporting subject take over. A location change must follow a described route or a motivated cut, and the new shot must preserve the action's readable direction and spatial order.

These are prompt-planning constraints, not a guarantee from H3. Review every segment and both sides of each cut for role, identity, and screen-direction continuity before accepting the assembled clip.

For `longvideo_review_policy: every_segment` with human review enabled, each rendered segment is inspected technically and then presented for explicit acceptance. The accepted review output supplies the video path for tail-frame extraction, and all segment acceptances gate final assembly; rejection or unavailable review blocks the sequence. The normal final-video review remains a separate decision. Segment acceptance authorizes that take as a continuity source; it does not establish that the whole video is visually successful or ready to publish.

## Lively loop direction (collection reference videos)

The reference clips in the collection are 4–9 second flat illustrated loops. Two same-seed, same-keyframe H3 I2VA comparisons (121 frames, 512x640, rain-puddle keyframe) set the direction:

- The earlier direction (anticipation, goal, strong camera dolly, reaction) had plenty of energy but dollied until the character filled the frame, so the picture no longer matched the keyframe.
- A strictly locked camera with "one small action, then settle" kept the look, but the character acted once and the clip stayed static for the remaining 60%, which the user judged too still.
- Continuous action fixed it: constant energetic motion through the last frame, with a locked camera or one slow gentle push-in of about 15%, kept both the energy and the painted look. A push-in plus a longer "performance arc" prompt overshot the zoom and made the character jump less, so the arc wording is not used.

`VIDEO_MOTION_DIRECTION` and `VIDEO_OPENING_FRAME_DIRECTION` in `agentic/src/agentic/video_directing.py` therefore default to: at most one slow gentle push-in (no pan, orbit, or cut), unchanged illustrated look, constant energetic character motion (anticipation, a big action with stretch and squash, the same action chained into two or three connected moves, expression change), props and existing scene elements responding on every beat, still moving in the last frame, and no second goal or subplot. The vision writer is told to start with a motion verb and never describe the picture as it stands; without that, a calm keyframe produced a still description. The opening keyframe shows the character small with open space so nothing crops when it jumps. A brief that explicitly asks for camera movement or stillness still overrides this. Rendering one 5-second clip took about 11 minutes on the local 8 GB GPU.

### Replayable gag and the loop-style strategy

Without a news source, the native story prompt asks for one small replayable gag: the character wants something within reach, one prop pushes back, the whole body and face react big, and the last move returns the character and every prop to the opening arrangement, still in motion in the last frame. News-grounded stories keep their resolved final pose. Multi-beat stories inside one clip continue "in the same unbroken shot" (never "the camera cuts to"), consistent with the no-cut rule above.

The loop strategy builds on this:

- `native_h3_loop_story` (10 s, 243 frames): one reviewed Krea2 keyframe is bound as both `image_path` and `last_image_path` of an FL2VA render, so the clip ends where it began. No ending keyframe, FLUX refine, or distinct-state check is involved; `native-keyframe-source-check` reads the opening node for both frames. The local FL2VA contract accepts the same image for both inputs, and one prototype render completed on the local GPU. Whether the seam reads as seamless on a given take is a human judgement in Discord review; tests only prove the binding.

## Review and limits

Local contracts verify which prompt and image paths reach the ComfyUI boundary. They do not prove visual quality. Review the generated video for identity, image adherence, visible action, endpoint continuity, and audible scene response. The official MiniMax prompt skill and ComfyUI H3 documentation are the format sources:

- [MiniMax H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md)
- [Official base-mode prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)
- [Official Ref2VA prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)
- [ComfyUI native H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)

## Code entry points

- Prompt formats: `agentic/src/agentic/minimax_prompting.py`, `agentic/src/agentic/h3_reference.py`
- Image-conditioned H3 prompt writing: `agentic/src/agentic/runtime/prompt_tasks/scenes.py`
- Long-video prompt dependencies: `agentic/src/agentic/runtime/planning/long_video.py`
- Render integration: `agentic/src/agentic/skills/longvideo.py`
- ComfyUI renderer: `agentic/src/agentic/tools/comfy_h3.py`
