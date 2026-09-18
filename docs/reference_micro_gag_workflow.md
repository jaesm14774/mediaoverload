# Reference micro-gag workflow

This is an optimization profile inside the existing MediaOverload route. It
does not add a renderer or bypass the planner, runner, ComfyUI tools, QA, or
manifest lifecycle.

```text
reference video
    -> existing reference.video.analyze evidence node
    -> existing agent.goal.expand with extracted keyframes
    -> existing Krea2 text-to-image first frame
    -> existing MiniMax H3 low-VRAM I2V
    -> existing technical video QA + explicitly requested subject-count checks
    -> existing run manifest / prompt lineage
```

The profile is activated when the existing `text2image2video` generation type
receives `reference_video_source`. The source contributes timing, framing,
motion grammar, and escalation shape only. Characters, props, setting details,
logos, UI, text, and plot are regenerated as original content.

The collection-informed creative contract is deliberately small:

- 4–9 seconds, with the benchmark defaulting to 5 seconds to match the
  existing H3 124-frame I2V profile;
- one protagonist, one tactile prop or force, one objective;
- the hook/action onset is visible in the first frame;
- anticipation -> contact/impact -> consequence -> reaction -> settled payoff;
- the ending echoes the opening enough to loop without hiding the payoff;
- multi-character or rapid-montage references are treated as rhythm references,
  not as one continuous H3 shot.

The benchmark runner performs one initial attempt plus at most three prompt
retries. It keeps one stable seed per case, changes only the retry direction,
and writes every prompt, seed, workflow result, QA contact sheet, and attempt
record under `output/reference_micro_gag_e2e/`.

The prompt contract follows the official MiniMax H3 guidance: I2VA starts from
the first-frame anchor and describes action onset, continuous development, and
result/reaction. See the [H3 video prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
The architecture also remains compatible with the hosted provider's
image-to-video contract, where the prompt describes how the supplied image
evolves; see the [MiniMax I2V API documentation](https://platform.minimaxi.com/docs/api-reference/video-generation-i2v).

## Run

```powershell
python scripts/run_reference_micro_gag_e2e.py `
  --collection-root 'C:\Users\jaesm14774\Downloads\收集' `
  --comfy-root 'D:\ComfyUI_windows_portable' `
  --duration-seconds 6 `
  --max-retries 3
```

The runner never publishes. Hard media checks cover technical requirements
and any explicit subject-count contract. Story alignment, motion, visual
identity, and gag quality belong to the human reviewer in Discord; no LLM
score accepts, rejects, or regenerates a candidate.
