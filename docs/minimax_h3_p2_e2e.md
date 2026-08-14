# MiniMax H3 P2 canonical modes

P2 now exposes five explicit H3 modes through one conditioning contract:

| Mode | Conditioning | Canonical workflow | Default E2E length |
| --- | --- | --- | --- |
| T2VA | prompt only | `minimax_h3_lowvram_t2v` | 362 frames / 15 s |
| I2VA | one generated opening image | `minimax_h3_lowvram_i2v` | 124 frames / 5 s |
| FL2VA | generated opening + generated landing image | `minimax_h3_lowvram_15s_fl2va_i2v` | 362 frames / 15 s |
| L2VA | one generated landing image; opening input is cleared | `minimax_h3_lowvram_15s_fl2va_i2v` | 362 frames / 15 s |
| Ref2VA | runtime image/video reference manifest with roles | `minimax_h3_ref2va` | 124 frames / 5 s |

The same base graph is reused only where the conditioning semantics are the
same. A mode cannot silently receive a frame or reference that belongs to a
different mode. Reference audio is intentionally disabled.

## Real ComfyUI E2E

The runner generates all prerequisite images and videos through ComfyUI, then
executes the H3 render and strict technical QA. Models are resolved from the
configured D/E-drive ComfyUI root; the runner does not create model or media
artifacts on C:

```powershell
python scripts/run_h3_modes_e2e.py `
  --comfy-root 'D:\ComfyUI_windows_portable' `
  --output-root 'D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_h3_p2_e2e'
```

For a single mode, repeat `--mode` as needed. The report is merged instead of
overwritten when modes are run one at a time:

```powershell
python scripts/run_h3_modes_e2e.py --mode fl2va
```

To re-run QA without regenerating expensive clips:

```powershell
python scripts/verify_h3_e2e_outputs.py
```

## Quality gate

The P2 gate checks file existence, video stream, exact 608x352 canvas, 24 fps,
target duration, native generated audio, stereo channels, audio/video duration
alignment, mean loudness, peak clipping, sustained silence, and a contact
sheet for visual inspection. The production Kirby recipes enable the same
audio checks through `native_h3_recipe`; human review remains separate from
the technical gate.
