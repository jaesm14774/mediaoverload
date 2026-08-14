# MiniMax H3 進階玩法：L2VA 與 Ref2VA

這個 repo 現在把 H3 拆成四條可追溯路徑：

| mode | 條件影像 | 適合用途 |
|---|---|---|
| `native_h3_story` | opening frame，可選 approved last frame | 角色一致性的標準 causal I2V |
| `native_h3_l2va_story` | 只有 approved last frame | 從結尾姿態／構圖反推一段新片，避免 opening frame 污染 |
| `native_h3_t2v_story` | 無影像 | 純 prompt 探索 |
| `native_h3_ref2va` | 多張 reference image + reference video | identity、style、motion、camera、environment 的組合控制 |

Ref2VA 本地實作只允許 image/video reference。`audio`、`reference_audio`、音訊檔案作為 reference 都會在 manifest、skill 和 Comfy binding 邊界被拒絕；影片內的聲音仍由 H3 原生 audio VAE 依 prompt 生成。

## P0：L2VA

L2VA 是 FL2VA workflow 的明確變形，不是把一張「ending image」誤塞進一般 I2V：

1. 生成 ending keyframe。
2. 人工 review（預設開啟）。
3. `media.image.validate_last_frame` 驗證檔案並固定 selected asset。
4. Comfy binding 清掉 `MiniMaxH3ImageToVideo.first_frame`，只保留 `last_frame`。
5. 完成同一套 technical QA、semantic QA、GIF preview、package。

執行：

```powershell
python run_media_interface.py `
  --character kirby `
  --generation-type native_h3_l2va_story `
  --comfy-host 127.0.0.1 `
  --comfy-port 8188 `
  --comfy-root D:\ComfyUI_windows_portable `
  --no-publish
```

若暫時不要 Discord review，可以加 `--no-review`；這會只產生一個 ending candidate，且不自動重生 continuity frame。

## P1：Ref2VA reference manifest

在 `configs/characters/kirby.yaml` 的 `generation.native_h3_ref2va` 填入 reference。每筆 record 至少需要 `path`；建議明確指定 `type`、`role`、`retention`：

```yaml
  native_h3_ref2va:
    storyboard_path: configs/storyboards/kirby_native_15s_5beat.yaml
    duration_seconds: 15
    workflow_name: minimax_h3_ref2va
    reference_image_size: match
    reference_max_images: 3
    reference_max_videos: 1
    reference_manifest:
      - path: D:/MediaOverload/references/kirby_identity.png
        type: image
        role: identity
        retention: identity_and_appearance
        notes: preserve the approved pink/red silhouette and face proportions
      - path: D:/MediaOverload/references/storm_camera.mp4
        type: video
        role: camera
        retention: camera_motion_and_timing
        notes: use as motion/camera reference only
    width: 608
    height: 352
    length: 124
    steps: 20
```

可用 `role`：`identity`、`subject`、`style`、`environment`、`motion`、`camera`、`continuation`。預設低 VRAM policy 是最多 3 張 reference image、1 支 reference video；要試官方較高上限，可以把 recipe 的 `reference_max_images` 提到 9、`reference_max_videos` 提到 3，但 RTX 4060 8 GB 應先從 1 image + 1 video 開始。

執行：

```powershell
python run_media_interface.py `
  --character kirby `
  --generation-type native_h3_ref2va `
  --comfy-host 127.0.0.1 `
  --comfy-port 8188 `
  --comfy-root D:\ComfyUI_windows_portable `
  --no-publish
```

Ref2VA 執行時會建立：

- `reference_manifest`：送入 H3 的 deterministic slot order。
- `reference_lineage`：每個來源的 path、type、role、tag、retention、檔案大小、mtime。
- `reference_audio_enabled: false`：防止後續 workflow 變更時誤接 audio reference。
- `native_h3_prompt`：六段 Ref2VA prompt contract，包括 Subject Definitions、Summary、Reference Retention、Detailed Description、Overall Soundscape、Non-Diegetic Music。

## ComfyUI 與模型

`minimax_h3_ref2va.json` 是唯一的 Ref2VA graph；loader 由 API 的
`model_profile` 在 queue 前決定：

- `q4`（RTX 4060 default）：`MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf` + Q4_K_M text encoder
- `q2`（OOM fallback）：沿用 Q4 Ref2VA diffusion，只把 text encoder 切成 Q2_K
- `native`（quality A/B）：`minimax_h3_ref2va_pruned_int8_convrot.safetensors` + native text encoder
- H3 video/audio VAE → `models/vae`
- `VHS_LoadVideoPath` → `ComfyUI-VideoHelperSuite`

Ref2VA render 前會 preflight `MiniMaxH3ReferenceToVideo`；若 manifest 中有 video，還會 preflight `VHS_LoadVideoPath`。缺 node 時會在 queue 前失敗，不會產生一個看似成功但缺 reference 的結果。
Reference video 先以 24 fps 讀入，讓 H3 VAE 保留時間訊息；H3 的 Qwen reference presentation 會在 node 內另外以 2 fps 取樣，不要在 loader 層先降成 2 fps。

## Workflow tuning order

建議每次只改一個軸，並保留 run summary：

1. 先固定 608×352、124 frames、20 steps。
2. 先用 1 identity image；確認角色後再加 1 motion/camera video。
3. identity 漂移時，先提高 identity reference 的明確 retention，不要先增加 reference 數量。
4. 動作不對時，加入 `role: motion` 或 `role: camera` video；不要把 style image 當 motion reference。
5. VRAM 壓力高時，先減少 reference 數量、length、steps，再考慮升高解析度。
6. 只在 approved last frame 穩定後才用 L2VA；只在 reference lineage 完整後才用 Ref2VA。

## 相關程式位置

- manifest／lineage／prompt：`agentic/src/agentic/h3_reference.py`
- H3 asset profile：`agentic/src/agentic/assets/minimax_h3.py`
- Comfy slot binding：`agentic/src/agentic/tools/comfy_workflow_tool.py`
- planner：`agentic/src/agentic/runtime/planner.py`
- L2VA／Ref2VA skills：`agentic/src/agentic/skills/longvideo.py`
- approved last-frame gate：`agentic/src/agentic/skills/agent_primitives.py`
