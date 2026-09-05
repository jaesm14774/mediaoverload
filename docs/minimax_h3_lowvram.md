# MiniMax H3 + Kirby 低配影音管線

這個 repo 的預設 H3 profile 是為 RTX 4060 8 GB VRAM、約 128 GB system RAM 設計的。主路徑使用社群 GGUF 量化模型，影音生成仍走 ComfyUI upstream 的原生 MiniMax H3 nodes。

## 預設 profile

| Profile | 用途 | 下載量 | 工作流 |
| --- | --- | ---: | --- |
| `balanced-lowvram` | 預設；Q4 diffusion + Q4_K_M text encoder | 約 29.59 GiB | `minimax_h3_lowvram_i2v` |
| `ultra-lowvram` | balanced 初始化或反覆 OOM 時；只把 text encoder 降到 Q2_K | 約 23.92 GiB | `minimax_h3_lowvram_i2v` + `model_profile: q2` |
| `native-quality` | 官方 INT8 ConvRot + NVFP4 對照組 | 約 39.55 GiB | `minimax_h3_native_t2v` |

所有 profile 都固定從 608×352、124 frames（24 fps，約 5 秒）開始。這個尺寸是低配 draft 的起點；先完成構圖、角色辨識與動作，再提高解析度或關閉 Spectrum 做 final。

## 下載與啟動

```powershell
python scripts/setup_minimax_h3.py --profile balanced-lowvram --comfy-root D:\ComfyUI_windows_portable --json
powershell -ExecutionPolicy Bypass -File scripts/run_comfyui_h3_lowvram.ps1 -ComfyRoot D:\ComfyUI_windows_portable
```

模型會直接放在 portable ComfyUI 的 `ComfyUI\models` 對應資料夾。下載器使用 `.part` 暫存檔、Windows `curl.exe` retry/resume，以及最終 byte-size 驗證；重跑同一指令會沿用已完成或部分完成的檔案。

模型全部 ready 後，可以先跑單段驗證：

資產就緒後，正式產出一律從 repo 根目錄的 `run_media_interface.py`
進入，不使用獨立 H3 生成器或 smoke test。

正式入口會在 runtime 內準備並驗證 Kirby keyframe；不再使用獨立 smoke test 產物。

若 balanced profile 在你的本機初始化失敗：

```powershell
python scripts/setup_minimax_h3.py --profile ultra-lowvram --comfy-root D:\ComfyUI_windows_portable --json
```

本機 smoke 已驗證 fallback lowvram flags 可完成生成，而 dynamic VRAM 組合在 conditioning 階段 OOM；因此 launcher 預設使用 fallback。若要做 dynamic VRAM A/B：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_comfyui_h3_lowvram.ps1 -ComfyRoot D:\ComfyUI_windows_portable -DynamicMode
```

預設會使用 `--disable-dynamic-vram --disable-async-offload --lowvram`；`-DynamicMode` 才會保留 dynamic VRAM。ComfyUI 官方指出 dynamic VRAM 對大模型 offload 通常有幫助，但你的實機 A/B 已顯示 8 GB RTX 4060 在這組 H3 Q4/Q4 conditioning 會 OOM，所以保守模式是目前推薦值。

## Kirby 正確輸入與長片流程

Kirby 的短片 I2V 鏈路固定為：

```text
krea2_turbo
  -> Kirby input gate（pink/red silhouette + generic-example block）
  -> minimax_h3_lowvram_i2v
  -> extract last frame
  -> krea2_turbo_img2img（denoise 0.25）
  -> minimax_h3_lowvram_i2v
  -> ffmpeg concat
```

長片使用同一套 segment/concat 機制，production profile 以約 5 秒為一段，
例如 30 秒拆成 6 段、45 秒拆成 9 段。第一段使用已審核的 opening image 或
reference；後續段落固定用上一段實際渲染出的最後一幀進行
`minimax_h3_lowvram_i2v` 接續。只有需要明確抵達新狀態時才使用
`minimax_h3_lowvram_15s_fl2va_i2v`，不再以 T2V 片段或預設圖像代替真實畫面連續性。

執行完整兩段流程：

```powershell
python run_media_interface.py --character kirby --generation-type native_h3_story --comfy-host 127.0.0.1 --comfy-port 8188 --comfy-root D:\ComfyUI_windows_portable --no-publish
```

這個 script 會把通過驗證的第一張 keyframe 複製到 `D:\ComfyUI_windows_portable\ComfyUI\input\kirby_keyframe_seed.png`。若 H3 tool 收到 `example.png` 或沒有 Kirby pink/red signal 的圖，會直接拒絕，不會浪費數分鐘跑錯影片。

## Workflow 結構

```text
GGUF / native H3 loader
  -> MiniMaxH3SigmaShift
  -> SpectrumApplyMiniMaxH3（draft，history_storage=system_ram）
  -> BasicGuider + res_multistep sampler
  -> VAEDecode + VAEDecodeAudio
  -> CreateVideo（24 fps + native stereo audio）
  -> SaveVideo
```

I2V 由 Kirby keyframe 鎖定角色外觀；T2V 仍是獨立的單一短片探索路線，不參與
`text2longvideo` production path。long-video planner 將 H3 prompt builder 接到每個
segment，加入角色 identity lock、具體 story-state、動作方向與 native stereo audio
direction；每段結束後抽取真實尾幀，作為下一段的 first-frame conditioning。

## 在 ComfyUI 畫布查看完整 workflow

`configs/workflow/*.json` 原本是 Agentic 呼叫 ComfyUI `/prompt` 的 API graph，不是 ComfyUI 畫布檔，所以不能用它確認節點位置與 widgets。這次新增的原生畫布檔案是：


兩份也已同步安裝到：

```text
D:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\
```

開啟方式：

1. 啟動 `D:\ComfyUI_windows_portable\run_nvidia_gpu.bat` 或 repo 的 H3 launcher。
2. 開啟 `http://127.0.0.1:8188/`。
4. 先按畫布的 `適應視圖 (.)`，即可看到完整節點、連線與所有參數。

第一張是 repo 的 `krea2_turbo` / Kirby keyframe 生成接 H3 I2V；第二張是 repo 的 Krea2 img2img identity continuity 接 H3 I2V。長片 production route 使用審核過的 opening image、H3 I2V 與真實尾幀接續；FL2V 只處理明確的 landing state。若要換首幀，修改 Krea2 prompt 或 `Load Image`；H3 低配主參數集中在 `608×352`、`120 frames`、`16 steps`、`Spectrum history_storage=system_ram`、`24 fps`。

這兩張畫布 workflow 是以 repo 現有結構擴增而成，不會取代原本給自動化程式使用的 API graph。

Spectrum 是近似加速器，不是 lossless mode。預設只作 draft；需要品質對照時，改用 `minimax_h3_native_t2v` 或把 Spectrum node 的 `enabled` 設為 `false`，並比較同 seed 的結果。不要同一 model branch 同時疊加 EasyCache 與 Spectrum。

## 研究依據與限制

- [MiniMax H3 官方介紹](https://minimaxi.com/blog/minimax-h3)
- [ComfyUI H3 native support](https://github.com/Comfy-Org/ComfyUI/pull/15224)
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [ComfyUI-GGUF MiniMax H3 fork](https://github.com/molbal/ComfyUI-GGUF)
- [ComfyUI Spectrum MiniMax H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
- [ComfyUI Dynamic VRAM](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local)

Q4/Q2 檔案與 Spectrum 的速度/VRAM 數字是社群實測或模型檔案當日 metadata，不代表所有 driver、prompt、sampler 或解析度都能重現。這個 profile 優先追求「能在 8 GB 卡上穩定迭代」，再由 native-quality 做 final A/B，而不是宣稱低配與 full-quality 等價。
# MiniMax H3 + Kirby low-VRAM notes

## Verified duration boundary

The ComfyUI MiniMax H3 node accepts a wide frame-count input, but its own
tooltip documents approximately 124-362 frames as the trained range at 24 fps
(about 5-15 seconds). MediaOverload therefore uses 362 frames for the
production native H3 T2V/I2V recipes. A 20-second direct H3 request would be
482 frames and is rejected by the agentic render skill before it enters the
ComfyUI queue; this prevents a slow, unvalidated run from consuming the local
machine.

The production direct-H3 route uses
`configs/storyboards/native_h3_15s.yaml` as its timing and continuity
contract. It generates the plot and beats from the current news item at
runtime; there is no hidden duration fallback.
