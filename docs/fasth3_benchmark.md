# FastH3 by FastVideo 深度測試報告

日期：2026-09-18（Asia/Taipei）  
分支：`codex/fasth3-benchmark`  
GPU：NVIDIA GeForce RTX 4060 8 GiB  
測試 prompt、seed、畫布與輸出規格全部固定：

> A polished 2D anime short about Kirby discovering a tiny glowing seed in a windy meadow. [0s-2s] Kirby notices the seed rolling toward a cliff as grass bends in a strong gust. [2s-4s] Kirby runs, reaches out, and catches it just before it falls. [4s-5s] Kirby lifts the seed; warm light spreads across the meadow as the wind settles. Use a clear beginning, middle, and payoff, readable subject silhouette, deliberate camera movement, continuous motion, and native stereo audio with wind, a soft impact, and a warm uplifting musical resolve.

## 結論

FastH3 V2 可以在目前 8 GiB GPU 上完成現有 H3 T2AV 類型的 end-to-end 生成，而且不需要 GGUF。這次實際測的是官方 INT8 safetensors，不是 BF16 full-precision model；它和 GGUF baseline 的速度差異不是單純「8 steps 一定更快」：cold run 受到大型模型初始化、CPU/RAM offload、VAE decode 與輸出保存影響，FastH3 V2 反而比現有 GGUF cold run 慢；模型已常駐/可重用的 warm run 則略快。

FastH3 V2 適合新增成為快速 T2AV route，不適合直接取代目前所有 H3 route。現有 first/last-frame、Ref2VA、FL2VA 與已審核的內容策略應維持原契約。

單一 Kirby prompt 的初測沒有看到 FastH3 的品質勝出；後續 3 個不同 prompt × 3 個 route 的矩陣測試則顯示，FastH3 V2 在動作順序與物件互動的 prompt adherence 上有較穩定的優勢。這個更新後的判定與逐格證據見 [`fasth3_prompt_matrix.md`](fasth3_prompt_matrix.md)。

## 實測速度

所有 run 都是完整 ComfyUI queue execution，包含模型準備、sampling、video/audio VAE decode、CreateVideo 與 SaveVideo。數字不是只量 diffusion sampling。

| Route | Runtime | Steps | Cold run | Warm run | Peak observed VRAM | 結果 |
|---|---|---:|---:|---:|---:|---|
| 現有 GGUF baseline | current ComfyUI 0.30.0 / port 8188 | 20 | 291.598s | 3.109s | 7,912 MiB | 2/2 passed |
| Official MiniMax H3 INT8 | isolated ComfyUI 0.36.0 / port 8189 | 20 | 345.534s | 2.925s | 7,502 MiB | 2/2 passed |
| FastH3 V2 INT8 + VSA | isolated ComfyUI 0.36.0 / port 8189 | 8 | 323.961s | 2.901s | 7,461 MiB | 2/2 passed |

相對於現有 GGUF baseline：

- FastH3 V2 cold run 慢 32.363 秒，約慢 11.1%。
- FastH3 V2 warm run 快 0.208 秒，約快 6.7%；這個差異很小，不能視為穩定收益。
- FastH3 V2 的 peak observed VRAM 少 451 MiB，約低 5.7%；這是取樣期間觀察值，不是模型的絕對 VRAM 上限。
- Official native INT8 diffusion 是非 GGUF 路徑，但在這台機器上的 cold run 為 345.534 秒，慢於 FastH3 V2；FastH3 V2 相對它快 21.573 秒，約 6.2%。

FastH3 server log 顯示第一次初始化約 133.78 秒、第二次約 91.21 秒；8-step sampling 本身約 52.2 秒，每 step 約 6.6 秒。這說明目前瓶頸不只有 step 數，還包含 offload / model initialization。第二輪完整 E2E 為 2.901 秒，是 cache reuse 的結果，不應當作冷啟動部署 SLA。

## 輸出驗證與視覺結果

三組輸出都通過 ffprobe：`608x352`、`124 frames`、`24/1 fps`、`5.167s`、H.264 video、AAC stereo audio。

這次品質判定不是只看輸出檔是否存在，而是對每支正式輸出以 `ffmpeg` 每 0.5 秒抽一格，直接逐段對照同一個 prompt 的三個時間區間：`0-2s` 發現／追向種子、`2-4s` 追逐／接住、`4-5s` 舉起／暖光 payoff。檢視素材保存在 `E:\comfyui\_extra\benchmarks\fasth3_quality_review`。

### 品質與語意結論

直接回答「FastH3 V2 有沒有比較好」：以這組 controlled same-prompt、same-seed E2E 輸出來看，沒有。FastH3 V2 沒有在影片品質或語意理解上形成可確認的勝出，不能因為它使用 8 steps 就當成品質升級。

| 評估面向 | 現有 GGUF | Official native INT8 | FastH3 V2 INT8 + VSA |
|---|---|---|---|
| 高層語意 | 都理解 Kirby 接近發光種子並在結尾產生暖光 | 同左 | 同左 |
| `0-2s` 種子滾向懸崖、強風 | 懸崖與風格草地可讀，但種子「滾動」不明確 | 花草細節最多，但風／滾動也不明確 | 景深與構圖好，但種子形狀／顏色先出現漂移，風／滾動仍不明確 |
| `2-4s` 跑、伸手、接住 | Kirby 有清楚接近並靠向種子；「伸手」較像靠近／碰到 | 動作較保守，較像姿勢與鏡頭變化 | 動作較有動感，接近過程可讀；但沒有穩定呈現「伸手後在懸崖邊接住」 |
| `4-5s` 舉起、暖光、風停 | 暖光 payoff 最清楚，整段敘事最容易讀 | 暖光成立，畫面細節最好；舉起動作仍偏含蓄 | 暖光成立且構圖漂亮，但種子外觀漂移，舉起仍不比 baseline 清楚 |
| 角色／物件一致性 | Kirby 穩定，種子外觀相對穩定 | Kirby 與種子最穩定 | Kirby 尚可，但種子由綠／紫到粉綠的形狀與顏色變化較明顯 |
| 純畫面細節 | 乾淨、清楚，背景相對簡化 | 花草、光照與紋理最豐富 | 場景深度與構圖不錯，但不穩定物件削弱完成度 |

因此目前的品質排序不是「FastH3 > native > GGUF」：

- 若重視 prompt 的故事節點是否容易讀，GGUF baseline 反而最穩。
- 若重視單張畫面的花草、光照與紋理，native INT8 最漂亮，但不代表動作語意更好。
- FastH3 V2 的優點是 8-step route 與較有動感的畫面，不是這次測試中已證實的語意或整體品質提升。

### 三支輸出的人工檢視摘要

- GGUF baseline：Kirby、懸崖、發光種子和由風到暖光的場景語意最容易讀；動作節點較明確，結尾的暖光 payoff 清楚。
- Native INT8：色彩更鮮亮、花草細節較多；Kirby 與種子的關係可見，但中段動作幅度較保守，部分 frame 主要是姿勢/鏡頭變化。
- FastH3 V2：畫面較清爽，背景樹線與草地景深不同，動作比 native 更有變化；但種子顏色／形狀漂移較明顯，且「滾向懸崖／伸手接住」沒有比 baseline 更明確。

這仍然是單一 prompt、單一 seed 的人工品質審查，不足以宣稱所有題材都同樣排序；但它足以回答本次最重要的問題：這個 prompt 上沒有看到 FastH3 V2 的品質／語意優勢。三支影片都有 AAC stereo audio stream 並完成 E2E 封裝；本次沒有做真人聽感盲測，因此不把「有音訊軌」誤寫成「風聲、碰撞與音樂情緒已比較好」。若要正式選 production default，下一輪應用多個 prompt/seed 做盲測，並把 motion consistency、prompt adherence、音訊聽感與人審結果納入 gate。

## 模型與下載內容

下載根目錄固定為 `E:\comfyui\_extra\models`，五個檔案均已完成下載，下載器以 expected size + SHA-256 驗證後才 rename 成正式檔名。總大小約 70.8 GiB。

本次實測使用：

- FastH3 V2 INT8 diffusion：`fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`，約 22.13 GB。
- Official MiniMax H3 native INT8 diffusion：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`，約 20.97 GB。
- Shared MiniMax H3 INT8 text encoder：`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`，約 27.14 GB。
- Shared video/audio VAE：約 5.21 GB + 0.61 GB。

這不是 BF16 full model 測試。官方 repack 另有約 44.08 GB 的 FastH3 V2 BF16 diffusion 與約 51.51 GB 的 BF16 text encoder；在 8 GiB GPU 上不具備合理的本機 E2E 測試條件，因此沒有下載或宣稱 BF16 已測。這次的結論是「可用官方 INT8 safetensors 取代 GGUF」，不是「已驗證 BF16 full model」。

## Workflow / runtime 變更

- 新增 `minimax_h3_fasth3_v2_t2v.json`：官方 FastH3 V2 INT8 diffusion、MiniMax H3 INT8 text encoder、8 steps、VSA。
- VSA 依 ComfyUI 0.36 DynamicCombo API 契約使用 `selection="vsa"` 與 `selection.keep_percent=10`；巢狀 dict 會導致 `BlockSparseAttention.execute()` 收不到 selection。
- FastH3 V2 runtime log 顯示 `VSA: extra_tokens ignored (the trained sparse pattern is the target)`，因此 workflow 的 `extra_tokens=256` 是官方 schema 欄位，但實際 VSA pattern 由 trained sparse pattern 決定。
- 新增 `minimax_h3_native_int8_t2v.json`，用來隔離「native safetensors」與「FastH3 distillation/VSA」的效應。
- 兩個新 workflow 的 registry metadata 都指向 `E:/comfyui/_extra`，因此正式 workflow asset check 與 benchmark 使用同一個下載根目錄。
- 新增 resumable downloader、asset manifest、isolated staging launcher 與 benchmark runner；staging 放在 `E:\comfyui\_extra\fasth3_staging`，避免修改現有 main / 8188 runtime。

測試過程中，曾把現有 Spectrum custom node 複製到新版 ComfyUI staging，結果新版 `FinalLayer.forward` 契約與舊 node 不相容而失敗；這個 baseline 失敗被排除，正式 baseline 改在使用中的 8188 runtime 測量。這個 evidence 仍保留在 `E:\comfyui\_extra\benchmarks\fasth3\fasth3_benchmark.json`，沒有把相容性錯誤誤報成模型品質結論。

## 建議

1. 保留現有 GGUF route 作為 production baseline 與 fallback contract，不改 default routing。
2. 將 FastH3 V2 加入明確的 `fasth3_v2_t2av` strategy，只用於不需要 first/last-frame 或 Ref2VA conditioning 的短 T2AV 內容。
3. 若目標是降低 cold latency，優先研究單一常駐 ComfyUI process、model cache/offload policy、VAE reuse 與同一批次多 prompt；只把 steps 從 20 降到 8 並不能保證 cold E2E 變快。
4. production 前再用至少 10 組 prompt × 2 seeds 做 quality gate，檢查角色一致性、動作連續性、種子 payoff、音訊可聽性與 render failure rate。

後續多 prompt 結果請見 [`fasth3_prompt_matrix.md`](fasth3_prompt_matrix.md)：3 個新 prompt、3 個 route、9 支獨立 E2E 影片顯示 FastH3 V2 在動作順序與因果語意上比單一 Kirby 測試更有優勢，但仍不是無條件的畫質升級。

## 可重現指令

```powershell
$env:PYTHONPATH = "agentic\src"

# 下載/驗證官方資產
python scripts/download_fasth3_assets.py --status --json

# FastH3 V2：8189 isolated runtime
python scripts/run_fasth3_benchmark.py `
  --case fasth3_v2 --repeats 2 --skip-asset-hash `
  --output-root E:\comfyui\_extra\benchmarks\fasth3_opt_v2

# 現有 GGUF：8188 current runtime
python scripts/run_fasth3_benchmark.py `
  --case baseline_gguf --repeats 2 --skip-asset-hash `
  --comfy-port 8188 --comfy-root D:\ComfyUI_windows_portable `
  --output-root E:\comfyui\_extra\benchmarks\fasth3_baseline_retry
```

原始研究來源：

- [FastVideo repository](https://github.com/hao-ai-lab/FastVideo)
- [FastH3 V2 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)
- [FastH3 Comfy repack](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy)
- [Official ComfyUI FastH3 workflow template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_fastvideo_fasth3_t2v.json)
- [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
