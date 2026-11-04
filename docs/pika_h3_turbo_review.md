# Pika H3 Flex Turbo 本機驗收與審查

## 結論

本機 Pika-only 4-step 工作流已在 832x480 產生三支影片。提高到官方 480p bucket 後，紙船主體比 608x352 舊輸出完整、邊緣也更清楚；但這仍是 Q4 GGUF 底模上的 Turbo LoRA 工作流，並非 Pika 官方六步 Flex 權重路徑。現有證據不足以取代原版 H3 作為高保真預設；保留 Pika 作快速草稿與迭代路徑。

本次高解析驗收只執行 Pika，沒有重跑原版。三支輸出可驗證 Pika 工作流與媒體交付契約，不能當成 832x480 的原版畫質 A/B。整理文件後又經既有 `comfy.render_h3` 路徑即時重跑一支，231.82 秒完成，實際輸出仍為 832x480、124 frames、24 fps、AAC stereo。

若 Pika LoRA 不在既有模型根目錄，可設定 `COMFYUI_ASSET_EXTRA_ROOT` 指向另一個 ComfyUI 模型根目錄；資產檢查會一併搜尋其中的 `models/loras`。實測影片、contact sheet 和 summary 保存在本機被 `.gitignore` 排除的 `output/` 目錄，沒有隨 PR 提交；以下產物路徑只供該工作站追查，GitHub 上無法預覽。

## 被測工作流

| 設定 | 值 |
|---|---|
| Workflow | `minimax_h3_pika_turbo_t2va` |
| Diffusion base | `minimax_h3_fl2va_pruned_fp8_Q4_0.gguf` |
| Text encoder | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` |
| Adapter | `pika_h3_flex_turbo_4step_fl2va.safetensors`，strength 1.0 |
| Adapter pin | revision `dcacd93a50f714ba1cd61c7ca2cc5f65e9a1ec60`，SHA256 `8b0b5d2c53e435e637ed80afeff540135b90603ae53906f6aa8e66066776179d` |
| Sampling | 4 steps、`res_multistep`、CFG 1、video/audio shift 6/3 |
| Canvas | 832x480、124 frames、24 fps |
| Audio | AAC stereo，技術 metadata 驗證；未做主觀聽感評分 |

官方 Diffusers 文件把 832x480 列為 Pika H3 Flex 的 480p 16:9 bucket；官方模型卡稱 Flex 為六步生成。本機工作流使用 Turbo 4-step LoRA 和量化 Q4 GGUF，因此與官方六步 Flex 推論配方不同。[Pika H3 Flex 模型卡](https://huggingface.co/Pika-Labs/Pika-H3-Flex) · [官方解析度 bucket 與推論文件](https://huggingface.co/docs/diffusers/main/api/pipelines/pika_h3_flex) · [Turbo LoRA 權重](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Turbo-LoRA)

## 實測結果

| 案例 / seed | Pika 832x480 時間 | 畫面抽查 |
|---|---:|---|
| paper boat rain / 10303 | 250.14 秒 | 紙船、城市水道與黃葉清楚；比舊低解析輸出少了重疊船形，但船的位移和雨勢表現仍偏弱。 |
| woodland bike jump / 10304 | 237.09 秒 | 騎士與越過倒木的動作可辨識；接觸表不能充分確認落地後持續騎行。 |
| cafe latte / 10305 | 227.49 秒 | 拉花與遞杯場景大致符合 prompt，雙人構圖可辨識。 |

平均生成時間為 238.24 秒。三支皆成功；本機報告 `output/pika_turbo_480p_832x480_20261104_073825/pika_only_report.json` 的實際媒體欄位均為 832x480、124 frames、24 fps、AAC stereo。同目錄另有 `pika_turbo_832x480_review_contact_sheet.jpg`、`paper_boat_rain_sequence_review.jpg` 和 `paper_boat_resolution_comparison.jpg`。

文件整理後的單支即時 smoke 於同一 ComfyUI 環境、使用 seed 10303 完成，耗時 231.82 秒。影片、五時點畫面抽查和 ComfyUI summary 位於 `output/pika_turbo_workflow_smoke_20261104/paper_boat_seed_10303/`。畫面可見單一紅紙船、水道和黃葉，鏡頭追蹤清楚；紙船相對位移小，雨勢不明顯。主觀畫面檢查與媒體 metadata 是兩項分開的驗收。

解析度對照使用相同紙船 prompt、seed 10303、Pika Turbo 4-step，只比較兩次 Pika 輸出：舊 608x352 為 214,016 pixels，新 832x480 為 399,360 pixels，多 86.6%。比例分別約 1.727 與 1.733，兩者很接近；舊尺寸的主要問題是沒有使用官方驗證的 480p bucket，不是比例嚴重錯誤。此對照顯示高解析改善了主體完整度與細節，但改尺寸也會改變生成構圖，不能只解讀成放大同一畫面。

### 歷史 matched benchmark

2026-11-03 的三組同 prompt、同 seed、同 608x352、同 124 frames / 24 fps 實測中，原版 H3 用 16 steps，Pika Turbo 用 4 steps；兩者共用 Q4 GGUF base 與 Q4_K_M text encoder。

| 案例 | 原版 H3 | Pika Turbo |
|---|---:|---:|
| paper boat rain | 327.43 秒 | 108.03 秒 |
| woodland bike jump | 323.01 秒 | 117.35 秒 |
| cafe latte | 326.85 秒 | 109.47 秒 |
| 平均 | 325.76 秒 | 111.62 秒 |

低解析度的這組 matched run 中，Turbo 平均快 2.92 倍（生成時間少約 65.7%），但相同 seed 不代表輸出內容或品質相同。該組影片與原始 metadata 保存在本機 `output/h3_pika_turbo_comparison_20261103_232641/`；本次沒有再提交原版工作流。

## 畫質落差的審查判斷

1. **舊輸出尺寸低於官方 bucket。** 608x352 的比例接近官方 832x480 bucket，但像素量少 46.4%；同 prompt、同 seed 的 Pika-only 對照中，低解析紙船主體出現重疊形狀，832x480 的船體較完整。因此舊畫面偏糊和主體破碎，尺寸是有證據支持的原因之一。
2. **Turbo 的四步品質不能視為無損輸出。**「無損」融合 adapter 的說法不代表影片和原始 16-step H3 逐幀相同。歷史 matched benchmark 的速度優勢來自 4-step Turbo，但實測畫面有構圖及 prompt fidelity 差異。
3. **目前測的是量化底模上的 Turbo adapter。** Q4 base 在歷史原版與 Turbo matched run 相同，所以 Q4 無法單獨解釋兩者相對落差；但本機結果也沒有驗證官方六步 Flex adapter 在 BF16 base 上的畫質，不能把此工作流的結果泛化成「Pika Flex 本身畫質差」。
4. **提高解析度有時間成本。** Pika 832x480 平均約 238 秒，舊 Pika 608x352 平均約 112 秒。這兩次是不同解析度的分批實跑，不是重複統計；解析度以外的執行環境差異也未控制，所以不把 2.13 倍當作嚴格基準。

**決策：**保留目前 Pika 4-step 工作流供草稿與速度優先場景使用，不切換原版 H3 預設。若要判斷能否替代，下一個有效比較應在 832x480 固定 prompt、seed、Q4 base 和交付規格，另外測 Pika 官方六步 Flex 或 Turbo 8-step HD；本文件不宣稱那些路徑已完成驗收。

## 重跑方式

透過既有 `comfy.render_h3` 工具提交下列 payload；將輸出資料夾換成新的目錄即可重跑。不要在這次 Pika-only 驗收中選用 `minimax_h3_t2va`。

```json
{
  "workflow_name": "minimax_h3_pika_turbo_t2va",
  "h3_mode": "t2va",
  "run_dir": "output/pika_turbo_rerun/paper_boat_rain",
  "prompt": "A red paper boat drifts through a rain-fed city gutter at blue hour, passing yellow leaves. A low tracking camera follows it. Sound: rain, soft water, and distant traffic; no dialogue or music.",
  "width": 832,
  "height": 480,
  "length": 124,
  "fps": 24,
  "steps": 4,
  "seed": 10303,
  "model_profile": "q4",
  "adapter_profile": "pika_h3_flex_turbo_4step_fl2va",
  "video_count": 1
}
```

提交前要確認 ComfyUI 可連線，以及 Q4 模型、VAE、Turbo LoRA 都已就緒。完成後讀取 `comfy_h3_summary.json` 的 `actual` 與 `saved_files`，再人工檢查動作、構圖和聲音；metadata 成功只代表檔案規格正確，不代表畫面品質通過。
