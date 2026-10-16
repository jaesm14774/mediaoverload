# MiniMax H3 工作流

Krea 生圖、修圖與 H3 影片共用同一個 ComfyUI HTTP 服務。應用程式不啟動 GPU 子程序；ComfyUI 負責模型載入、顯存管理、推論及原生音畫輸出。

| 模式 | 工作流 | 輸入契約 |
|---|---|---|
| T2VA | `minimax_h3_t2va` | 只有文字，不接收首尾圖 |
| I2VA | `minimax_h3_i2va` | 已生成或審查通過的首圖 |
| FL2VA | `minimax_h3_fl2va` | 首圖與尾圖 |
| L2VA | `minimax_h3_l2va` | 只有尾圖 |
| Ref2VA | `minimax_h3_ref2va` | 有序圖片與可選參考影片，不接收參考音軌 |

各工作流都是 ComfyUI API node graph。資產、預設值與 conditioning recipes 放在 `configs/workflow/workflow_config.yaml`。`comfy.render_h3` 以同一個 ComfyAdapter 提交；策略、故事規劃、候選圖選擇、人工審查、編輯、QA 與發布仍使用既有技能圖。

## 服務與模型

設定 `COMFYUI_HOST`、`COMFYUI_PORT` 與 `COMFYUI_ROOT`，或從既有入口傳入對應參數。`COMFYUI_ROOT` 必須是實際安裝目錄或容器內掛載位置；工作流只記錄模型檔名，不綁定磁碟機代號。

ComfyUI 需要原生 H3 conditioning／audio-video 節點及新版 ComfyUI-GGUF 的 `UnetLoaderGGUFDynamicVRAM`、`CLIPLoaderGGUFDynamicVRAM`。執行時會查詢 `/object_info`，缺少節點時直接回報，不切換其他 loader 或模型。

FL2VA Q4 diffusion 用於 T2VA、I2VA、FL2VA 與 L2VA；Ref2VA 使用其專用 Q4 diffusion。既有 `q4`／`q2` profile 指定 Qwen3-VL 32B text encoder 的量化版本，diffusion 仍為 Q4。video VAE 使用 FP16、audio VAE 使用 ComfyUI 的 fused FP32 權重。資產大小與相對目錄由 `agentic/assets/minimax_h3.py` 宣告。

低顯存服務建議啟用 BF16 diffusion/text 與 SageAttention，並保留動態顯存管理。這些是 ComfyUI 服務的啟動設定，應依已安裝版本與硬體驗證，不由排程程式啟動或修改另一個 GPU 環境。

## 輸入與交付

H3 使用 24 FPS、res_multistep、有效 CFG 1、video shift 12／audio shift 3。原始場景描述直接傳入，不加入提示詞增強、插幀或超解析度。使用者指定的 seed、步數及模型 profile 保留在請求紀錄。

模型原生長寬對齊 32、幀數對齊 17k+5；交付沿用請求的長寬與幀數，必要時在完整生成時間線取樣，保留首尾錨點及音訊。既有的 2x 播放、canvas、QA 與包裝節點仍在上層技能圖處理。

圖片與影片參考素材經 ComfyUI `/upload/image` 上傳；參考影片由 `LoadVideo`／`GetVideoComponents` 只取影像，不接入其音軌。非法條件、無效 prompt、缺檔或損壞素材會在提交 GPU 前拒絕。

每次工作保留 `comfy_h3_request_*.json`、`comfy_h3_summary.json`、原生 MP4 及交付影片。summary 包含條件檔 SHA256、模型 profile、實際長寬／幀數／FPS 與音軌 metadata。

產生影片在最終交付前會經 NVIDIA VFX 後製：2x ULTRA 放大、2x MEDIUM 補幀，並保留原始音軌及片長。此階段在 canvas／播放速度調整之後、技術 QA 之前執行；設定與 ComfyUI 安裝方式見 [NVIDIA VFX 後製](comfyui_nvidia_vfx_postprocess.md)。

## 驗證

```powershell
python -m pytest agentic/tests/test_comfy_h3.py agentic/tests/test_h3_modes.py agentic/tests/test_h3_prompt_delivery.py -q
python scripts/run_h3_modes_e2e.py --comfy-root <ComfyUI安裝目錄> --output-root <驗收輸出目錄> --mode t2va --smoke
```

受控 GPU 驗收與本機契約測試分開記錄；本機通過不代表 LLM、Discord、發布或自然排程已驗收。行為契約見 [H3 驗收](comfyui_h3_acceptance.md)。
