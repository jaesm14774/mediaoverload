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

ComfyUI 需要原生 H3 conditioning／audio-video 節點及新版 ComfyUI-GGUF 的 `UnetLoaderGGUFDynamicVRAM`、`CLIPLoaderGGUFDynamicVRAM`。執行前只查詢本次 workflow graph 使用的節點型別；缺少節點時直接回報，不切換其他 loader 或模型。

T2VA、I2VA、FL2VA 與 L2VA 使用 pi-H3 2 步蒸餾：MiniMax H3 FL2VA 基礎模型（INT8 ConvRot）外掛 `Load LakonLab Adapter` 載入的 2 步 DWD adapter、NVFP4 AWQ Qwen3-VL 32B text encoder（Ada 顯卡以模擬運算執行）、Spark 稀疏注意力（`MiniMaxH3SparkAttentionSM89`，`speed` 模式）與 video shift 12／audio shift 3，由 `pi-Flow Scheduler` 固定跑 2 步（max_raw_t 0.93）、euler 取樣。DWD 在最後一步直接預測像素，由 `DWD Decode` 輸出影像，不經 video VAE 解碼（video VAE 只用來編碼首尾幀）。蒸餾模型只在訓練步數有效，因此這四種模式拒絕自訂 `steps` 與 `model_profile`。需要 ComfyUI 0.45.0 以上、[ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) 與 [Spark-H3](https://github.com/zechengtang/Spark-H3)（Windows、SM89、cu130 套件）；Spark 節點與 GPU 架構綁定，換顯卡須換成對應的 SM 版本節點。內嵌 Python 的 `python_embeded` 必須有 `Include/` 與 `libs/`，否則 Spark 的 Triton 即時編譯在 10 秒以上的長片會失敗。Ref2VA 使用其專用 Q4 GGUF diffusion，外掛 lightx2v 的 8 步 turbo LoRA（`LoraLoaderModelOnly`，強度 1，video shift 6／audio shift 3，`BasicScheduler` simple、euler 取樣），因此同樣只接受 8 步；仍以 `q4`／`q2` profile 選擇 Qwen3-VL 32B text encoder 的量化版本。video VAE 使用 FP16、audio VAE 使用 ComfyUI 的 fused FP32 權重。資產大小與相對目錄由 `agentic/src/agentic/assets/minimax_h3.py` 宣告。

低顯存服務應保留動態顯存管理。這些是 ComfyUI 服務的啟動設定，應依已安裝版本與硬體驗證，不由排程程式啟動或修改另一個 GPU 環境。

本機 RTX 4060 8GB（128GB RAM）實測（608x352，T2VA，含音訊）：pi-H3 2 步 5 秒片段約 27–39 秒、10 秒約 54–57 秒、15 秒約 96–220 秒（單次最慢 465 秒，原因未明，兩個 ComfyUI 同時各鎖約 52GB 記憶體時更慢）；同機同條件下原 FastH3 8 步為 5 秒約 88 秒、10 秒約 215 秒、15 秒約 330 秒。VRAM 峰值約 7.5GB、系統 RAM 峰值約 48GB。5 秒片段的 token 數低於 Spark 的 `min_tokens`，不會啟動稀疏注意力，加速只來自步數。3 組不同方向的盲測沒有分出勝負（見 `output/pih3_ab/RESULTS.md`）。Ref2VA 15 秒（362 幀）、512x640、3 張參考圖、q2 text encoder：原 16 步約 75 分鐘（取樣約 73 分鐘），改用 8 步 turbo LoRA 後 33 分 34 秒（取樣約 32 分鐘，約快 2.3 倍）。這是單次實測，未做同輸入的 16 步畫質對照，也未測 q4 text encoder 搭配 LoRA。

## 輸入與交付

H3 使用 24 FPS、有效 CFG 1；pi-H3 模式為 2 步、euler、video shift 12／audio shift 3，Ref2VA 為 8 步、euler、video shift 6／audio shift 3。影片模式會依官方欄位組裝場景、環境音與配樂方向；I2VA／FL2VA／L2VA 另標明圖片在時間軸上的位置。長影片先用場景草稿製作錨點圖，再由 vision model 讀取實際首尾錨點寫出 H3 場景 prompt。此流程不含插幀或超解析度。使用者指定的 seed 保留在請求紀錄；Ref2VA 另保留模型 profile。

模型原生長寬對齊 32、幀數對齊 17k+5；交付沿用請求的長寬與幀數，必要時在完整生成時間線取樣，保留首尾錨點及音訊。既有的 2x 播放、canvas、QA 與包裝節點仍在上層技能圖處理。

圖片與影片參考素材經 ComfyUI `/upload/image` 上傳；參考影片由 `LoadVideo`／`GetVideoComponents` 只取影像，不接入其音軌。非法條件、無效 prompt、缺檔或損壞素材會在提交 GPU 前拒絕。

每次工作保留 `comfy_h3_request_*.json`、`comfy_h3_summary.json`、原生 MP4 及交付影片。summary 包含條件檔 SHA256、模型 profile、實際長寬／幀數／FPS 與音軌 metadata。

產生影片在最終交付前會經 NVIDIA VFX 後製：3x ULTRA 放大、2x MEDIUM 補幀，並保留原始音軌及片長。此階段在 canvas／播放速度調整之後、技術 QA 之前執行；設定與 ComfyUI 安裝方式見 [NVIDIA VFX 後製](comfyui_nvidia_vfx_postprocess.md)。

## 驗證

```powershell
python -m pytest agentic/tests/test_comfy_h3.py agentic/tests/test_h3_modes.py agentic/tests/test_h3_prompt_delivery.py -q
python scripts/run_h3_modes_e2e.py --comfy-root <ComfyUI安裝目錄> --output-root <驗收輸出目錄> --mode t2va --smoke
```

受控 GPU 驗收與本機契約測試分開記錄；本機通過不代表 LLM、Discord、發布或自然排程已驗收。行為契約見 [H3 驗收](comfyui_h3_acceptance.md)。
