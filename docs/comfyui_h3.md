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

T2VA、I2VA、FL2VA 與 L2VA 使用 FastVideo 的 FastH3 8 步蒸餾：INT8 ConvRot diffusion、NVFP4 AWQ Qwen3-VL 32B text encoder（Ada 顯卡以模擬運算執行）、`BlockSparseAttention` VSA（保留 10%）與 video shift 10／audio shift 3，由 `BasicScheduler` 固定跑 8 步。蒸餾模型只在訓練步數有效，因此這四種模式拒絕自訂 `steps` 與 `model_profile`。Ref2VA 使用其專用 Q4 GGUF diffusion，外掛 lightx2v 的 8 步 turbo LoRA（`LoraLoaderModelOnly`，強度 1，video shift 6／audio shift 3，`BasicScheduler` simple、euler 取樣），因此同樣只接受 8 步；仍以 `q4`／`q2` profile 選擇 Qwen3-VL 32B text encoder 的量化版本。video VAE 使用 FP16、audio VAE 使用 ComfyUI 的 fused FP32 權重。資產大小與相對目錄由 `agentic/src/agentic/assets/minimax_h3.py` 宣告。

低顯存服務應保留動態顯存管理。這些是 ComfyUI 服務的啟動設定，應依已安裝版本與硬體驗證，不由排程程式啟動或修改另一個 GPU 環境。

本機 RTX 4060 8GB（128GB RAM）實測：FastH3 5 秒 608x352 片段暖機後約 72 秒（原 16 步 Q4 GGUF 約 290 秒）、15 秒片段約 277 秒，VRAM 峰值約 7.3GB、系統 RAM 峰值約 69GB。首次載入需額外數十秒；盲測品質與原版互有勝負。Ref2VA 15 秒（362 幀）、512x640、3 張參考圖、q2 text encoder：原 16 步約 75 分鐘（取樣約 73 分鐘），改用 8 步 turbo LoRA 後 33 分 34 秒（取樣約 32 分鐘，約快 2.3 倍）。這是單次實測，未做同輸入的 16 步畫質對照，也未測 q4 text encoder 搭配 LoRA。

## 渲染前後重啟 ComfyUI

每次 `comfy.render_h3` 在連線前與渲染結束後（成功或失敗都做），各經 ComfyUI-Manager 的 `POST /v2/manager/reboot` 重啟一次 ComfyUI，等新程序回應 `/system_stats` 才繼續；Manager 拒絕、程序未重啟或逾時未回來都直接失敗，不在舊程序上渲染。因此 ComfyUI 必須以 `--enable-manager` 啟動，且 Manager 允許 reboot（監聽 loopback，或 Manager 設定 `network_mode = personal_cloud`，安全等級為 normal 以下）。每次重啟實測 31.3 秒，Manager 會自動拿掉 `--windows-standalone-build`，其餘啟動參數保留（reboot 端點、放行條件與參數處理依 ComfyUI-Manager 4.2.2 原始碼）。「舊程序已關閉」以連續多次探測失敗判定，單次斷線不算重啟完成。

原因：本機（RTX 4060 8GB、Windows）反覆藍屏，2026-11-09 至 11-27 共 7 份 dump 皆為 `dxgmms2.sys` 的 VidMm page-in use-after-free（`ExpandLargePagePte`，與 [ComfyUI#16246](https://github.com/Comfy-Org/ComfyUI/issues/16246) 同堆疊）。對照執行紀錄，觸發點是「大模型在動態顯存下載入、而先前的大模型仍常駐於同一個程序」，方向不固定：Krea2 之後載入 H3（11-09、11-23、11-26）、H3 之後載入 Krea2（11-27 受控重現，基準第 1 次迭代即藍屏，ComfyUI log 停在 `Requested to load Krea2` 後約 3 秒）、同一程序連續約 15 次 H3 之後（11-20）、以及只有圖片工作的 Krea2 節點內（11-21，當時沒有任何 H3 渲染）。因此 H3 前後各重啟一次，讓 H3 永遠單獨活在自己的程序裡；11-20 與 11-21 不一定被這個規則涵蓋（見下方「未涵蓋」）。

受控重現（`Krea2 → H3 → Krea2 …` 交替、每步 fsync 日誌）：不重啟的基準在第 1 次迭代藍屏；H3 前後各重啟的版本連續 40 次迭代、80 次模型切換沒有藍屏。過去不重啟時約每 12 至 15 次 H3 渲染一次藍屏，80 次乾淨切換純靠運氣的機率低於 1%，所以這是強的支持證據，但基準只有 1 次樣本、也只在本機這一張卡上驗證；關掉 dynamic VRAM 或 Comfy compiler 也能避開，但 H3 渲染會慢 5 至 10 倍，因此不採用。

未涵蓋：圖片模型之間的切換（例如 Krea2 與 Klein，11-21 的情形）沒有測試，也不重啟；並行限制見下。

代價：每次 `comfy.render_h3` 多約 62 秒，外加重新載入模型。serial_video 每集 3 至 4 段連續 H3，重啟的效益只來自「H3 之後有別的模型要載入」，連續 H3 之間的重啟是為了讓每次 H3 都從乾淨程序開始。

並行限制：同一個 run 內所有節點循序執行，NVIDIA VFX 也是循序節點，不會被重啟切斷；但跨程序（排程器、手動 CLI、ComfyUI 網頁介面）沒有鎖，重啟前不檢查佇列，會殺掉其他來源正在跑的工作。`execute` 內的接線需要真實 H3 資產，只有重啟契約有本機測試，尚未以真實 H3 渲染驗收。

## 輸入與交付

H3 使用 24 FPS、有效 CFG 1、8 步；FastH3 模式為 res_multistep、video shift 10／audio shift 3，Ref2VA 為 euler、video shift 6／audio shift 3。影片模式會依官方欄位組裝場景、環境音與配樂方向；I2VA／FL2VA／L2VA 另標明圖片在時間軸上的位置。長影片先用場景草稿製作錨點圖，再由 vision model 讀取實際首尾錨點寫出 H3 場景 prompt。此流程不含插幀或超解析度。使用者指定的 seed 保留在請求紀錄；Ref2VA 另保留模型 profile。

模型原生長寬對齊 32、幀數對齊 17k+5；交付沿用請求的長寬與幀數，必要時在完整生成時間線取樣，保留首尾錨點及音訊。既有的 2x 播放、canvas、QA 與包裝節點仍在上層技能圖處理。

圖片與影片參考素材經 ComfyUI `/upload/image` 上傳；參考影片由 `LoadVideo`／`GetVideoComponents` 只取影像，不接入其音軌。非法條件、無效 prompt、缺檔或損壞素材會在提交 GPU 前拒絕。

每次工作保留 `comfy_h3_request_*.json`、`comfy_h3_summary.json`、原生 MP4 及交付影片。summary 包含條件檔 SHA256、模型 profile、實際長寬／幀數／FPS 與音軌 metadata。

產生影片在最終交付前會經 NVIDIA VFX 後製：3x ULTRA 放大、2x MEDIUM 補幀，並保留原始音軌及片長。此階段在 canvas／播放速度調整之後、技術 QA 之前執行；設定與 ComfyUI 安裝方式見 [NVIDIA VFX 後製](comfyui_nvidia_vfx_postprocess.md)。

## 驗證

```powershell
python -m pytest agentic/tests/test_comfy_h3.py agentic/tests/test_comfy_restart.py agentic/tests/test_h3_modes.py agentic/tests/test_h3_prompt_delivery.py -q
python scripts/run_h3_modes_e2e.py --comfy-root <ComfyUI安裝目錄> --output-root <驗收輸出目錄> --mode t2va --smoke
```

受控 GPU 驗收與本機契約測試分開記錄；本機通過不代表 LLM、Discord、發布或自然排程已驗收。行為契約見 [H3 驗收](comfyui_h3_acceptance.md)。
