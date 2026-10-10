# Krea 生圖耗時調查與優化驗證

調查日期：2026-10-09 至 2026-10-10；機器：Windows、RTX 4060 8 GB。

## 結論與已採用方案

Krea 2 Turbo 生圖與續圖已改用 **原生 ConvRot W4A4** checkpoint，透過
ComfyUI 內建 `UNETLoader` 執行。維持原本提示詞、8 步、尺寸與六張候選
契約；沒有用減少候選圖、降低步數或縮圖來取得速度。

正式 `comfy.render_image` 工具入口重測六張 512×640，清除 ComfyUI 模型／
節點快取後，整批 **107.596 秒**，各張 provider 時間依序為
**46.855、9.985、9.964、11.263、9.864、17.144 秒**。六張 PNG 皆有效，
seed 依序遞增，沒有 OOM 重試。原 production run 同一階段約耗時 109 分鐘。
這是實際本機結果，不能當作所有負載與角色的速度保證。

## 驗收情境

下列情境於相關實作前記錄：

- User Given 多張候選圖共用模型與提示詞 When 依序產生不同 seed Then 每張存檔且模型快取可保留。
- User Given ComfyUI 閒置且保留生圖模型 When WanGP 開始影片生成 Then 先釋放 ComfyUI 模型再載入 H3。
- User Given ComfyUI 已有工作 When 診斷 benchmark 啟動 Then 拒絕排入或中斷原工作。
- User Given 每張圖 210 秒的目標 When 比較模型 Then 記錄實際產物與耗時，不能拿廣告中的硬體速度代替。
- User Given Krea 圖片工作流程 When 執行 T2I 或續圖 Then 使用同一原生 checkpoint 及一致的資產 manifest，保留提示詞、8 步與人工審核契約。
- User Given 清除 ComfyUI 快取 When 產生 1024×576 圖及正式續圖 Then 有效產物於 210 秒內完成才採用。

## 原 run 為什麼慢

Run `81a8ec92f6ed` 於 10/09 22:19:44 開始，22:20:01 完成提示詞準備。
瓶頸在六張依序執行的 Krea 圖片，當時尚未開始 WanGP Ref2VA 影片。

| 候選 | 舊 GGUF provider 秒數 | 分鐘 |
|---|---:|---:|
| 1 | 1185.751 | 19.8 |
| 2 | 1121.419 | 18.7 |
| 3 | 211.644 | 3.5 |
| 4 | 1718.905 | 28.6 |
| 5 | 434.308 | 7.2 |
| 6 | 1841.820 | 30.7 |

六張除了 seed／輸出名稱外，皆使用同一份 4987 字元提示詞、512×640、
8 步、Euler/simple、CFG 1、denoise 1。不是某張誤用了更高解析度或更多步。
前五張在 benchmark 模型下載之前已完成；第六張與下載重疊，因此不把
第六張當作純粹且受控的模型效能比較。

可確認的瓶頸：

1. **舊 diffusion GGUF 執行方式與 offload。** 模型檔 8.31 GB，日誌顯示
   5608.67 MB 可用、5446.67 MB 載入、2669.18 MB offload。慢的時間主要在
   denoising，GPU 使用率高且 VRAM 接近滿載；GGUF forward 會解量化。
2. **每張前後無條件清快取。** 應用程式呼叫 `/free`，而目前 ComfyUI
   `free_memory: true` 會呼叫 `PromptExecutor.reset()`，模型與節點快取
   都被清除。這是舊 ComfyUI H3 時期留下的生命週期，WanGP 已有自己的
   影片交接釋放流程。
3. **其他 GPU 使用與小 VRAM。** 兩個 LDPlayer 程序仍在使用 GPU。
   全機 CPU RAM 約有 64 GiB 可用／共 128 GiB，不支持 RAM 耗盡的說法。
   GPU 約 48°C、時脈正常，當時沒有支持熱降頻的證據。

[NVIDIA 官方文件](https://nvidia.custhelp.com/app/answers/detail/a_id/5490)
說明 VRAM 不足時的 system-memory fallback 可能顯著變慢；但 WDDM shared
memory 數字不能證明本次已啟用該機制，唯讀 NVAPI 查詢也未確認設定。
沒有修改驅動設定、關閉 LDPlayer 或重啟 production run。

### 為什麼每張都清快取：追查歷史後的更正

`git log -S 'self.free_memory()'` 指向 2026-08-15 的 `ce9d6f3`。
該修改的註解說明：在 8 GB GPU 上，生圖模型可能佔住 H3 影片需要的
VRAM，因此要釋放模型。但實作放在共用 `process_workflow` 的送出前與
存檔後，沒有辨別是否真的切換圖片／影片模型。這使同一 Krea 連續六張
也每次遭到重置。**這是應用程式的釋放範圍錯誤，並非 GGUF 要求清快取。**

同模型／同提示詞連續生圖應保留模型及文字 conditioning；換 seed 仍要
重新算 diffusion，VRAM 不足也可能由 provider 自動 offload，保留 cache
不等於所有權重永遠留在 GPU。新版六張的首張 46.855 秒、後五張
9.864–17.144 秒，符合冷啟動後可重用快取的預期。

先前把 GGUF 第一張 240 秒逾時說成「單靠快取修正不足」，結論過強。
實驗在首張完成前已取消，**沒有測到第二張暖快取 GGUF**，不能判斷其
後續加速幅度。日誌顯示 diffusion 載入約 6.7 秒、第一 sampling step
約 122.7 秒，說明該次長時間不只花在讀模型檔；但仍不能排除首張初始化
成本或據此保證後續暖圖同樣慢。採用原生 W4A4 的理由是它已完成真實
六張入口與續圖驗收，並非已證明 GGUF 暖快取完全無效。

## 實際 GPU 比較

| 實驗 | 條件 | 秒數 | 結果 |
|---|---|---|---|
| 舊 GGUF，只移除每張清快取 | 512×640、8 步 | 首張超過 240 秒，當時 1/8 | 首張未完成、暖圖未測；只取消本次 benchmark prompt。 |
| Krea 原生 W4A4 | 512×640、8 步，既有 server cache | 15.617 / 18.747 | 角色特徵通過目視檢查。 |
| Krea 原生 W4A4 | 清除 Comfy 快取後，1024×576、8 步 | 49.303 / 17.469 | 有效 PNG，超過原尺寸仍低於 210 秒。 |
| 正式 `comfy.render_image` | 清除 Comfy 快取後，512×640 六張 | 整批 107.596 | 每張 9.864–46.855 秒，六張存檔，0 OOM 重試。 |
| 正式 `comfy.workflow.image_to_image` | 512×640、8 步、denoise 0.25 | 18.892 | 有效續圖，0 OOM 重試。 |
| FLUX.2 Klein 4B FP8 | 同 production prompt／前兩個 seed、512×640，原生 4 步 | 50.368 / 9.922 | 速度通過；角色外型不通過，未改 production routing。 |

「清除快取」指 ComfyUI 模型／節點快取，不包含 Windows 檔案快取，也
不是重啟整個 ComfyUI。LDPlayer 在以上實驗中仍開著。原生 W4A4 的 fused
quantized compute 與 dynamic VRAM patcher 一起改變，這些結果不能分離
兩者各自貢獻的比例。

六張正式候選中，1–4、6 可辨識橘色圓身、無嘴淺色臉、短黃腳、藍頭巾
與長矛；第 5 張背對鏡頭，臉部特徵無法確認，需要原有人類審核挑選。
不宣稱所有 seed／角色與舊量化逐像素相同，也沒有跳過審核。
Klein 兩張則有嘴、類人肢體／身材及錯誤頭巾紋樣，不能為了速度直接
取代這條角色路線。

## 上網研究的模型選項

| 模型／版本 | 查證結果與本機判斷 |
|---|---|
| Krea 2 Turbo 原生 W4A4 | 已採用。checkpoint 約 8.06 GB，比舊 GGUF 僅小約 3%，改善主要是執行方式而非模型縮小。作者提供內建 UNETLoader 的 no-low-rank 版本與 3090 實測；本次自行驗證 4060，沒有套用作者速度宣稱。[作者實作](https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI)、[checkpoint](https://huggingface.co/AlperKTS/Krea-2-SVDQuant-ComfyUI)。 |
| Krea 2 Raw／更大 INT8 格式 | Raw 使用 52 步，Turbo 是 8 步。Raw 不是加速版，較大的 native INT8 檔也不代表 8 GB 可以免 offload。[官方設定](https://github.com/krea-ai/krea-2)、[Comfy 官方格式](https://docs.comfy.org/tutorials/image/krea/krea-2)。 |
| FLUX.2 Klein 4B distilled | 4 步、FP8 diffusion 檔 4.07 GB、支援生成及參考圖編輯，4B Apache 2.0。適合一般場景速度比較，但本次角色失真。Comfy 的 5090 1.2 秒／8.4 GB 與 BFL reference 約 13 GB 是不同配置，均不是本機保證。[BFL model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)、[Comfy guide](https://docs.comfy.org/tutorials/flux/flux-2-klein)。 |
| Anima Turbo v1.1 | 2B 插畫模型、8–12 步、CFG 1；本機已安裝的是 Aesthetic 而非 Turbo。模型是 Non-Commercial license，輸出權利與模型部署條件需分開判斷；本次未做 GPU benchmark。[模型作者](https://huggingface.co/circlestone-labs/Anima)。 |
| Qwen-Image-2.1／Viggle Turbo v0.3 | 新版統一生成／編輯模型，7B visual backbone。9/29 的 v0.3 支援 6 步或 9 步 hybrid，作者承認小字／細節限制；Research License 的使用條件也不同。沒有本機速度或角色品質證據，因此沒有替換 production。[Qwen](https://huggingface.co/Qwen/Qwen-Image-2.1)、[Viggle 作者](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)。 |
| Hosted Krea 2 Medium Turbo／Medium／Large | 官方列約 3／10／25 秒與 USD 0.015／0.03／0.06 每張。是另一種付費部署選項，非本機 benchmark 或 SLA；沒有呼叫付費 API。[官方價目](https://www.krea.ai/app/api/pricing)。 |

[目前 GGUF fork](https://github.com/molbal/ComfyUI-GGUF) 明確不支援 diffusion
`_K` quant，不能看到更小的 Krea Q3_K_M／Q4_K_S 就直接替換。
文字 encoder 的 K quant 不受這條 diffusion 限制。RTX 50 的 NVFP4
加速數字也不能直接移植到 RTX 4060。

[WanGP 本身](https://github.com/deepbeepmeep/Wan2GP) 支援 Krea／Klein 等圖片
模型，但目前本專案整合的是其 H3 影片接口。**影片 diffusion 已用 WanGP；
圖片／續圖仍用 ComfyUI。** 這次選擇已在本機驗證的原生 Comfy 方案，
沒有為了後端名稱改寫另一套未驗證的生圖接口。

## 實作與測試

- 兩份 Krea workflow 的 diffusion loader 改成 stock `UNETLoader`，移除舊 diffusion GGUF 路徑。
- workflow asset manifest 更新 checkpoint 名稱、來源、目錄、大小與 SHA-256；兩條 route 的三項資產皆 ready。
- `comfy_backend.py` 移除每張前後清快取；保留明確的圖片 OOM cleanup 與 WanGP 交接。
- native recipe contracts 先跑出 3 個失敗再改設定，修正後 focused scope 63 通過。
- 前一輪全套 non-integration：**619 passed、1 skipped、1 deselected，118.49 秒**。
- 原本寫死舊 GGUF 檔名的 H3 測試改驗工作流程有效及續圖保留原圖內容的 denoise 邊界，沒有刪除有用的行為測試。
- 閒置檢查曾對真正忙碌的 Comfy server 拒絕排入 benchmark；GPU、T2I、I2I 均已實測。沒有完整重跑 LLM → 影片 → 發布。
- 後續補上真實 ComfyUI cache 契約回歸測試 `agentic/tests/integration/test_comfy_image_cache.py`：兩次存圖必須產出不同檔案且第二次重用上游節點快取。不使用 mocks；只驗 provider 快取生命週期，不冒充 GGUF 暖圖 GPU benchmark。
- 將相同回歸測試對隔離的舊 HEAD backend 執行，因第二次 cache 為空而失敗；目前修正版通過，連同既有 Comfy workflow tool scope 共 **9 passed，4.30 秒**。

checkpoint：`Krea2-Turbo-W4A4-noLowRank.safetensors`，8,057,520,008 bytes，
SHA-256：`f16df7a51f632d0743fb5402d47609ec0572a0fbcda9a4a2bf971f177bde101f`。
安裝於 `D:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models`。
文字 encoder／VAE 保留既有資產；沒有加套件或客製 node。

已有 production process 使用已匯入的舊 Python backend；快取程式修正從
後續新 process 生效。沒有為了 reload 而中斷等待審核的 run。

## 原 run 的目前狀態

截至 **2026-10-10 00:39 Asia/Taipei**，`81a8ec92f6ed` 六張已於 00:09:06
完成，00:09:24 已交付既有 Discord 人工審核。最新 lifecycle 仍在
`native-ref2va-reference-review`，尚未開始 WanGP 影片。此時等待的是
審核回應，並非 Krea 持續計算。沒有替使用者選圖、發訊息或重啟 run。

## 可重查的本機證據

資料位於 `output/benchmarks/krea_latency_20261009/`：

- `baseline_history.json`、`baseline_timings.json`：舊 provider histories／前五張時序。
- `cache_retained/benchmark.json`：保留快取的首張仍逾時；未測第二張。
- `krea_native_w4a4/benchmark.json`：同尺寸原生量化比較。
- `krea_cold_1024/benchmark.json`：清快取後較大尺寸比較。
- `production_six_candidates/acceptance.json`、`agentic_image_summary.json`：正式六張入口、每張 history、seed、PNG。
- `production_img2img/acceptance.json`：正式續圖入口與真實產物。
- `klein_fp8/benchmark.json`：快速但角色不合格的比較產物。

重測單一 API graph 可使用 `scripts/benchmark_comfy_images.py`。它只在
provider 閒置時排入自己的工作，儲存實際 PNG／history／diagnostic logs，
每張用 210 秒驗收；沒有 fake GPU 或自動更換模型。
