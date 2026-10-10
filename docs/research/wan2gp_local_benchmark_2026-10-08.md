# Wan2GP 本機研究與同條件速度實測

研究／測試起始日：2026-10-08。這份報告對應真實 RTX 4060 執行，不把 upstream 宣傳數字當成本機結果。

## 測試目的與範圍

比較 MediaOverload 現有 ComfyUI H3 工作流與 Wan2GP H3 工作流。在相同 prompt、diffusion 與 text encoder 檔案、seed、width、height、length、fps、steps 下生成真正影片，再驗證輸出規格與耗時。使用一個約五秒的 T2V 片段，涵蓋完整文字編碼、取樣、影音解碼與輸出。

這是兩個完整工作流的比較。ComfyUI 的 Spectrum 和 Wan2GP 的 Spectrum 實作不同，attention、記憶體排程及 PyTorch 版本也不同；因此結果不能歸因於單一優化，也不能宣稱生成品質或像素完全等價。

## Wan2GP 適合這個專案的原因

Wan2GP 是涵蓋多種模型的生成執行環境，包含 H3、Wan、LTX、Hunyuan，以及 Krea2／Qwen Image 等圖像模型。它的優勢集中在小 VRAM 的模型排程、量化執行和批次／API 入口。MediaOverload 可以繼續負責故事、提示詞、候選選擇與影音收尾，把 Wan2GP 視為可以比較的生成 backend。[README](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/README.md)

Profile 4 逐塊送模型到 GPU，主要模型保留於 reserved RAM。MMGP v4 提供自訂 VRAM allocator、RAM spilling、記憶體 pinning 及 preload 選項；這些對「模型大於 8 GB VRAM」的 H3 路徑尤其相關。README 的速度與 VRAM 改善是作者的測試條件，不能直接推算 RTX 4060 的加速倍數。[CLI 與 memory profiles](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/docs/CLI.md)

H3 handler 支援 FL2VA Pruned、GGUF text encoder、RES Multistep、Spectrum，並且可以用 finetune definition 指向本機模型，這讓比較可以沿用現有 diffusion 與 encoder。[H3 handler](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/models/minimax_h3/minimax_h3_handler.py)、[checkpoint loader](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/models/minimax_h3/minimax_h3_main.py)

H3 的有效影格數使用 `17n+5`。124 frames 恰好符合這個規則，24 fps 的實際長度是 5.1667 秒，而不是整數五秒；比較前必須用 ffprobe 查實際 frames 與 duration。[H3 pipeline](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/models/minimax_h3/pipeline.py)

Python API 會保留 session 和已載入模型，可提交單一 task 並讀取 progress／完成／error。相較每張影片重啟 CLI，這更接近 MediaOverload 的持續 backend 使用方式。本次比較將 Python API 初始化時間獨立記錄，生成時間從 task 提交到影片完成。[Python API](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/docs/API.md)

目前版本採用 WanGP Community License 2.0，允許免費個人、研究、評估與公司內部使用，但對付費軟體、付費 API／SaaS 等整合設有另外的授權條件。不要把「原始碼可讀、本機免費使用」等同於 Apache／MIT 的商用自由；這次是私人本機測試。[LICENSE.txt](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/LICENSE.txt)

## 隔離環境與固定條件

- Wan2GP checkout：`D:\Wan2GP_benchmark`，commit `6479db36bdc2619a904a852bba9c2d78e1a83f82`。
- 獨立 Python 3.11.13 venv；Torch 2.10.0 + CUDA 13.0，Triton Windows 3.6、SageAttention 2.2 與 GGUF CUDA kernels 1.0.25。完整版本保存在 `wan_packages.txt`。
- 顯卡：RTX 4060 8 GB；CPU：Intel i5-14500；RAM 約 128 GiB。
- 現有 ComfyUI：0.30.0、Python 3.13.12、Torch 2.11.0 + CUDA 13.0。
- 共用 diffusion：`minimax_h3_fl2va_pruned_fp8_Q4_0.gguf`。
- 共用 text encoder：`qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf`。
- Video VAE：共用現有 FP16 video VAE，Wan2GP 的不同檔名以同磁碟 hard link 對應同一檔案。
- Audio VAE：ComfyUI 使用已合併 weight normalization 的 FP32 格式；Wan2GP 使用原版 H3 FP32 `weight_g/weight_v` 格式。逐一驗證 915 個 learned tensors、172 組 weight normalization，合併後最大差值為 `2.38e-7`，全部通過 `atol=1e-6, rtol=1e-5`；ComfyUI 額外的 normalization buffers 也與 Wan2GP constants 一致。證據：`audio_vae_equivalence.json`。兩邊使用同一組學習權重，但檔案表示方式不同。
- 規格：608×352、124 frames、24 fps、16 steps、seed 20261008；T2V + native stereo audio，RES Multistep，video shift 12、audio shift 3。
- prompt：見 `manifest.json`。兩邊使用同一字串，停用 prompt enhancer、LoRA、第二階段、超解析、插幀與倍速後製。
- 佇列：提交前確認空佇列，兩個 GPU 生成工作依序執行。

安裝堆疊依 upstream RTX 30XX–50XX 指南；使用已存在的 Python 3.11.13，而不是修改 portable ComfyUI 的 Python。[安裝與 acceleration kernels](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/docs/INSTALLATION.md)

## 已觀察到的失敗

首先直接用現有 8188 服務跑相同測試，文字編碼在 43.25 秒時 OOM，沒有生成影片，也沒有進入 denoising。錯誤發生在 GGUF text encoder 的 Q4_K dequantization；要求再配置 1.45 GiB 時沒有剩餘 CUDA memory。記錄位於 `failed_dynamic/`。

接著使用另一個 port 8189、同一套 ComfyUI 與模型，套用專案既有保守啟動設定：`--reserve-vram 1.0 --disable-pinned-memory --disable-dynamic-vram --disable-async-offload --lowvram`。此設定使 text encoder 使用 CPU，這個差異會計入工作提交到成品完成的時間。原 8188 服務與 production route 保留。

Wan2GP 的前置設定與載入也有失敗紀錄：`force_fps` 需要字串 `"24"`；自訂 model definition 需要 `description`；ComfyUI 的 fused audio VAE 缺少 Wan2GP 所需的 weight-normalization parameter keys。已改用正確設定與原版 VAE，未修改 upstream 載入器或原 ComfyUI 模型。這些前置失敗不計入成功生成的單次速度。

## 實測結果

**這組約 5 秒 H3 T2V 測試，Wan2GP 的 task-to-file 時間約為 ComfyUI 保守模式的 1/4.53，減少 77.9% 等待。**

| 階段 | ComfyUI 保守模式 | Wan2GP Profile 4 + Sage2 |
| --- | ---: | ---: |
| 完整生成：提交到影音成品 | **902.98 秒（15:03）** | **199.48 秒（3:19）** |
| 文字編碼／conditioning | 305.44 秒，CPU | 約 4.39 秒，GPU |
| 取樣階段 | 563.06 秒，含取樣節點內的模型載入／切換 | 約 158.10 秒，含 Spectrum smoothing replay |
| Video + audio decode | 25.02 秒 | 約 26.37 秒 |
| 整張卡 VRAM 最高取樣值 | 7,843 MiB（7.66 GiB） | 5,931 MiB（5.79 GiB） |
| 影片實際 width × height | 608×352 | 608×352 |
| 影片實際 frames／fps／duration | 124／24／5.166667 秒 | 124／24／5.166667 秒 |
| 音軌 | 32 kHz、雙聲道 AAC | 32 kHz、雙聲道 AAC |

Wan2GP API 初始化另用了 **22.21 秒**，未計入 task-to-file；從新 Python process 啟動到退出的總時間為 **228.02 秒**。ComfyUI 也使用預先啟動的 server，因此兩邊主表皆不計 server／API 的啟動時間。安裝、下載、首輪 CPU 套件編譯及失敗重試更不屬於日常單片時間。

VRAM 數值是每兩秒 `nvidia-smi` 取樣的整張卡使用量，包含桌面／其他 GPU 程式，並非各 process 的精確峰值。兩個生成工作串行執行；Wan2GP 生成時兩個 ComfyUI queue 均為空。

完整時間的改善有很大部分來自文字編碼器由 CPU 執行改為 GPU 量化執行；不能把 4.53 倍稱為單純 GPU 核心或單一 kernel 的速度改善。取樣階段亦有明顯縮短，但 attention、offload、Spectrum 和 PyTorch 都不同，尚未隔離各自貢獻。兩邊 VAE 解碼時間接近。

Wan2GP log 確認 GGUF CUDA embedding／linear fast paths 使用於 Q4_K、Q6_K、Q4_0；其中有 Q4_0 的 non-contiguous input 觸發 upstream 自帶的 PyTorch dense materialization 路徑。這次結果包含那段實際執行行為，不能宣稱所有 layers 都使用同一 CUDA fast path。

抽取五個時點的畫面，兩邊都出現 Kirby 接近種子、彎身拾起、舉到面前的動作與相近構圖。這只是有限的畫面檢查，沒有完成逐幀盲評、音質／同步評估或跨 seed 的品質比較。Contact sheet 的第六格是排版留白，不是輸出黑幀。

**結果適用範圍：每個 backend 一次成功的冷模型 T2V 生成。** OS file cache 沒有刻意清空。ComfyUI 後段實際步的耗時波動很大，因此不能直接推算 15 秒、512×640、I2V／FL2V、六張候選圖或完整長片 route 也固定快 4.53 倍。原 8188 dynamic mode 本次 OOM，沒有可比較的成功完成時間。

## 對目前專案的判斷

Wan2GP 值得作為 H3 的新 backend 候選：相同主要權重、相同生成規格，本機完成時間與 VRAM 取樣值均有改善。這次已完成真正的兩邊生成，尚未把 production route 改為 Wan2GP。

接入 production 前，需要把目前使用的 image conditioning 與長度納入驗收：固定同一首／尾幀，使用 512×640、362 frames、24 fps、16 steps 的 L2VA／FL2VA；至少三組配對 seed，分開報 cold／warm，並由使用者判斷角色、動作與音畫品質。最後才量測完整 Krea2 候選圖 → H3 → 收尾 route，因為這次單段 T2V 不涵蓋那些等待。

## 重現與證據

腳本：`scripts/benchmark_wan2gp.py`。測試前的 Given／When／Then 合約在 `output/benchmarks/wan2gp_20261008/acceptance.md`。

```powershell
python scripts/benchmark_wan2gp.py prepare
python scripts/benchmark_wan2gp.py comfy --comfy-url http://127.0.0.1:8189
python scripts/benchmark_wan2gp.py wan
python scripts/benchmark_wan2gp.py summarize
```

`prepare` 產生固定 manifest、ComfyUI API graph 與 Wan2GP settings；`comfy` 保存 prompt receipt、history、逐節點事件與 GPU telemetry；`wan` 保存 API 事件、完整 process log、初始化／生成耗時與 output probe。`summarize` 只有在兩邊生成成功，且實際 width、height、frame count、fps、duration 符合固定條件時才計算速度比。

完整 evidence bundle：`output/benchmarks/wan2gp_20261008/`。重現前先啟動隔離的保守 ComfyUI；啟動 arguments 與 log 在 `comfy_server.json`。測試完成後已停止該隔離 server，原 8188 服務保留。Wan2GP checkout、venv、模型定義、下載的原版 audio VAE 與 tokenizer 留在 `D:\Wan2GP_benchmark`。

主要成品：`comfy_matched.mp4`、`wan_matched.mp4`；機器可讀結果：`comparison.json`、`verification.json`、`input_contract.json`；原始輸入與權重檔案身分：`manifest.json`、`comfy_graph.json`、`wan_settings.json`、`asset_identity.json`、`audio_vae_equivalence.json`。

目前腳本測試的是生成 backend，不包含新聞／LLM 寫稿、六张候選圖、人工審核、Discord 或發布；也不會把技術規格通過當作角色、動作、畫質或故事品質通過。
