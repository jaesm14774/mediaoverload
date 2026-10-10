**本地生成模型研究：文生圖、文生影、圖生影，2026-10-08**

研究窗口為 2026-07-08 至 2026-10-08。本報告同時納入窗口內的新發布／新版本，以及窗口內仍有明顯採用證據的較早模型。這兩種時間證據分開標示；GitHub 建庫、Hugging Face 建庫或 README 更新日期，都不能直接當成模型發布日期。

本次結論是：**近期影片模型優先比較 MiniMax H3 與 LTX-2.5；文生圖優先比較 Qwen-Image-2.1、Krea 2 Turbo、Z-Image-Turbo、FLUX.2 klein；插畫另看 Anima Turbo。** 要求 Apache／MIT 等寬鬆授權時，名單會改為 Z-Image、FLUX.2 klein 4B、Wan2.2，以及較早的 Qwen-Image-2512／Edit-2511。以下推薦是來源評估與部署判斷，本次沒有生成圖片或影片做盲測。

**如何判定「厲害、多人用、能本地跑」**

強度看獨立的人類偏好評測、模型能力與適合的題材；採用看近期 HF 下載、衍生模型、可下載工作流及維護中的推論工具；本地能力看已發布的權重、推論程式與具體硬體配置。廠商範例與自稱 SOTA 只證明廠商的展示／評測結果，不足以建立你的量化配置的畫質排名。

HF 顯示的下載量是近期一個月的 repository 指標，包含特定檔案的 GET／HEAD 請求；GGUF、多檔案與不同函式庫的計數方式不同。它不等於獨立使用者數，也不能跨倉庫相加推算總用戶。GitHub stars、HF likes 是累計興趣指標。本次沒有取得所有模型完整 90 天的下載曲線，因此不宣稱「近三個月使用人數排名」。[HF 計數方式](https://huggingface.co/docs/hub/models-download-stats)

**文生圖：五個主力選項**

| 模型 | 最值得測試的用途 | 近期性 | 授權與本地路徑 |
|---|---|---|---|
| Qwen-Image-2.1 | 生圖與編輯合一、人物／產品參考、原生透明背景 | 2026-09-20 發布 | 開放權重，Qwen Research；ComfyUI／Diffusers／SGLang 等 |
| Krea 2 Turbo／RAW | 美術方向、風格探索、分鏡首幀；RAW 用於 LoRA 訓練 | 2026-06-22／23 開放，窗口內仍有量化與工作流活動 | Krea Community；ComfyUI、官方推論程式 |
| Z-Image-Turbo／Base | 寫實、中文英文文字、少步快速出圖 | Turbo 2025-11，Base 2026-01；較早的活躍模型 | Apache 2.0；ComfyUI／Diffusers／GGUF |
| FLUX.2 klein 4B／9B | 快速迭代、生成與多參考編輯 | 2026-01 系列；較早的活躍模型 | 4B Apache 2.0；9B 非商業授權；ComfyUI／Diffusers |
| Anima Turbo v1.1 | 動漫、角色插畫、非寫實藝術 | 2026-08-24 UTC 上傳新權重，台灣時間 08-25 | 自訂模型授權；輸出可商用；ComfyUI 原生 |

Qwen-Image-2.1 的視覺生成部分為 7B，官方列出最多 10 張參考圖、局部編輯及 RGBA 生圖。它適合需要先生成、再修改、再去背的工作，而不是每一個步驟換一個模型。官方給出 CPU offload 範例，但「7B」沒有涵蓋所有文字編碼器與執行時記憶體。研究授權將免費使用限制在研究／評估，商業用途需要另取得授權。[官方模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1)、[發布紀錄](https://github.com/QwenLM/Qwen-Image-2.1)、[授權全文](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

Krea 2 Turbo 是 8 步蒸餾的文生圖模型；RAW 是未蒸餾版本。官方建議在 RAW 上訓練 LoRA，再用 Turbo 推論。它值得保留為美術與風格候選，且有本地生態；Krea 網站上的 Large／Medium／Turbo 產品和本地 RAW／Turbo checkpoint 必須精確對應，雲端展示不能直接代表本地 Q4 結果。授權允許年營收低於 100 萬美元的主體依條款商用，達到門檻需企業授權。[官方 repo](https://github.com/krea-ai/krea-2)、[Krea 授權](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE)

Z-Image-Turbo 的優勢是 6B、約 8 次模型評估，以及寫實與中英文字的能力。官方低延遲展示使用 H800；一般消費卡的標準配置說明是 16GB 顯存級別。8GB 應使用量化與 offload 配置，不能照搬「不到一秒」的宣傳數字。Base 適合更多控制、負面提示與微調，代價是更多步數。官方表中 Z-Image-Omni-Base／Edit 仍列為待發布，不能把預告算成可部署模型。[官方模型卡](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)、[模型列表](https://github.com/Tongyi-MAI/Z-Image)

FLUX.2 klein 支持文生圖及多參考編輯。4B 是部署與授權較容易的選項，9B 可作較大模型的比較候選；4B 的官方配置約需 13GB VRAM，8GB 需量化／offload。官方最低延遲是特定硬體與配置的結果。**4B 和 9B 的授權不同**，不能用 4B 的 Apache 標示概括全系列。[4B 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)、[9B 模型卡](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)、[官方系列說明](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

Anima 是 2B 插畫模型，Turbo 官方建議 CFG 1、8–12 步，是本機值得優先試的低成本候選。它不適合拿來當寫實通用模型。作者明確推薦從 Turbo 開始，也指出蒸餾會降低多樣性。模型本身與付費託管／嵌入產品受到商業限制，但模型卡明確允許生成圖片商用；這與 Qwen Research 的情況不同。[官方模型卡與作者推薦](https://huggingface.co/circlestone-labs/Anima)、[Turbo v1.1 權重提交](https://huggingface.co/circlestone-labs/Anima/commit/f973fc41ec7545364ac9776c2440285f43ff2a30)

**文生影與圖生影：優先比較的模型**

| 模型／版本 | 文生影 | 圖生影 | 同步聲音 | 推薦原因與主要限制 |
|---|---|---|---|---|
| MiniMax H3-Base FL2VA／Ref2VA | 是 | 是；首尾幀與參考模式 | 是 | 近期能力與生態強，重型模型；完整官方 2K 流程另有 API 階段 |
| H3 + LightX2V Turbo LoRA | 對應 T2VA 版本支持 | FL2VA／Ref2VA 對應版本支持 | 是 | 4／8 步候選，適合降低迭代成本；需匹配模式與 LoRA |
| LTX-2.5 distilled／dev | 是 | 是；關鍵幀控制 | 是 | 多鏡頭、可調整本地流程、細節渲染；22B 模型加專用編碼器仍重 |
| Wan2.2 TI2V-5B／A14B | 5B 與 T2V-A14B | 5B 與 I2V-A14B | 基本 T2V／I2V 不生成音訊 | Apache、成熟生態，適合商用與基準比較；不同任務要用正確權重 |
| HunyuanVideo-1.5 | 是 | 是 | 這些基礎 T2V／I2V 權重不含同步音訊生成 | 8.3B 的額外比較候選；官方流程需至少 14GB，低顯存工具可再降低 |

H3 在 2026-08-02 發布，其開放 H3-Base 有文字／首尾幀模式和多模態參考模式。官方系統包含 H3-Context-IR、H3-Base、H3-Regenerate-2K；官方完整 2K 復現示例將本地 Base 與另外兩個 API 階段結合。本地 Base 的 768p 示例可獨立運行。ComfyUI 支持剪枝與 INT8 ConvRot，並稱最小版本總記憶體需求由 123.6GB 降至 42.5GB，再依靠動態 offload。**42.5GB 是整體記憶體規模，不能讀成只要一張 42.5GB 卡，也不能讀成 8GB 就能快速生成。** [官方系統與部署](https://github.com/MiniMax-AI/MiniMax-H3)、[Comfy 優化說明](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)

H3 的近期採用有實際作品支持：Comfy 的 8–9 月同步聲音挑戰收到數百件作品，來自近 50 個國家；參加者可用本機或 Comfy Cloud，因此這是創作採用證據，不能當成全部都是本地用戶。[主辦方結果](https://blog.comfy.org/p/comfy-h3-sync-sound-challenge-the)

LightX2V 的 H3 Turbo 生態值得優先看，因為它提供 FL2VA／T2VA 與 Ref2VA 的不同 LoRA，而不只是泛稱「加速」。4 步、8 步、544p、768p 等版本的對應關係寫在開發者表格中；不能把文字模式的蒸餾 checkpoint 當成參考圖模式通用權重。Turbo 結果應與同一 Base 在你的題材上比較動作、身份與聲音。[開發者規格表與 ComfyUI 路徑](https://github.com/ModelTC/Minimax-H3-Turbo)、[權重](https://huggingface.co/lightx2v/Minimax-h3-Turbo)

LTX-2.5 在 8 月 11–12 日發布並有 ComfyUI 原生工作流，重點是原生多鏡頭、同步音訊、專用 Gemma 4 12B 編碼器、改善的蒸餾模型與 Diffusion Fidelity Rendering。distilled 適合先測；DFR 多了關鍵幀與細節處理，增加執行成本。官方 repo 說明 Comfy INT8 檔只能用在相應 Comfy 路徑，NVFP4 又有 Blackwell／kernel 條件。4060 不能套用 B200 的推論速度。[發布與 T2V／I2V 工作流](https://blog.comfy.org/p/ltx-25-day-0-support-in-comfyui)、[官方模型卡](https://huggingface.co/Lightricks/LTX-2.5)、[推論與 DFR](https://github.com/Lightricks/LTX-2)

Wan2.2 是 2025 年的較早系列，仍有成熟工作流和大量重包下載。TI2V-5B 用一個 checkpoint 支持文字或首幀；A14B 則有分開的 T2V 與 I2V 權重。官方 5B 的 720p 命令說明至少 24GB VRAM，A14B 示例需要更多。消費卡配置依靠量化、offload、少步 LoRA；LightX2V LoRA 是加速附件，不是另一個基礎影片模型。已有影片／動作驅動的 Wan Animate 屬於另一類任務，不能等同普通 I2V。[官方任務與硬體要求](https://github.com/Wan-Video/Wan2.2)、[Comfy 重包與工作流](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)

HunyuanVideo-1.5 於 2025-11 發布，提供 480p／720p 及 480p I2V step-distilled 等版本。官方說明至少 14GB、Linux；同一 repo 列出 Wan2GP 的約 6GB 社群路徑。這兩個數字描述不同推論流程。它適合做額外對照，但本次取得的原倉近期下載與更新證據弱於 H3／LTX，不把它列為近三個月最熱門首選。[官方部署及社群路徑](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)

**可核對的近期採用快照**

以下為 2026-10-08 讀取的 HF API 數字，已四捨五入。原始整數、權重 commit SHA、來源 URL 保留在研究附件。每一列是一個具名 repository；不合併不同格式、量化或重包。

| 具名 repository | 近期一個月下載指標 | 用來支持的結論 |
|---|---:|---|
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 359 萬 | 新系列有很高採用活動 |
| [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 2,329 萬 | Comfy 分拆檔與量化部署活動強；不能解讀為用戶數 |
| [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | 169 萬 | 新影片版本已形成使用生態 |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | 151 萬 | H3 少步部署的實際採用 |
| [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) | 651 萬 | Wan 較早，但本地生態仍活躍 |
| [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | 123 萬 | 插畫專用模型有明顯採用 |
| [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) | 750 萬 | Krea 本地部署生態活躍 |
| [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | 9.05 萬 | 官方原倉數字遠低於 Comfy 包；單看一個格式會失真 |
| [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 62.1 萬 | 小型少步模型持續被採用 |
| [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 40.6 萬 | 寬鬆授權的小模型有持續採用 |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 11.7 萬 | 九月新模型開始形成生態 |
| [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | 747 萬 | 原生 Comfy 支持與重包活動強 |
| [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 80.5 萬 | 量化版本被大量取得 |

**獨立畫質證據怎麼讀**

查詢時，Artificial Analysis 的 T2V／I2V 頁面把 H3 列為領先的開放權重系列。Qwen-Image-2.1 在 T2I 頁面為 1035±9；Z-Image Turbo 與 klein 9B 各 941±8，klein 4B 為 863±9。這支持「4B 優先解決速度與部署，不適合直接宣稱畫質第一」的判斷。[T2I 評測](https://artificialanalysis.ai/image/leaderboard/text-to-image)、[T2V 評測](https://artificialanalysis.ai/video/leaderboard/text-to-video)、[I2V 評測](https://artificialanalysis.ai/video/leaderboard/image-to-video)

評測頁上的 H3／LTX Fast／Pro 可能透過供應商服務執行；Krea 列名也包含雲端 Large／Medium。只有名稱一致不足以證明就是相同的開放 checkpoint、prompt enhancer、精度及採樣設定。因此不將這些 Elo 當成 RTX 4060 上 H3 Q4、LTX INT8 或 Krea Q4 的測試成績。不同模態／含聲音與無聲榜的 Elo 也不能橫向相減。

**授權會直接改變選擇**

| 選項 | 與這次選型有關的條件 |
|---|---|
| Z-Image、FLUX.2 klein 4B、Wan2.2、Qwen-Image-2512／Edit-2511 | 官方相應權重標示 Apache 2.0；別把此授權沿用到不同版本 |
| Krea 2 | 自訂授權；100 萬美元年營收門檻，以及再散布條件 |
| FLUX.2 klein 9B | 模型非商業授權；不要與 4B 混用授權結論 |
| Qwen-Image-2.1 | 研究／評估授權；商業使用另申請 |
| Anima | 模型託管與嵌入受限制，生成圖片可商用；另繼承 Cosmos 基底條件 |
| LTX-2.5 | LTX-2.x Community；年營收達 1,000 萬美元的商業主體需付費授權，微調轉移另有條件 |
| H3 與其衍生模型 | 自訂地區、用途與商業條件；存在官方授權與 Comfy 商用說明需要釐清的差異 |

H3 的模型授權明列排除美國、歐盟、英國、韓國，且限制文字也涉及輸出；台灣未列於該排除名單。授權 IV.1 列出商業產品／服務年收入超過 2,000 萬美元需事前書面許可。但 Comfy 的部署文件另寫本地生成的商業輸出需 MiniMax 商業授權。這兩份官方材料的表述範圍不同，本報告不把它們擅自合成「低於門檻可隨意全球商用」。LightX2V 頁面即使把 LoRA 標為 Apache，也不能抹除基礎 H3 的條件。[H3 授權全文](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)、[Comfy 商業說明](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

LTX 的當前 repo 區分新舊授權，2.5 對應 2026-08-11 的 LICENSE-2_x；不能用舊版的文件概括。[LTX-2.x 授權](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)。較早的 Qwen 權重仍可作授權較簡單的生圖／編輯對照。[2512](https://huggingface.co/Qwen/Qwen-Image-2512)、[Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

**前沿模型與觀察名單**

Ming-Image-0.1-Design 的九月發布包值得追蹤：6B、MIT、以 UI／海報／資訊圖表等文字設計為主，支持 RGBA。官方驗證配置是一張 80GiB CUDA GPU；這不是證明最小需求為 80GB，也不能證明 8GB 可直接跑。當前 API 下載欄位是 0、likes 399，權重實際存在；0 可能反映計數路徑，不足以說沒人用。因近期採用證據較少，先列觀察，而不是「非常多人推薦」主力。[官方模型卡](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)

MAGI-2 Preview 是 114B MoE 音畫模型，官方已發布 T2V／I2V 程式與權重，但目前配置要求 8 張 NVIDIA Hopper GPU，整體 checkpoint 約 307GB。每 token 只啟用 6B 不代表只要存 6B 權重。它可以自託管，但不適合這台家用卡。[官方要求](https://github.com/SandAI-org/MAGI-2-preview)

Cosmos3-Super-Image2Video 是 NVIDIA 的 64B I2V／物理世界模型，發布於五月，官方提供自託管方式與 OpenMDW1.1 授權。獨立評測中有競爭力，但其體量與機器人／物理模擬定位使它不是 4060 優先的創作工具。[官方模型卡](https://huggingface.co/nvidia/Cosmos3-Super-Image2Video)

FastH3 Preview v1 是值得研究的加速衍生模型，但此次查到的 VSA-DataFree checkpoint 只蒸餾 T2VA，沒有 FL2VA／Ref2VA；作者測試預設為 4 張 B200，還需要 VSA-H3 backend。因此不能直接當成「4 步、家用 8GB、T2V/I2V 都適用」的替代品。[開發者模型卡](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)

Krea Realtime 14B 與 Krea 2 是不同模型：前者為自回歸影片模型，官方建議 40GB 以上 VRAM 且採非商業授權。本次不把它列為你這台卡的主力。[官方 repo](https://github.com/krea-ai/realtime-video)

較早的 CogVideoX、Mochi、SDXL 及一些小眾新 checkpoint 沒有列入首選，理由是這次需要兼顧近期能力、採用與本地成本；這不表示它們不能使用。排行榜上的 Wan 3.0、Qwen-Image-3.0、Krea Large 等名字也不自動意味已提供對應本地權重。

**對這台 MediaOverload 電腦的具體建議**

本次實際讀取硬體為 RTX 4060，8,188 MiB 顯存，系統記憶體約 127.8 GiB。現有工作流使用 Krea 2 Turbo Q4_0、Qwen3VL-4B Q4_K_M 和 Qwen-Image VAE；這是配置檢查，本次沒有另跑生成。[現有 Krea 工作流](/C:/Users/jaesm14774/Desktop/self_project/mediaoverload/configs/workflow/krea2_turbo.json)、[本機部署文件](/C:/Users/jaesm14774/Desktop/self_project/mediaoverload/docs/krea2_comfyui_best_practice.md)

| 需求 | 我建議的優先測試順序 | 判斷依據 |
|---|---|---|
| 卡通／角色首幀 | Anima Turbo → 現有 Krea 2 Q4 | 2B 插畫模型成本較低；Krea 有現成部署 |
| 寫實與中英文字 | Z-Image-Turbo 量化 → Krea 2 Q4 | 活躍生態與題材相符；需逐一確認文字與身份 |
| 快速編輯與多參考 | FLUX.2 klein 4B 量化；研究用途再看 Qwen-Image-2.1 | klein 的 4B 與 Apache 條件；Qwen 的能力與授權需一起考量 |
| 動作與同步聲音 | H3 + 匹配的 LightX2V Turbo → LTX-2.5 distilled | 先比較 4／8 步和實際動作，不先追 2K／4K |
| 授權簡單的無聲影片 | Wan2.2 5B 量化 → A14B I2V 量化 | 5B 更適合先建成本基準；14B 額外測品質收益 |

大量 RAM 讓 offload 更可行，但不會消除 PCIe 搬運、attention 或 VAE 解碼的成本。我的部署建議是：8GB 先測一张首幀與一段 3–5 秒、約 480p 的影片；需要頻繁測試重型影片模型時，16GB、24GB、32GB 是逐步增加的餘裕，並非所有模型的官方最低需求或速度保證。不要直接從整段 30 秒、2K 多鏡頭批次開始選模型。

最值得額外比較的推論工具是 Wan2GP。2026-10-07 的 v17.17 說明宣稱 H3 15 秒 480p 可用 5–6GB VRAM，1080p 約 11GB，並新增顯存／RAM 管理。這是開發者在其流程的測試與宣稱，**尚未在本機驗證，也不代表 ComfyUI 的同名配置會有相同成本**。它證明模型選型要同時選推論引擎，不能只比較模型名稱。[開發者更新與低顯存配置](https://github.com/deepbeepmeep/Wan2GP)

之後的比較應固定同一首幀、分辨率、秒數及題材，每個配置產生多個 seed：觀察「角色確實完成動作」「接觸與因果合理」「角色身份保持」「畫面穩定」「聲音與動作同步」，同時記錄端到端時間、峰值 VRAM／RAM、失敗率與人工挑選成本。技術上生成成功與人看起來好看是兩個不同的驗證結果。

**研究附件與驗證限制**

來源原文、失敗／gated 抓取紀錄、HF／GitHub API 快照以及版本時間證據保存在 `C:\Users\jaesm14774\.codex\visualizations\2026\10\08\01a11bfa-eaee-7eb2-be79-ea4b527cc9fc\model-research\`。主清單與額外採用快照分別為 `adoption_snapshot.json`、`adoption_snapshot_extra.json`，廣泛檢索快照為 `landscape_snapshot.json`；來源 manifest 記錄成功與 401／404，失敗內容沒有被當成已讀證據。gated 權重的公開模型卡可讀，不等於本次已取得下載權限。

本次完成來源與硬體核對，沒有安裝新模型、更新推論環境、修改生成路徑或提交程式碼，也沒有本機橫向畫質／速度測試。對你的最終生產品質，仍應以上述相同條件的生成與人工檢視決定。
