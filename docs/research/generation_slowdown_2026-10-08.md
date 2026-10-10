# Run 耗時診斷：2026-10-08

本次以既有 run、ComfyUI `/history`、`/queue`、`/system_stats`、ComfyUI 本機日誌與目前程式碼進行診斷。沒有啟動新生成、重啟 ComfyUI、調整生成設定或改動產品程式碼。

結論：耗時主要集中在 ComfyUI 取樣；六張依序生成的候選圖與較大的 H3 工作量放大總時間。另外已重現一個明確的 prompt 合約錯誤：LLM 回傳物件時，程式把物件轉成字串送進 H3，導致文字輸入膨脹與編碼時間增加。ComfyUI 的取樣速度異常波動已確認，但動態 VRAM、非同步 offload 或 Windows GPU 記憶體行為之間的具體責任，仍須控制變因實測。

## 最新 run 的實際耗時

Run `5fbf5fbe6369`，路線 `native_h3_l2va_story`，台北時間 20:06:15–22:21:39，總計 **135.4 分鐘**。

| 階段 | 耗時 | 證據 |
| --- | ---: | --- |
| 六張 ending keyframe 候選圖 | 61.9 分鐘 | `logs/runs/5fbf5fbe6369/lifecycle.log:17–18` |
| H3 L2VA 影片生成 | 62.9 分鐘 | 同檔案 `26–27` |
| 兩次審核節點合計 | 9.7 分鐘 | 同檔案 `19–24`、`40–44`；包含傳輸等節點開銷 |
| 其餘流程 | 約 0.9 分鐘 | 包括提示詞、QA、預覽與收尾 |

兩個生成節點約占總時間 **92%**。影片生成成功，run 最後因人工拒絕成品而結束；不能把這次 run 的 `failed` 狀態解讀成生成逾時。

實際 H3 輸入是 `512×640`、`length=362`、`24 fps`、`16 steps`，生成約 15 秒原始影片，之後套用 2 倍速。這與較早的 `640×360`、144 frames、20 steps 的短片工作量不同；不能直接用兩者總時間宣稱相同設定變慢幾倍。

## 確定問題一：LLM prompt 型別沒有落實驗證

Run `17782b80e099` 的 `llm/0006_segment_prompt.json`：

- schema 要求 `prompt`、`narration` 都是字串。
- 實際 `parsed_payload.prompt` 是含 `video_prompt`、`image_prompt_alternative` 的物件；`narration` 也是物件。
- 記錄卻標示 `status=success`。
- 用記錄中的 schema 驗證同一份 payload，得到 `prompt: expected string, actual dict`。

程式路徑：

1. `agentic/src/agentic/runtime/llm_engine.py:3013–3027` 只檢查 JSON 能否解析，沒有驗證解析後資料是否符合 schema。
2. 同檔案 `1778–1783` 使用 `str(payload.get("prompt"))`、`str(payload.get("narration"))`，讓錯誤物件變成 Python dict 的文字表示。
3. `agentic/src/agentic/skills/longvideo.py:418–421` 把 prepared prompt 追加到已組好的 H3 prompt。

結果是第 2 段送進 H3 的 prompt 達 **17,037 字元、3,948 個文字 token**，其中 **11,354 字元** 是巢狀物件的文字表示。這是確定的程式／資料合約問題。

ComfyUI 本機日誌 `D:/ComfyUI_windows_portable/ComfyUI/user/comfyui.log:1701–1704` 顯示 text encoder 載入完成後，直到 VAE 開始前經過約 **15.4 分鐘**。第 2 段最後在 **60 分鐘**客戶端上限處被取消，見 run 的 `lifecycle.log:40–43`。

文字 token 數以目前 ComfyUI 使用的本機 `qwen25_tokenizer` 與 `Qwen2Tokenizer` 計算，只計文字部分，不包含影像 token。H3 tokenizer 不會把全文自動裁成短提示詞，見 `comfy/text_encoders/minimax.py:141–186` 與 `qwen3vl.py:147–151`。

## 確定問題二：送入生成模型的共用指導文字很長

`agentic/src/agentic/video_directing.py:4–39` 的共用動作／鏡頭指導，在最新 prompt 中占 **3,522 字元、698 個文字 token**。最新 H3 完整 prompt 為 **10,076 字元、2,027 個文字 token**；共用指導占約 35% 的文字 token。

這些是如何寫場景的通用指導，與本次具體場景、人物動作、镜頭與聲音描述混在生成模型輸入中。文字編碼確實處理了這些 token；它們對總時間的獨立影響尚未以同設定 A/B 量測，不能宣稱刪短後會快特定倍數。

## 確定問題三：ComfyUI 取樣速度波動，六張候選圖放大等待

最新六張 Krea2 候選圖使用相同 prompt、模型、`512×640`、8 steps；工作流只有 seed 依序改變。ComfyUI execution 時間依序為：

| 候選圖 | 時間 |
| --- | ---: |
| 1 | 20.22 分鐘 |
| 2 | 2.83 分鐘 |
| 3 | 14.85 分鐘 |
| 4 | 2.72 分鐘 |
| 5 | 16.76 分鐘 |
| 6 | 4.05 分鐘 |

合計約 61.43 分鐘，加上節點間開銷後為 61.9 分鐘。這不是單純候選數增加：相同設定的單張推論本身也忽快忽慢。日誌 `comfyui.log:1859–1992` 顯示差異主要發生在 KSampler 取樣；模型載入多為數秒，VAE 解碼也很短。8 步取樣在快案例約 87–159 秒，在慢案例約 856–1,171 秒。

最新 H3 的日誌 `comfyui.log:2012–2017` 顯示：

```text
loaded partially; 0.00 MB usable, 0.00 MB loaded,
11144.41 MB offloaded, 542.06 MB buffer reserved
16/16 [52:39<00:00, 197.45s/it]
```

這表示 H3 進入大量權重 offload 的執行狀態；不表示 H3 完全改用 CPU 計算。52 分鐘以上花在 16 步取樣內。對照同一個 ComfyUI process 的 10/7 快案例，取樣為 20 步約 5:47，當時約 3.9 GB 權重常駐 GPU。不同影格數、尺寸、文字及影像條件都有影響，兩者只用於定位瓶頸，不能當成嚴格的速度回歸測試。

目前 ComfyUI 是以一般 `run_nvidia_gpu.bat` 啟動，沒有套用專案的 `scripts/run_comfyui_h3_lowvram.ps1` 保守設定。啟動日誌確認：`NORMAL_VRAM`、dynamic VRAM、pinned memory 與兩條 async offload streams 都啟用。

專案保守啟動設定為 `--disable-pinned-memory --disable-dynamic-vram --disable-async-offload --lowvram`，但本次没有重啟或比較。**目前證據不足以斷言其中某個旗標就是慢速原因。**

此外，9/26 的舊日誌已經是相同 ComfyUI 0.30.0、PyTorch 2.11.0+cu130 與動態模式；9 月底已存在慢案例。因此沒有證據把這次問題归因於近期 ComfyUI 升版。

## 修正與驗證順序

1. 在 LLM 回應進入下游前落實 schema／欄位型別驗證，拒絕物件型 prompt。保留完整創作資料於記錄，生成輸入只接受場景描述字串。
2. 將通用寫作指導留在 LLM 編寫階段，整理送進 H3 的場景、鏡頭與聲音描述；消除兩個來源重複追加的內容。不得直接截斷而丟失故事資訊。
3. 用完全相同的模型、prompt、seed、尺寸、影格數與步數，對照目前 ComfyUI 記憶體模式及專案保守模式。分開記錄文字編碼、取樣、VAE、GPU 專用／共用記憶體，才能確認非同步 offload 或動態 VRAM 的影響。
4. 速度穩定後再決定候選數與路線工作量；減少候選數會縮短總時間，但不會修復單張取樣的速度波動。

本次檢查時 ComfyUI running／pending queue 都為空。使用的是既有真實生成證據與本機資料合約驗證，沒有新的 GPU 生成驗證，也沒有可宣稱的修復後加速結果。

## 調整 1：H3 只接收場景描述

已完成 LLM 回傳與 H3 提交邊界的修正：

- `segment_prompt`、`goal_brief`、`compose_prompt` 的 `prompt` 必須是非空字串；分段的 `narration` 也必須是字串。巢狀物件不再透過 `str()` 接受，而是進入原有的 LLM 修正流程；修正仍失敗就停止準備，沒有模板代替品送往 H3。
- H3 工具入口再次驗證 prompt，拒絕物件、陣列、空白、序列化的物件、程式碼區塊及已知的解釋／通用寫作指導區塊。各種 H3 模式與分段配方都受此邊界保護。
- 分段渲染直接使用已驗證的 LLM 描述，不再重新建一份 H3 模板後追加 LLM 結果。通用動作與鏡頭指導留在 LLM 的輸入，供它寫成具體動作，取消 producer 和工具端的重複追加。
- Native H3 只組合角色外觀、場景、美術語言、實際分鏡與音效；不再追加新聞來源、角色對照、編劇卡、故事規劃摘要或額外的創作指令。完整資料仍保留在原本的故事與 run 記錄中。Ref2VA 也不再額外拼接參考素材摘要／保留分析。

離線重播真實紀錄：

| 紀錄 | 修正後的結果 |
| --- | --- |
| `17782b80e099/llm/0006_segment_prompt.json` 的巢狀 prompt | 立即拒絕，無法提交 H3 |
| `5fbf5fbe6369` 的 Native H3 故事 | 原 prompt 10,076 字元；以保留的故事重新格式化後為 3,190 字元，三段實際分鏡仍保留，沒有通用指導或編輯來源區塊 |

回歸測試 `agentic/tests/test_h3_prompt_delivery.py` 的 31 項情境通過，包含 LLM 修正／修正失敗、分段各種配方、所有 Native H3 模式，以及 provider 提交前的拒絕行為。相關範圍共 192 項通過，排除下面那項既有文案斷言。

完整整合檢查：`PYTHONPATH=agentic/src python -m pytest agentic/tests scripts/tests -q --disable-warnings --tb=short`，結果 **615 通過、2 跳過、1 失敗**。唯一失敗是 `test_shared_video_system_prompt_does_not_downgrade_news_grounding`：它要求 system prompt 含有原本的 `news-grounded` 等固定措辭，而工作區原有的 system prompt 已換成另一套文字。本次沒有修改那段 system prompt，這項文案斷言不代表 H3 的型別／提交回歸。

`require_h3_prompt` 新增的三個拒絕判斷均覆蓋接受／拒絕分支（100%）。整個 `minimax_prompting.py` 的分支覆蓋為 85.2%，尚低於 90% 診斷目標；未覆蓋的是既有角色參照的可選分支，以及目前 Native formatter 未使用的官方多鏡頭切換時間格式，沒有用純粹湊覆蓋率的測試補數字。語法檢查與 `git diff --check` 通過。

這次沒有呼叫真實 LLM、執行新的 ComfyUI／GPU 生成、重啟服務或發送通知。輸入精簡與提交合約已驗證；GPU 文字編碼時間、整體加速幅度及生成畫面仍待下一次自然 run 驗證。
