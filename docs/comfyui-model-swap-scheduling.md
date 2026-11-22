# ComfyUI 換模型排程：減少 unload / reload

日期：2026-11-21。狀態：研究與方案。提案 A（news_fusion 重排）曾實作並實測，沒有可量測的加速，已還原；程式碼沒有因此變動。

## 結論先講

1. **每個 job 內部大多已經是「同模型先做完再換」**。真正反覆來回換的只有兩處：
   - `news_fusion_story`：Krea → Klein → H3 → Klein → H3 → Klein → H3，3 個 beat 有 6 次家族切換，5 個 beat 有 10 次。依賴圖允許重排成固定 4 次，但實測（見「實測結果」）在這台機器上切換只花 0 到 6 秒，重排沒有可量測的加速，所以沒有採用。
   - `serial_video`：每集 Ref2VA（第 1 段）→ pi-H3（第 2..N 段），下一集又回 Ref2VA。每集至少 2 次切換，7 集的系列約 13 次。依賴鏈限制，**無法用重排解決**，只能靠 A/B 決定是否改配方（見下）。
2. **跨 job 批次化目前不值得做**。scheduler 是單執行緒、一次一個 job（`scheduler/scheduler.py:254-256`），目前環境檔設為每 1 小時一次（`SCHEDULER_INTERVAL_HOURS=1`，程式預設值是 4）。一個 job 在 H3 上就花數十分鐘，job 之間多一次切換相對很小。串行劇集還因為人工審核與 ledger 而不能並行（見提案 B）。手動連跑多個策略（例如 A/B 比較）時，把只用 Krea 的策略排在一起即可，不需要程式支援。
3. **ComfyUI 端的佇列排序幫不上忙**。runner 每次只送一個 `/prompt` 並等它完成（`runner.py:56-143`、`comfy_backend.py:101-115`），ComfyUI 的 `front` / `number` 優先級無事可排。排程必須在 planner 端做。
4. **不要在家族之間呼叫 `/free`**。Dynamic VRAM 本來就按「最近載入者優先」自行驅逐，`docs/krea2_comfyui_best_practice.md:51-65` 也已有同樣結論。
5. **還沒量到兩件事：切換的實際成本，以及「連續同家族 prompt 是否真的保持熱」**。現有 `comfy_h3_summary.json` 每份只有一個 prompt，且混有舊的 Q4 GGUF 16 步實驗，我用「前後 prompt 的時間間隔」粗估冷熱差異，結果被混雜因素污染，所以不引用任何秒數。整份方案的前提是「同家族連續送件會留在熱狀態」；若本機像 issue 14276 那樣每個 prompt 都重載，分組就省不到東西，改善要從環境下手。下面「先量再決定」一節說明怎麼量，第一個實驗同時回答這兩件事。

## 背景：「換模型很慢」在 Dynamic VRAM 下是什麼意思

ComfyUI 新版預設啟用 Dynamic VRAM（aimdo）。權重以檔案映射（mmap）方式存在作業系統快取，不再「VRAM 退回 RAM」；載入別的模型時，舊權重在 VRAM 不足時被較新載入的模型驅逐。作業系統隨時可以回收這些檔案快取，下次要用就從磁碟重讀。（[Comfy 官方部落格](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local)）

本機實測 H3 的系統 RAM 峰值約 69 GB（`docs/comfyui_h3.md:19-25`），所以 128 GB RAM 裡其他模型的檔案快取很容易被擠掉，切回去時就是整份從磁碟重讀。**切換成本主要是磁碟讀取量，不是 GPU 運算。**

已知的相關問題：[ComfyUI issue 14276](https://github.com/Comfy-Org/ComfyUI/issues/14276) 回報在 Dynamic VRAM + 模型檔是 symlink 時，切換 workflow 後每個 prompt 都重載模型（v0.23，仍為 open，維護者未能重現）。本專案 routing 文件提到 ComfyUI 0.41.0，版本已遠超該回報，是否仍受影響要在本機確認。

## 每個策略的家族順序與切換次數

「家族」＝同一組模型檔。H3 的 `t2va / i2va / fl2va / l2va` 四個 workflow 的 UNET、adapter、CLIP、VAE 檔名與載入參數完全相同（已逐一核對），所以 pi-H3 內部換 workflow 不算切換。`Ref2VA` 用不同的 UNET（GGUF）、不同的 32B 文字編碼器檔、再加一個 LoRA，與 pi-H3 是兩個家族。

「現況切換次數」只算一個 job 內相鄰 GPU 送件的家族變化，不含人工審核等待。

| 策略 | 現況家族順序 | 現況 | 最少可能 | 動作 |
|---|---|---|---|---|
| text2img | 單一 image workflow ×6 | 0 | 0 | 不動 |
| story_card / series_carousel / motion_comic | Krea only | 0 | 0 | 不動 |
| sticker_pack | 單一 image workflow | 0 | 0 | 不動 |
| native_h3_t2v_story | H3 only | 0 | 0 | 不動 |
| native_h3_ref2va（有 manifest） | Ref2VA only | 0 | 0 | 不動 |
| native_h3_story / native_h3_loop_story / text2image2native_h3_ref2va / text2image2video / game_sprite | Krea ×K → H3（→ NVIDIA VFX） | 1 | 1 | 不動：圖必須先審核才能餵影片 |
| native_h3_fl2va_story | Krea ×6 → Klein ×6 → H3 | 2 | 2 | 不動：結尾幀依賴審核後的開場幀 |
| native_h3_l2va_story | Krea ×1 → Klein ×6 → H3 | 2 | 2 | 不動 |
| **news_fusion_story** | Krea → Klein → H3 → Klein → H3 → Klein → H3 | **2N**（N 為 beat 數；3 個 beat＝6，5 個＝10） | **4** | **重排節點順序** |
| **serial_video** | 每集 Ref2VA → pi-H3 ×（N-1）；下一集再 Ref2VA | 每集 2 | 每集 2 | 配方 A/B（可選） |

另兩點屬於「單一 workflow 內部」：

- `nova-anime-xl.json` 的三個 SDXL checkpoint（novaAnimeXL ilV180、waiIllustrious v150、novaAnimeXL ilV140）**全部在輸出路徑上**（我從輸出節點反向追蹤確認過，不是死節點），也就是每張圖會依序載入三個 SDXL checkpoint；`z_image_plus_nova_model` 與 `nova_model_plus_z_image_anime` 則是 Z-Image 加 SDXL 兩個模型。text2img 每次出 6 張候選，若 ComfyUI 無法同時留住這些模型，每張都可能來回換，次數可能比 news_fusion 還多。這三個 workflow 合計佔 text2img 權重的 4/10（nova 2、兩個混合各 1；`routing.yaml:44-50`，權重總和 10）。**這是假設，需先量**（見「先量再決定」）。若證實，先評估 ComfyUI 本身的批次機制（空 latent 的 `batch_size` 一次出多張，重用現有機制，但 8 GB 可能 OOM），再考慮把圖拆成兩階段、6 張都先過第一階段再過第二階段。
- NVIDIA VFX 沒有 ComfyUI 權重檔，但與 H3 搶 VRAM 的情形沒有量過，不列入。

## 提案 A：news_fusion_story 重排節點（已實測，未採用）

以下是當時的設計，保留作為日後 RAM 吃緊、切換變貴時重做的依據。

### 為什麼安全

`planning/news_fusion.py:136-225` 的依賴是：

- `clip_i` 只依賴 `ready`（故事、素材檢查、全部已核准關鍵幀），**不依賴任何 bridge**。
- `tail_i` 依賴 `clip_i`；`bridge_frame_i`（Klein）依賴 `tail_i`；`bridge_in/out_i`（H3）依賴 `bridge_frame_i`。

執行順序來自 `ExecutionGraph.topological_order()`，是照 `plan.nodes` 清單順序走 DFS（`contracts.py:112-137`）。所以只要換清單順序，不用改 runner。

我逐一核對了各技能挑輸入的方式，執行順序不影響結果：

- `generate_keyframe`：用 `depends_on` 反向掃描，只看邊，不看執行順序（`skills/shared.py:76-95`）。
- `render_clip`：用 `anchor_nodes` 以 id 取，`concat` 用明確的 `video_node_ids`。
- `extract_last_frame`：只有單一依賴。
- `review.assets.select` 的自動挑選會受建立時間影響，但 news_fusion 的關鍵幀審核每次只有一張候選，重排不改變它。

### 新順序

```
Krea 關鍵幀 1 → 審核
Klein 關鍵幀 2..N（各自審核）
H3 i2va clip 1..N
ffmpeg 抽尾幀 1..N-1            （CPU）
Klein bridge_frame 1..N-1
H3 fl2va bridge_in/out 1..N-1
concat → QA → GIF
```

家族切換：Krea→Klein→H3→Klein→H3，**固定 4 次**，與 beat 數無關。

### 副作用要知道

- 全部 clip 先算完才會做 bridge。抽尾幀或 Klein 銜接幀若失敗，現在是在 N 個 clip 都算完之後才發現（以前在第 1 個 clip 之後就發現）；runner 遇到第一個失敗節點就停（`runner.py:132-143`），且只有 serial 模式會重用已完成節點，所以這種失敗會浪費 N 個 clip 的 GPU 時間。這是用「失敗時多浪費」換「正常時少切換」。緩解：Klein workflow 在同一個 job 裡已經為關鍵幀 2..N 成功跑過，素材或設定上的失敗會在 clip 開始前就出現；`image_to_image` 也有一次 OOM 重試；抽尾幀是 CPU 上的 ffmpeg。是否接受這個取捨由使用者決定。
- 影片最後的拼接順序由 `video_node_ids` 決定，不受影響；測試要直接驗證最終拼接順序與現在一致。

### 驗收測試（BDD，打真正的 planner 輸出；當時寫過，隨還原一起移除）

> **User** 要一支 5 個 beat 的新聞融合影片。**Given** 預設設定與全部關鍵幀已核准。**When** planner 產出執行順序。**Then** 相鄰 GPU 節點之間的模型家族切換次數 ≤ 4，且最終拼接順序與 beat 順序一致。

家族標籤要從 workflow JSON 的載入節點推導，不要用 `tool_name` 猜；測試不要寫死節點順序，只斷言切換次數與拼接順序這兩個對外行為。

## 提案 B：serial_video 要不要改配方（需 A/B，不是排程問題）

事實（皆已核對程式碼）：

- 第 1 段固定走 Ref2VA，第 2..N 段固定走 pi-H3 i2va（`long_video.py:78`；`workflow_config.yaml:118-122, 198-206`）。
- 第 k 集第 1 段需要第 k-1 集已組裝完成的結尾幀當 `continuation` 參考（`serial_video.py:1842-1884`、`1952-1957`），且每段都要人工驗收才能做下一段。所以**不能把多集的第 1 段集中先渲**。
- 每段驗收時 runner 被 Discord 審核阻塞，ComfyUI 閒置（`discord_review_view.py:817`）。這是閒置，不是換模型。

可能的槓桿，都需要先證明品質不掉：

1. 第 1 段改用 pi-H3 i2va，以上一集結尾幀當首幀，則整個系列只用一個 H3 家族，每集省 2 次切換。代價是失去 Ref2VA 的多參考身份綁定，**需要盲測**（做成單頁、勾選投票的 HTML，不要給零散的影片檔）。
2. 兩個 32B 文字編碼器是不同量化檔（Q4_K_M GGUF 與 NVFP4 AWQ）。若兩者可互換，CLIP 就能共用；程式碼目前刻意把 `model_profile` 限定在 Ref2VA（`comfy_h3.py:131-135`），不確定能否互換，要先驗證。

兩項都不是「重排」，而是產品決策，我不建議沒有 A/B 資料就動。

## 不採用的做法與原因

- **跨 job 批次**（先生完 N 個 job 的圖，再統一跑 H3）：scheduler 每 4 小時一個 job，加上人工審核必須在圖與影片之間進行，批次會把延遲拉長、狀態拆成可恢復的多階段，換到的只是每個 job 一次切換。等第 5 節的量測顯示一次切換真的很貴，再作為後續一層考慮。（每小時一個 job 時，job 邊界約一次切換，仍遠小於 job 本身的渲染時間。）
- **每個家族之間 `/free`**：與 Dynamic VRAM 的驅逐策略重複，且 repo 文件已反對。
- **用 ComfyUI 的 `front` / `number` 控制順序**：一次只有一個 prompt 在佇列，沒有可排的東西。
- **`--cache-none` + `--disable-smart-memory`**：社群回報互相矛盾（[討論 6659](https://github.com/Comfy-Org/ComfyUI/discussions/6659)、[issue 14481](https://github.com/Comfy-Org/ComfyUI/issues/14481)），而且這是要「更積極卸載」，與我們想「少卸載」相反。
- **移除 text2image2video 的 Tile Upscaler**：我原本以為它是白跑的，核對後發現角色設定走的路線已經跳過放大（`character_payload.py:639` 的 `skip_upscale_for_i2v`），只有直接呼叫 planner 才會建放大節點。所以預設路線沒有這個浪費，不用改。

## 先量再決定

（text2image2video 的上述結論再確認：`skip_upscale_for_i2v` 比對的是 routing 選出的策略本身 `routing["generation_type"]`，所以不論是 CLI 指定還是 scheduler 加權抽到，都會跳過放大。）

**環境（便宜，可能比重排效果更大）**

1. ComfyUI 與 comfy-aimdo 版本。
2. 模型資料夾是不是 symlink、是否在 NVMe：
   ```powershell
   Get-Item <ComfyUI 的 models 目錄>\* | Select-Object Name, LinkType, Target
   ```
   issue 14276 指出 symlink 會觸發每次重載，junction 不會。
3. `qwen_image_vae.safetensors` 與 `anima\qwen_image_vae.safetensors` 是否內容相同（相同內容放兩個路徑會被當成兩份載入）：
   ```powershell
   Get-FileHash <路徑1>, <路徑2>
   ```

**切換成本實測**：在 `comfy_backend.py:450-469` 已記錄每個 prompt 的 `node_execution_intervals` 與 `cached_node_ids`。下一次跑 news_fusion 時，比較「Klein 之後的第一個 H3 prompt」與「連續第二個 H3 prompt」的 `server_execution_seconds`，就能得到一次 Klein→H3 切換的真實成本；對 `z_image_plus_nova_model` 跑一次 6 張，比較第 1 張與第 2 張。這兩個數字決定提案 A 的收益與前述 hybrid workflow 拆階段值不值得做。同時看第二個 H3 prompt 是否明顯比第一個快、`cached_node_ids` 是否非空：若沒有，代表本機同家族連續送件也不熱，要先處理上面的環境項目，再談排程。

## 實測結果（2026-11-21，news_fusion_story，3 個 beat，固定 seed，不審核不發佈）

環境：RTX 4060 8GB、RAM 128 GB（實驗開始時約 50 GB 空閒）、ComfyUI 0.45.3、comfy-aimdo 0.6.0。耗時取自 ComfyUI `/history` 的每個 prompt 的 `execution_start` 到 `execution_success`。兩次是各跑一次，故事由 LLM 現寫，內容不同。

| | 舊順序 | 新順序 |
|---|---|---|
| prompt 數 | 12 | 12 |
| 家族切換 | 6 | 4 |
| ComfyUI 執行時間總和 | 1099 秒 | 1107 秒 |
| 整個 job 牆鐘時間 | 1191 秒 | 1188 秒 |

- **兩種順序的總時間沒有可分辨的差異**（差 3 秒，小於同家族 prompt 之間的自然波動，H3 約 109 到 132 秒）。
- 切換本身很便宜：切換後第一個 prompt 比暖機後慢 0 到 6 秒，唯一明顯的是整個 job 第一次載入 H3，慢約 20 秒（131 對 111 秒）。
- 時間主要花在 H3 取樣：7 個 H3 prompt 約佔總和的 73%，4 個 Klein 約 23%，Krea 約 4%。
- 切換後 Klein 的 `cached` 節點數為 0，代表前面家族換掉後，提示詞編碼等節點的快取也一併失效，但這部分成本同樣很小。

結論：**在這台機器、這個條件下，news_fusion 的重排沒有帶來可量測的加速**，切換不是瓶頸。這個重排正確、有測試，但新增了「接縫失敗要等 N 個 clip 跑完才發現」的代價（見提案 A 的副作用），收益不夠抵銷時應還原。

這個量測沒有涵蓋：RAM 被其他程式吃掉、作業系統快取被擠掉時的切換成本；`serial_video` 的 Ref2VA 與 pi-H3 互換（兩個不同的 32B 文字編碼器檔）；`nova-anime-xl` 一張圖載入三個 SDXL checkpoint 的情形。這三處的成本可能比上面大，需要各自量。

## 實作順序（照 AGENTS.md「分層成長」）

1. 量測（上面清單），不改程式。
2. 提案 A：已做完並還原（實測無加速）。
3. 視量測結果決定：hybrid image workflow 拆階段、serial_video 配方 A/B；兩者的切換成本都還沒量，先量再做。

## 資料來源

- [ComfyUI 官方部落格：Dynamic VRAM](https://blog.comfy.org/p/dynamic-vram-in-comfyui-saving-local)
- [ComfyUI issue 14276：Dynamic VRAM 與 symlink 造成每次 prompt 重載](https://github.com/Comfy-Org/ComfyUI/issues/14276)
- [ComfyUI 討論 6659：多模型循環時未使用模型堆積在 RAM](https://github.com/Comfy-Org/ComfyUI/discussions/6659)
- [ComfyUI issue 14481：`--disable-smart-memory` 在 GGUF workflow 下仍卸載](https://github.com/Comfy-Org/ComfyUI/issues/14481)
- [ComfyUI Cloud API：`/prompt` 的 `number` 與 `front`](https://docs.comfy.org/api-reference/cloud/workflow/submit-a-workflow-for-execution)（注意：Cloud 版忽略這兩欄，本機版才有效）
