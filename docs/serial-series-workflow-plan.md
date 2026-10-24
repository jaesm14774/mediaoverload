# Serial series 故事先行流程計畫

## 驗收情境

### User Given 新系列沒有指定角色且新聞靈感模式啟用 When 建立故事草稿 Then 系統從現有新聞庫隨機挑一則可追來源的新聞作靈感，產生完整且有結尾的系列故事稿，保存來源與草稿 SHA，並停在人工審核前

### User Given 新系列使用新聞靈感但新聞庫沒有可用且可追溯的素材 When 建立故事草稿 Then 系統說明無素材並停止，不把它靜默改成純即興故事

### User Given 創作者提供自訂故事想法並明確停用新聞靈感 When 建立故事草稿 Then 系統只用創作者想法寫故事，並把靈感模式記為 none

### User Given 創作者編輯了待審故事檔 When 以該檔案核准故事 Then MySQL 保存核准當下的完整文字、審核人與 SHA，之後的角色及分集規劃都以這份核准版本為準

### User Given 故事尚未核准或被拒絕 When 角色卡、分集或影片生成被要求啟動 Then 系統不會把該故事當成核准來源或開始 H3 render

### User Given 故事已核准而創作者提供角色偏好 When 產生角色身分卡 Then 角色外觀與行為錨點會遵守偏好並保持在故事世界規則內

### User Given 故事已核准但創作者沒有角色偏好 When 產生角色身分卡 Then LLM 依故事創作主角，交由創作者審核外觀文字與候選參考圖後才存成固定角色版本

### User Given 角色卡或參考圖已核准 When 後續集數進入規劃 Then 每集都引用同一個角色版本與已核准圖片 SHA，角色或圖片版本改變時下游計畫必須重審

### User Given 完整故事與角色版本都已核准 When 規劃某一集 Then 系統先產生該集的因果劇情節點，再拆成場景與鏡頭；每個鏡頭清楚寫明目的、開場狀態、動作節拍、道具變化與結尾狀態

### User Given 故事或鏡頭草稿需要人工修改 When 創作者保存並核准新版本 Then 下游版本綁定新的 SHA，已不再符合上游版本的舊計畫與影片不會被接續使用

### User Given 分鏡獲核准且每段影片通過技術 QA 與人工接受 When 下一段開始生成 Then 它只使用該段受審影片的實際核准尾幀；拒絕、逾時、失敗或素材 SHA 不符時停止接續與組裝

### User Given 最終影片通過 QA 且創作者核准發布內容 When 平台發布完成 Then 每個平台各自留下影片 SHA、結果與 provider receipt，已回報成功的平台不會重送

### User Given 日常一般生成沒有 serial series 選項 When 建立與執行生成計畫 Then 原有非連載路徑不讀寫連載狀態

## 已有能力

目前 serial video 已使用專用 MySQL `mediaoverload_serial`。`SerialVideoStore` 建立並維護 `series`、`episodes`、`episode_plans`、`segments`、`render_attempts`、`reviews`、`reference_assets`、`episode_publications` 八張表。集數版本、段落、render 嘗試、人工與技術審核、尾幀與參考圖、最終影片和逐平台發布結果都可透過 ID 關聯；重要素材有 SHA-256。

每段影片會先做技術 QA，再等 Discord 人工接受；只有受審且獲接受的 MP4 才能抽尾幀、接下一段或進入組裝。最終影片另有發布審核與逐平台 receipt。`long_video` planner 已經把故事拆成 H3 大小的連續段落，並保留前段實際狀態與已接受片段的接續限制。

已有單集自動點子產生器 `generate_serial_episode_idea`，輸出一個主題與一段短故事前提。`NewsContextService` 已連接本機設定的 `news_ch.news`，並能挑選可追來源新聞；其他 Native H3 故事流程已使用新聞內容。一般生成的策略分流也已能啟動 serial runner。

## 盤點時發現的缺口

- 新系列現在先要求固定角色與參考圖，之後才開始第一集；缺少「完整故事先核准，再從故事創作角色」的建立順序。
- 自動點子只產生一集簡介，沒有完整的系列故事稿、已知結局、世界規則或版本審核記錄。
- serial 自動點子沒有使用新聞靈感，也沒有把新聞標題、網址、來源、刊登時間與挑選時間綁進故事版本。
- 角色只以 `character_name` 和 `reference_assets` 保存；沒有可審核、可版本化的身分卡文字與審核紀錄。
- 每集分段已存在，但目前缺少以核准系列故事為唯一上游、可先審核「劇情節點 → 場景 → 鏡頭」的前置規劃階段。
- 現有 `episode_plans` 儲存執行計畫與 SHA；還沒有把每集內容明確綁定到核准故事版本及角色卡版本。

## 目前實作狀態（2026-10-24）

故事、角色卡、參考圖選擇、分集節點／場景／鏡頭審核與新系列影片 gate 已接進 CLI 和 MySQL。每集開始時才建立 production series，並留下故事、角色、分鏡與圖片 SHA 的 lineage。既有 accepted-tail 與發布帳本沒有改掉。

實際新聞／LLM／圖片／H3／Discord／平台試作尚未執行。本次測試只證明程式提示、狀態、SHA、MySQL 和 serial integration contract；不代表外部模型或服務產物品質已驗收。

## 建議流程

```text
使用者想法 ───────────────────────┐
                                  ├─> 完整系列故事稿 -> 人工核准 -> 角色身分卡與參考圖 -> 人工核准
現有新聞庫隨機抽一則（預設） ─────┘                                      |
                                                                        v
                  核准故事 -> 每集劇情節點 -> 場景與鏡頭清單 -> 人工核准
                                                                        |
                                                                        v
  已核准角色參考 + 前一段實際核准尾幀 -> H3 -> 技術 QA -> 人工逐段接受
                                                                        |
                                                                        v
                                  組裝 -> 最終 QA -> 發布審核 -> 各平台 receipt
```

新聞只作為故事靈感，不作為新聞重現、事實查核或新聞報導。提示詞會要求改寫人物、地點與事件細節，避免把真實人物當成角色或把新聞原文搬進劇情。新聞來源資訊保留在 MySQL，讓創作者看得到這次靈感從哪裡來。

新聞來源預設使用 `NewsContextService` 已連接的 `news_ch.news`。serial 專用的選取會在現有可用性、安全類別、時間範圍與來源網址檢查之後，從符合條件的文章均勻隨機抽一則；一般新聞流程維持原選取方式。新聞資料庫無法查詢或找不到合格素材時，預設新聞模式明確停止。創作者可用 CLI 明確關閉新聞靈感，但必須另外提供故事想法。第一階段不新增 GDELT／NewsAPI 服務或金鑰。

完整故事稿先用可閱讀的繁體中文文字呈現，包含故事前提、基調、世界規則、主角要改變的事、起因與升級、高潮、結局，以及依故事自然篇幅提出的集數走向。它是供人確認方向的故事處理稿，不是分鏡。角色卡在故事核准後生成。分集節點與鏡頭清單再以核准故事和角色版本為輸入，之後才進入 H3。

使用者已明確要求不做 schema 驗證。新故事、角色卡、分集節點與分鏡都由模型產生可讀文字；系統只檢查審核狀態、核准檔案、圖片路徑／SHA 與上游版本，不加入 JSON Schema、Pydantic、Guidance 或 Instructor 驗證層。既有 `long_video` planner 和產片流程維持原有契約。

## 分階段實作

### 第 1 階段：新聞靈感與完整系列故事審核（已實作）

- Serial 用既有 `news_ch.news`，先套用現有安全、可用和來源網址條件，再均勻隨機抽一篇；新聞不可用時停止，創作者可明確選擇 `none` 並提供自己的故事方向。
- 純文字 LLM 提示產生有因果、世界規則、角色變化和完整結局的系列故事稿；不新增 JSON Schema、Guidance 或 Instructor 驗證。
- 草稿、來源快照、prompt SHA、模型 ID 和人工編修後的核准 SHA 留在 `series_story_plans`。故事核准之前不建立 production series。

### 第 2 階段：故事導出的角色卡與參考圖核准（已實作）

- 只有最新核准故事能建立角色卡；偏好可由創作者指定，空白時由 LLM 提出原創主角。
- 角色卡提供身份、行為與固定外觀錨點，也提供可複製給既有圖片生成 CLI 的角色參考圖提示。創作者檢視圖片輸出後，選一至三張檔案核准；資料庫保存核准文字、名稱、圖片路徑與 SHA。
- 圖片產生沿用 `--media-type image --execute`。候選圖在創作者選定前由既有圖片流程保管，不額外新增一套候選資產服務。

### 第 3 階段：從核准故事拆每集節點、場景與鏡頭（已實作）

- 每集文字按順序說明在系列中的目的、因果節點，再拆場景和鏡頭；鏡頭依序寫目的、開場狀態、動作節拍、道具變化、構圖／移動和結尾狀態。
- 自由文字草稿經人修改和核准，保存在 `series_episode_outlines`，每份明確記錄上游故事與角色卡 ID、SHA；不新增 schema 驗證層。
- 後續集規劃會帶入最近兩集核准規劃的尾段，協助接續狀態。影片入口以核准分鏡全文作為本集內容。

### 第 4 階段：接回既有影片流程（已實作）

- `--serial-mode --series-id` 可從資料庫載入角色、身份參考、集數和核准分鏡；新 series 缺故事、角色、參考圖或首集分鏡就會停止。
- `--serial-auto-next` 先恢復 blocked episode；否則依角色 YAML 的 `generation.serial_video.series_selection_weights`，以預設 `new: 0.5`、`continue: 0.5` 選擇新系列或續接符合條件的舊系列。沒有符合條件的舊系列時固定選新系列，建立完整故事草稿並停在人審；故事、角色、參考圖和首集分鏡核准後才建立 production series。舊有未導入 onboarding 的 production series 可繼續使用目前的集數 prompt。
- 既有每段技術 QA、Discord 人工接受、核准尾幀接續、拒絕／逾時／SHA 不符 fail-closed、成片審核和平台 receipt 均沿用。
- 每集新增 `episode_onboarding_lineage`，固定核准故事、角色卡、分鏡、圖片 SHA。production series 只在第一集開始時建立。

### 第 5 階段：版本失效、發布追蹤與營運手冊（程式接線已實作；需要實際模型試作）

- 故事或角色核准新版時，相關待審／核准角色卡或分鏡會 supersede；舊身份參考、目前集數、segments、renders 和尾幀會標 stale，舊完成影片不能直接續接。
- 發布沿用現有 `episode_publications` 帳本，透過 episode ID 回查 `episode_onboarding_lineage`。已送達平台的歷史 receipt 仍保留供查核。
- `docs/serial-video.md` 已加入故事、角色、參考圖、分鏡、核准與生成命令。實際 LLM、ComfyUI/GPU、Discord、社群平台試作尚未執行；程式契約測試不代表這些外部端點已驗收。

## 資料與程式落點

| 項目 | 已實作 | 後續界線 |
|---|---|---|
| 系列故事 | `series_story_plans` 保存可讀故事、新聞快照、審核決定和 SHA | LLM 生成的故事仍需創作者看稿確認 |
| 角色 | `series_character_cards` 保存偏好、核准身分卡、名稱、審核決定、參考圖路徑與 SHA | 圖片沿用既有 `--media-type image` 流程；創作者負責挑選，未新增候選圖審核服務 |
| 單集與分鏡 | `series_episode_outlines` 保存每集故事節點、場景、鏡頭和上游版本 SHA；核准全文交給既有 serial planner | 生成內容品質需以實際模型試作和人看結果驗收 |
| 每集版本關聯 | `episode_onboarding_lineage` 綁定故事、角色、分鏡、參考圖和雜湊 | 已發布 receipt 經 episode_id 回查；歷史已送達內容不會被撤回 |
| 影片審核 | `render_attempts`、`reviews`、`reference_assets` | 沿用目前每段人工接受和 accepted-tail fail-closed 機制 |
| 發布 | `episode_publications` | 沿用逐平台審核內容、素材 SHA、成功／失敗與 receipt |
| 執行入口 | `app/main.py` 提供 story、character、episode outline 草稿／審核；新 series 開始前 gate 所有核准 | `serial-auto-next` 依設定抽選新系列或續作；無候選舊系列時啟動 story-first onboarding 並停在人審 |
| 來源新聞 | `NewsContextService.select_news(random_inspiration=True)` 重用現有新聞查詢和安全過濾 | 新聞庫沒有可用素材時不切換成純即興；自訂方向需明確關閉新聞 |

草稿與核准資料先存在獨立 onboarding 表，不會提前建立可續拍的 `series`。第一集開始時將核准角色與參考圖寫入既有 production 表，並在 episode lineage 連結上游版本。上游故事或角色改版會將目前下游集數和參考素材標 stale；媒體檔與平台 receipt 仍保留。

## 外部參考與採用界線

GitHub 星數是 2026-10-24 查詢時的快照，只用來符合「500 星以上」的參考門檻，不代表品質排名：

| 來源 | 查詢星數 | 可借鏡的部分 | 不照搬的部分 |
|---|---:|---|---|
| [Blizaine/Maestro](https://github.com/Blizaine/Maestro) | 827 | 長故事使用完整故事聖經、因果大綱、章節與有限範圍的場次分批；保持角色、世界規則、狀態連續 | 不搬它的 WanGP、UI、影片供應商或結構化修復系統；目前僅將其當故事規劃參考 |
| [OpenAI Cookbook: Sora 2 Prompting Guide](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide) / [openai/openai-cookbook](https://github.com/openai/openai-cookbook) | 76.8k | 單鏡頭採一個明確動作與一個主要鏡頭移動；寫清楚動作節拍、照明與色彩錨點；圖片做身份與第一幀參考 | Sora API、影片規格與模型專屬參數不套用到 MiniMax H3 |
| [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) | 79.5k | 提示詞清楚說明目的、上下文、工作步驟與讀者能看懂的輸出順序 | 不把一般提示範例當成連續劇情保證 |
| [f/prompts.chat](https://github.com/f/prompts.chat) | 174.3k | 尋找創作語氣、角色設定與靈感寫法 | 不以自由創作 prompt 代替故事版本、因果狀態和人工核准 |
| [guidance-ai/guidance](https://github.com/guidance-ai/guidance) | 21.8k | 僅供理解結構化生成的選項 | 本需求明確不做 schema 驗證，因此不加入依賴 |
| [567-labs/instructor](https://github.com/567-labs/instructor) | 14.1k | 僅供理解模型輸出驗證的業界模式 | 本需求明確不做 schema 驗證，因此不加入依賴 |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 148k | 多步驟依賴需要清楚傳遞上一階段輸出 | 目前 `WorkflowRunner` 與 `ExecutionPlan` 已負責流程與依賴，不另裝 LangChain |
| [0xsline/StoryGen-Atelier](https://github.com/0xsline/StoryGen-Atelier) | 999 | 分鏡文字／圖片、相鄰鏡頭轉場與最終組裝分層 | README 描述的是 storyboard-to-video，不是完整故事與角色審核；它另需 React/Node、Gemini 與 Vertex AI，不移植其服務堆疊 |

[Wikipedia: Film treatment](https://en.wikipedia.org/wiki/Film_treatment) 與 [Wikipedia: Step outline](https://en.wikipedia.org/wiki/Step_outline) 只作術語參考：treatment 是完整故事的散文處理稿；step outline 才逐場列出故事；beat sheet 聚焦主要戲劇轉折。它們不作為新聞事實來源或自動故事規則。

## 風險、邊界與回復

- 隨機新聞只在通過現有標題、類別、時間與可追溯網址過濾後抽選；素材不足時明確停止。若新聞 MySQL 不可用，使用者可在提供自訂想法後明確關閉新聞模式。
- 寫程式與 MySQL 契約測試不代表 LLM 寫出的故事可用，也不代表 ComfyUI/GPU 參考圖或 H3 影片品質已通過。每階段都要保留人工審核及實際產物驗收。
- 第一階段不呼叫實際 LLM、圖片或影片生成。實際模型／GPU 驗收會在流程串接完成且使用者明確啟動一次試作時進行。
- 新增資料表與草稿記錄可以保留；正式系列只在故事、角色、圖片和首集分鏡核准後建立。上游改版會保留舊資料並標記 stale，不刪除媒體檔或發布 receipt。

