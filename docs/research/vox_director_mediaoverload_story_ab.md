# Vox Director 劇情節奏研究與 MediaOverload A/B

狀態：新聞 A/B 以 10 組為目標，實際 8 組配對完整並通過技術 QA，另兩組無法計入；人工評分 CSV 仍空白，研究不能證明弧線提示改善整體劇情。2026-09-26 依使用者要求已更新 Native H3 與長片的故事提示／拍點契約；新契約尚未用新成片驗證。

## 驗收情境

**User 想用 Kirby 製作一段短片。** Given 同一角色、故事 brief、seed、時長、渲染路由與 QA 規則，When 套用目前 prompt 或加入依題材選擇的故事弧線，Then 渲染影片應保留原故事，而處理組能更清楚呈現因果推進與結尾 payoff。

## 研究範圍與來源

研究對象是 [Alisa0808/vox-director](https://github.com/Alisa0808/vox-director)。本研究檢查其 README、`SKILL.md`、beat-layer 參考、範例 beat map、授權條款與展示影片；只提取可移植的編劇方法，沒有搬入它的程式、媒體素材或雲端服務依賴。

此 repo 是 MIT 授權的 agent skill，將主題製作成旁白拼貼影片，使用 Atlas Cloud 與本機 FFmpeg。流程以 `beats.json` 為中心：先選故事弧線、逐拍撰寫場景並審核 beat map，再製作 keyframe、動態、旁白、音樂與剪輯。B-roll、A-roll、C-roll 共用這種分階段編排方式。

## 可移植的編排方法

最有用的區分是把 **故事弧線、每拍內容、鏡頭執行** 分開。beat library 提供 14 種弧線，依主題類型提出候選，例如機制解說用 `how_it_works`、逆境後轉變用 `man_in_hole`、單一喜劇點子用 `hook_payoff`。弧線選擇與逐拍審核由 agent／人負責；下游腳本執行已寫好的 beats，不能據此證明自動弧線分類會改善影片。

節奏建議包括前三秒內提出 hook、縮短中段並保留 payoff、每三至五秒安排視覺變化。15 秒 money 範例用三個五秒段落，並停留在最後的「信任」意象；30 秒 Tang 範例用三個十秒段落，每段再配遠景與細節鏡頭。兩支展示片的 contact sheet 可看到畫面主題隨旁白切換，最後收在較安靜的觀點畫面。

這些案例支持「逐拍控制」作為設計假設。Repo 提到的三秒留存數字與「最高評分弧線」沒有附上專案受眾實驗，展示影片也不能代表觀眾偏好。每三至五秒切鏡的建議不能直接套到單一連續生成的 Native H3 影片。

## MediaOverload 現況與實驗目標

MediaOverload 已有共用的因果動作順序：hook、mechanism、consequence、reaction、payoff；`hypit_v1` 可補上定時 cue window。Native H3 15 秒 recipe 已有 hook、escalation、payoff 三個時間槽，現有 quality brief 也要求 setback 與 reversal。30／45 秒 storyboard 更已有失敗嘗試、重新理解、決定性 payoff 與 aftermath/coda。

當時的 A/B 刻意不重寫通用因果順序、不把 14 種弧線接入日常 production，也不改 Native H3 15 秒新聞路由；它只測「依題材選弧線並寫出階段順序」是否比當時 prompt 更易看懂。2026-09-26 的後續優化則已把 14 種弧線背後意圖及中性拍點指引寫進生成提示與模板。由於 A/B 固定了 15 秒三拍時長與鏡頭節奏，不能用它驗證 Vox Director 的前三秒 hook、每三至五秒視覺變化或 beat 時長比例。

### BDD 情境

- **User 想呈現一個短喜劇 payoff。** Given brief 與渲染設定相同，When 使用 `hook_payoff`，Then 開場動作應提出同一個結尾要解答的問題，最後一拍應解答它。
- **User 想說明一個可見機制。** Given brief 與渲染設定相同，When 使用 `how_it_works`，Then 影片應先按因果順序呈現機制如何運作，再呈現它帶來的結果。
- **User 想呈現角色受挫後復原。** Given brief 與渲染設定相同，When 使用 `man_in_hole`，Then 第一次失敗應造成可見困境，角色再依線索改變策略，最後完成同一個目標。
- **User 想避免無效實驗繼續消耗渲染資源。** Given 某變體未通過技術 QA，When runner 記錄該變體，Then 它應保存失敗證據並停止提交後續變體。
- **User 想讓 10 組新聞試驗的完整分鏡有足夠生成時間。** Given 本機 Qwen 已固定且該次生成需要超過 repo 預設 300 秒，When runner 開始或續跑故事生成，Then 單次 request 應允許至少 600 秒且總 deadline 應高於該限制；runner 在技術失敗後停止後續提交，之後只有在操作者明確使用 `--resume-dir` 時才會重試失敗變體，並將每次嘗試分開留證。

## A/B 測試方法

- A 使用配對 brief 與兩組共用主角限制，不附加題材弧線。B 只額外加入一段依題材選擇的故事弧線指令；A 的實驗標籤只指此控制內容，不代表重現了 MediaOverload 所有現行 prompt 層。
- 共三組 matched pairs、六支影片：lunchbox 喜劇 `hook_payoff`、風車點燈機制 `how_it_works`、紙船受挫復原 `man_in_hole`。
- 每組固定 brief、Kirby、單一主角、seed、原生六秒時長、畫布、`text2image2video → Krea2 → MiniMax H3` 路由與 hard media QA。文字規劃固定使用本機 Ollama Qwen 模型；本次執行另用背景 watcher 在 brief 完成後卸載模型，釋放 VRAM 給 ComfyUI。A/B 不發佈、不 dispatch。
- 三組交錯 A/B 執行順序，減少先後順序的影響。紀錄輸入 prompt、provider/model 標籤、seed、`run_config.json`、`variant_config.json`、`variant.json`、`pair.json`、`benchmark_summary.json`、下游 workflow 輸出路徑、ffprobe、contact sheet 與 QA 結果，媒體證據存放在 repo 外。Runner 在變體技術失敗時保存當下證據並停止後續提交。
- 人工檢視六格 rendered-frame contact sheet，以 0–4 分記錄：因果推進（主要指標）、開場可讀性、弧線適配、payoff、角色／場景連續性、整體畫面品質。0 代表缺失或矛盾；2 代表部分可辨但有斷點；4 代表清楚、連貫且有畫面證據。另記錄身分、地理位置或物件連續性的瑕疵。
- 只有三組 B 都在因果推進勝過 A、每組至少一項弧線適配或 payoff 提升、hard QA 相同通過，且沒有明顯連續性或整體畫質退步，才採用 production 變更。此小樣本用來判斷畫面故事是否更清楚，不推論受眾留存或偏好。

## 執行進度

- 三組 matched pairs 共六支影片皆完成。六支影片均使用固定本機 Qwen planner、模型 pin 相符、prompt generation 成功且 hard media QA 通過；同組共用 brief、Kirby、seed、原生六秒時長與渲染設定。原生輸出均為 6.583 秒，speed variant 均為 3.292 秒。
- 人工分數（0–4；前六項依序為因果推進／開場可讀性／弧線適配／payoff／連續性／畫面品質）：

  | 配對 | A 分數 | B 分數 | 判讀 |
  |---|---|---|---|
  | lunchbox `hook_payoff` | 2／3／2／2／3／3 | 3／3／3／3／3／3 | B 因果與 payoff 各 +1；餅乾落點的最後畫面仍未清楚呈現。 |
  | windmill `how_it_works` | 2／2／2／2／3／3 | 3／3／3／3／3／3 | B 因果與 payoff 各 +1；燈籠亮起較可讀，但風如何點亮燈芯仍未完整顯示。 |
  | paper boat `man_in_hole` | 2／3／1／2／3／3 | 2／3／2／3／3／3 | B 的弧線與 payoff 各 +1，但因果推進打平；紙葉橋最初失敗沒有在畫面中出現。 |

- B 在主要因果推進指標勝出 2／3，payoff 三組均不退步，沒有連續性或整體畫質分數退步；然而預先設定要求因果指標 3／3 勝出，故不採用通用 production prompt 變更。現有證據支持在喜劇 payoff／可見機制題材繼續做有針對性的試驗，不足以證明 setback/recovery 弧線已能穩定落到畫面。
- 可重跑入口：[scripts/run_vox_director_story_arc_ab.py](../../scripts/run_vox_director_story_arc_ab.py)。逐組 prompts、run config、variant/pair/benchmark JSON、workflow summaries、影片、contact sheets 與 QA 記錄位於 `E:\comfyui\_extra\benchmarks\vox_director_story_arc_ab`；三組彙總為 `E:\comfyui\_extra\benchmarks\vox_director_story_arc_ab\aggregate_benchmark_summary.json`。這些大型媒體與執行產物刻意留在 repo 外。
- 重跑紀錄目前固定 model tag `qwen3.8-27b-ud-q2xl-local:latest`，但未保存 Ollama model digest 或實驗當下 repo dirty-file hashes；若 tag 或相關程式變更，之後重跑不等於位元級重現。
- OpenRouter Gemma 遭上游共享池 429、Gemini 顯示預付額度耗盡、Mistral 回傳 429 的嘗試均因 prompt lineage 降級而排除；它們不算有效樣本。後續採 repo 已安裝的本機 Ollama Qwen，且 runner 遇到降級時會將該變體標為失敗。
- Production prompt 未改動。人工判讀以 QA contact sheets 為主；paper-boat A/B 另看 10 格、每秒 3 格的 contact sheet。這是三個案例的小樣本影像評估，不能推論觀眾留存或偏好。

## 新聞弧線驗證：10 篇新聞、10 種弧線

先前三組是虛構情境，不能代表新聞影片。本輪以 10 篇可查證新聞各建立一組 A/B 為目標，固定同一篇文章、Kirby、單一主角、15 秒 Native H3 T2V、seed、模型與 hard QA；A 使用當時的 Native H3 新聞故事契約，B 只多一條按題材選定的弧線指令。配對順序交錯，避免 A 永遠先生成。新聞摘要是研究者對來源的短摘要，並保存來源 URL、標題、日期與 manifest hash；實際只有 8 組完成配對。

實驗涵蓋 `hook_payoff`、`how_it_works`、`timeline`、`man_in_hole`、`story_spine`、`origin`、`myth_buster`、`listicle`、`three_act`、`story_circle`，各對應一篇不同新聞。完整案例與弧線指令在 [vox_director_news_arc_cases.json](vox_director_news_arc_cases.json)，rerun 入口是 [run_vox_director_news_arc_ab.py](../../scripts/run_vox_director_news_arc_ab.py)。runner 可在 repo 外產生 A/B 原片、逐組六時點左右並排表、HTML 播放頁及人工評分表；CSV 將「來源事實吻合」列為主要護欄，避免只因畫面節奏變得順而掩蓋新聞內容跑題。

Vox Director 列出的 14 種是 `hook_payoff`、`pas`、`bab`、`aida`、`storybrand`、`how_it_works`、`timeline`、`man_in_hole`、`story_spine`、`origin`、`myth_buster`、`listicle`、`three_act`、`story_circle`。案例 manifest 原規劃 10 種新聞題材弧線，但正式 run 只有 8 種形成完整配對；`pas`、`bab`、`aida`、`storybrand` 本輪沒有測，因為它們主要是銷售／CTA 說服結構，不等於新聞故事的通用戲劇弧線。因此現有 run 無法比較 10 種的穩定成效；新版 prompt 讓模型依 brief 選一種意圖，不要求每支片都用戲劇弧線。

採用門檻預先定為：人工 0–4 分中，B 的新聞事實吻合平均分不得低於 A；核心節奏分（因果推進、開場、弧線適配、payoff）平均至少高出 0.5 分，且至少 7／10 配對的核心節奏總分勝過 A；角色／場景連續性與畫面品質平均不得低於 A 超過 0.25 分。未達任一條件就不轉換；這 10 個故事每個弧線各一篇，只能決定是否值得保留「按題材選弧線」這個方法，不能證明哪個單一弧線已普遍勝出。

為讓 matched pair 真正固定同一主角，工作流程新增一項明確角色選擇行為：單主角模式若呼叫端指定 `selected_character_name`，就依指定名稱解析角色，而不再被角色群組的隨機權重覆蓋。兩版共用同一 creative brief；B 版的弧線則透過獨立、可留證的 `native_h3_arc_instruction` 欄位送入 Native H3 故事規劃器，避免與原有 creative brief 安全清理混在一起。空欄位不改變一般 production 輸出。

**有效試驗前有兩次無效先導，正式 run 也有兩個重試。** run `20260925T112037Z` 的已存 LLM prompt 顯示 creative brief 清理器把共用 brief 改成抽象氣氛指令，B 的弧線文字沒有進入故事模型；run `20260925T120311Z` 則確認弧線已進入 request，但 Qwen 在 300 秒 timeout 內未回覆。兩次都不列入成效樣本。正式 runner 修正為獨立傳遞弧線欄位，並將 LLM request / 總 deadline / Ollama request timeout 設為 600 / 660 / 600 秒，仍固定模型且禁止 provider fallback。正式 run 中 Hurricane B 曾遇到 300 秒請求逾時，重試成功；complex-life B 首次回傳無效時間區間，重試成功。兩組最終輸出均通過 hard QA，重試失敗不作為額外樣本。

### 結果與採用決定

> 證據校正（2026-09-26）：請以 run manifest 與每組 pair record 為準，不採用舊版報告中的 10／10 結論。

正式 run `20260925T120953Z` 的 benchmark summary 寫明目標 10 對，實際 **8 對完成且通過技術 QA**，共 9 筆 pair records，`technical_all_pairs_passed=false`。Climate Week 的 pair record 只有 A 變體且 `technical_both_passed=false`；Parker pair 被標成 `superseded_by_source_correction`，變體清單為空。這兩組不能算完成配對。摘要頁雖列出 10 個案例，但不能據此稱 10／10 成片通過。CSV 有 10 列案例，但評分欄仍全空。八組完成片有固定控制資訊、無 fallback、無 dispatch；這只能證明該批輸出通過既有技術 QA，不能證明故事品質變好。

我先按 HTML 頁中的 A/B contact sheet 做了**未盲測的視覺初判**，僅比較節奏／因果，不把技術 QA 當創意分數，也沒有填寫人工評分 CSV。表格列出八組完整配對，並另外標記兩組未完成案例：

| 新聞案例 | 弧線 | contact sheet 初判 |
|---|---|---|
| 玻利維亞新貓種 | `hook_payoff` | B 的開場到揭示較清楚；但兩版都沒呈現基因證據或新物種鑑定這個新聞核心。 |
| NOAA 商業衛星資料 | `how_it_works` | B 的衛星資料流向預報較容易讀；畫面更像地面碟形天線，未清楚呈現商業衛星。 |
| Hurricane Polo 快速增強 | `timeline` | 混合；A/B 各自有可辨識的風暴畫面，但都漏掉 24 小時升至 90 節／Category 5 等新聞關鍵變化。 |
| 密西西比浣熊生存訓練 | `man_in_hole` | 無明確勝者；兩版都沒有清楚呈現浣熊、孤兒幼崽或生存訓練，也看不出跌落—逆轉。 |
| 西澳沙漠野花 | `story_spine` | B 的乾土—降雨—花海因果鏈較清楚；A 的花朵特寫更強。 |
| NASA PRIMA 遠紅外線望遠鏡 | `origin` | 混合、略偏 B 的星球／環狀物構圖；兩版都沒有清楚呈現望遠鏡或任務。此案例是太空遠紅外線望遠鏡，不是海洋觀測。 |
| 耐熱複雜生命研究（校正重跑） | `myth_buster` | B 的起跳—回落—站穩收尾較像完整 payoff；A、B 都沒呈現微生物或實驗，B 的規劃文字還把 70°C 說成熱障礙（來源是短暫恢復測試，80°C 才無法恢復）。節奏可讀性 B 略好，但來源事實護欄失敗，不視為可採用勝出。 |
| 月球新撞擊坑 | `three_act` | B 的光點至坑洞揭示較有 payoff；A 一開始已有坑洞，之後 Kirby 跌入，沒有清楚表現「軌道器發現新坑」。 |
| Climate Week、潔淨能源與 AI 不確定性 | `listicle` | 不計勝負：pair record 缺少可核對的 B 變體，技術配對未完成。 |
| Parker Solar Probe 近日點後回報 | `story_circle` | 不計勝負：source correction 後的 pair record 標記為 superseded，沒有有效 A/B 變體。 |

八組完成配對的未盲初判是 **5 對 B 節奏較清楚、2 對混合、1 對無明確勝者**。其中耐熱複雜生命一對雖然 B 節奏看起來較完整，仍有新聞事實錯置與畫面跑題；因此這不是五個已確認的整體勝出。它不是 CSV 的 0–4 人工分數，也無法計算核心節奏平均是否提升 0.5 分。原先設定的 7／10 採用門檻、平均分與來源事實／連續性／畫質護欄都無法由現有評分檔驗證；這批實驗不能證明普遍成效。

**研究結論：這批 A/B 不足以證明弧線提示讓整體劇情變好。** 使用者其後明確要求依「貼題、拍點執行、14 種弧線的背後意圖」優化，因此目前採用的是較窄的提示詞與結構契約：說清各弧線要解決的敘事任務、依題材選一種、逐拍推進同一問題，並禁止新聞故事補造事件。這是一次有依據的 prompt/template 改善，不是已經由新成片驗證的品質結論；目前沒有以這批舊 A/B 宣稱達標。

Vox Director 的 14 種弧線在本研究中**作為設計參照**。八組完成的新聞 A/B 涵蓋 `hook_payoff`、`how_it_works`、`timeline`、`man_in_hole`、`story_spine`、`origin`、`myth_buster`、`three_act`；新聞 A/B 未完整驗證 `listicle` 與 `story_circle`。`pas`、`bab`、`aida`、`storybrand` 是說服／CTA 結構，這輪沒有測。這次提示詞明列 14 種意圖，但不硬套所有弧線：依來源／創作 brief 選一種，說服類只有在使用者明確要求時才用；沒有新增自動評分 gate，也沒有聲稱新提示已通過成片 A/B。

### 結果檔案

- [A/B HTML 對照頁](E:/comfyui/_extra/benchmarks/vox_director_news_arc_ab/20260925T120953Z/comparison_gallery.html)：10 組案例頁；兩組缺少有效配對結果，不能當成 10 組完整影片比較。
- [人工評分 CSV](E:/comfyui/_extra/benchmarks/vox_director_news_arc_ab/20260925T120953Z/manual_review_template.csv)：10 列案例，但本次讀取時評分仍全空。
- [技術結果與 controls](E:/comfyui/_extra/benchmarks/vox_director_news_arc_ab/20260925T120953Z/benchmark_summary.json)：目標 10 組、實際 8 組完整、技術全配對通過為 false；無 dispatch。實驗固定 model tag 與控制簽章，但未保存 Ollama model digest 或實驗當下 repo／工作樹 hash，不能聲稱位元級重現。

## 來源

- [Vox Director README 與 pipeline](https://github.com/Alisa0808/vox-director/blob/main/README.md)
- [Beat 與 shot library](https://github.com/Alisa0808/vox-director/blob/main/references/beat-layer.md)
- [Agent workflow 與 beats schema](https://github.com/Alisa0808/vox-director/blob/main/SKILL.md)
- [MIT license](https://github.com/Alisa0808/vox-director/blob/main/LICENSE)
- [Gemini API models（確認現行 Gemini 3.5 Flash provider 文件）](https://ai.google.dev/gemini-api/docs/models)
