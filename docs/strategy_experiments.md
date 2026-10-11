# 生成策略與受控實驗

擴充目標是 **社群內容的生成策略**：讓同一個主題或角色產生不同的視覺敘事，
交付能接入現有審查與發布流程的圖片、GIF 或短影片。`game_sprite` 是內容形式的
參考；候選應以成片的觀看體驗與匯出效率驗收。動態漫畫已有下述受控實驗入口；
`motion_comic` 已接入正式生成與隨機路由；其他候選仍在實驗階段。

## 近期 GitHub 來源

研究期間為 2026-09-11 至 2026-10-11。星數、建立時間、推送時間來自
2026-10-11 GitHub REST API，能力和限制核對官方 README。本月新成立與
既有且本月活躍分開；累積星數不是近月成長量，推送時間不是版本發布時間。
沒有歷史星數的地方保持未知。

| 原始 repo | 星數快照 | 最後推送 UTC | 分類與可借鑑能力 |
|---|---:|---|---|
| [FilmCraft](https://github.com/storytold/filmcraft) | 8,669 | 10-11 | 09-30 新成立；確定性剪輯時間、可驅動命令。年輕專案，整套 Rust 編輯器不適合替換現有工具。 |
| [OneTake](https://github.com/feitangyuan/onetake) | 2,087 | 10-10 | 09-26 新成立；跨段保留物件和連續鏡頭。PolyForm Noncommercial，借鑑方法，不移植程式。 |
| [ReelMimic](https://github.com/edenfunf/reelmimic) | 2,054 | 10-09 | 09-28 新成立；參考片的鏡頭長度、色彩和節奏，修正附前後證據。MIT。 |
| [Lemo-Opuscar](https://github.com/lemomo-ai/lemo-opuscar) | 1,812 | 10-10 | 09-26 新成立；43 種程式電影風格和導演指南。MIT；示範不證明目前文字模型能重現。 |
| [Awesome Opus videos](https://github.com/yihui-dev/awesome-opus5-5-videos) | 3,919 | 10-08 | 09-27 新成立；畫面和 prompt 探索入口。合集不是效能或品質證據。 |
| [Agent Sprite Forge](https://github.com/0x0funky/agent-sprite-forge) | 4,377 | 10-06 | 活躍 sprite 工具；同一主圖生成動作組、時間與引擎匯出。MIT。 |
| [sprite-gen](https://github.com/aldegad/sprite-gen) | 2,702 | 10-09 | 活躍 sprite 工具；週期、接縫、調色和場景組合。Apache-2.0。 |
| [Aseprite AI Artist](https://github.com/with-pebbly/aseprite-ai-artist) | 247 | 10-08 | 活躍利基工具；像素畫分層、動畫與觀察後修正。工具 MIT；需 Aseprite 和 Node。 |
| [Stage Gen](https://github.com/softmarshmallow/stage-gen) | 22 | 10-06 | 活躍小型參考，不能稱為熱門；sprite／portrait／parallax 契約。BSD-3-Clause。 |
| [HyperFrames](https://github.com/heygen-com/hyperframes) | 60,366 | 10-11 | 既有熱門且活躍；可 seek 的 HTML 動畫轉確定性 MP4。Apache-2.0；需 Node 22+，本輪不增加 runtime。 |
| [Remotion](https://github.com/remotion-dev/remotion) | 63,067 | 10-10 | 既有熱門且活躍；可重用動畫資產和程式渲染。特殊商業授權，不以 MIT 稱之。 |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 136,800 | 10-11 | 既有熱門且活躍；沿用目前 GPU 邊界。GPL-3.0。 |
| [StableAvatar](https://github.com/Francis-Rings/StableAvatar) | 1,263 | 09-25 | 近月有活動；音訊驅動角色。MIT 程式不涵蓋所有權重／素材；低 VRAM offload 伴隨慢推理。 |
| [ComfyUI SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) | 576 | 08-24 | 不在近月活躍範圍；分割／追蹤參考。授權 API 回傳 NOASSERTION，不視為已核可相依。 |
| [EasyCache](https://github.com/H-EmbodVis/EasyCache) | 300 | 05-12 | 不在近月活躍範圍；Wan／Hunyuan 加速，未確認原生 H3 適用。Apache-2.0。 |
| [KsanaDiT](https://github.com/Tencent/KsanaDiT) | 63 | 05-13 | 不在近月活躍範圍；DiT 快取和量化，未確認 H3 路線。Apache-2.0。 |

本專案的推論是讓模型生成素材、讓程式控制攝影和時間，以增加畫面種類並
保留角色一致性。星數不驗證這個推論，必須看固定條件的成片。

## 優先擴充的社群內容策略

| 優先 | 候選 | 社群上的可見結果 | 現有基礎 | 固定條件驗收 |
|---|---|---|---|---|
| 1 | 角色視覺笑點／連鎖反應短片 | 一個明確動作引起意外結果，角色反應形成收尾；適合時再做自然循環的 MP4／GIF | Krea／H3、角色設定、動作導演、FFmpeg | 固定角色、主題、模型、seed、片長與生成預算，比較結果可讀性、角色一致性、接縫及總耗時 |
| 2 | 動態漫畫 | 有起因、轉折與反應的分格漫畫短片；局部動作與精確字幕共同說故事 | Story Card、editing／motion graphics | 固定故事、原始圖、文字和片長，對比現有圖片剪輯；檢查敘事理解、文字可讀性與節奏 |
| 3 | 參考節奏剪輯 | 將參考片的節奏和鏡頭語法用在自己的素材，產生新的社群短片 | reference_video、EditPlan／DramaPlan | 固定全部素材與片長，只改剪輯；檢查連續性、匿名比較節奏及故事清晰度、記錄耗時 |

第一個候選的創意例子：角色想吹出一個小泡泡，泡泡反而把角色托起；破掉時
角色落回原位，以表情反應收尾。畫面中的互動是影片的故事事件。作品直接交付
影片或 GIF，並沿用既有角色、素材生成、媒體 QA 和審查流程。

現有動作導演已要求起勢、動作與結果；新候選必須在「連鎖因果與可見反差」上
形成可辨認的新作品，不能只增加泛用 prompt 或換個策略名稱。

**User Given** 一個主題、角色設定與固定生成條件，**When** 明確選擇角色視覺
笑點策略，**Then** 取得呈現起因、意外結果及角色反應的 MP4／GIF，附生成設定、
素材來源與 QA，並能由既有審查入口檢視。缺少必要素材或失敗時留下明確失敗
紀錄；未經審查的結果不自動發布。

先用現有生成路線做配對成片實驗，確認這個形式真的產生可讀的故事，再決定
是否值得增加專用策略。第一層驗收以成片、來源紀錄和審查結果為準。

自然循環是獨立的後處理候選。[sprite-gen 原始驗證](https://github.com/aldegad/sprite-gen/blob/main/docs/video-pipeline.md)
提醒半週期相似姿勢可能導致錯誤接縫。固定來源與裁切範圍比較首尾連續性與
可見動作；保留故事需要的單次收尾，不用反向播放掩蓋接縫。

暫緩 audio avatar、3D character 與新 DiT backend：需要新模型、資產和
conditioning／時間契約。EasyCache／KsanaDiT 沒有證明本機 H3 支援，
不能採用其 README 加速倍數。

## 每輪比較的固定契約

第一輪社群成片實驗選擇「分格逐步揭示漫畫」：同三張圖、同字幕和閱讀時間，
對比每段只顯示一張圖的圖片剪輯。漫畫版依序保留前格，最後以較大的結果格
收尾。沿用 Pillow 與現有 FFmpeg EditPlan，先驗證這一種呈現方法。

**User Given** 三張已完成圖片、標題、字幕及每格時長，**When** 執行比較入口，
**Then** 取得兩份同尺寸、FPS、片長的 MP4、來源 hash、QA、交錯計時與待填
觀感評分。缺少圖片、非法時長、過長字幕或既有輸出目錄，在渲染前拒絕。

1. 改動前寫 **User Given / When / Then**，定義社群媒體產物、審查入口與失敗邊界。
   尺寸、時長與封裝依目標發布渠道及現有契約設定。
2. 保存來源 SHA-256、新聞／故事快照、prompt、model／workflow、seed、steps、
   解析度、片長、FPS、候選數、conditioning 和 review 決策。
   相同 seed 不能保證不同 backend 逐位元確定。
3. 一輪只改一項；先 focused contract，再實際生成／渲染。
4. 暖機後交錯 A/B 順序，計時比較序列執行，避免搶 CPU／GPU。
   生成／後處理分開計時，記錄失敗率，不只挑漂亮結果。
5. 效率至少三次測量，保留原始紀錄和中位數。品質用至少五組配對 seed
   與匿名 A/B 比較；初期樣本用於篩選，不視為統計定論。
6. 硬性 QA 檢查尺寸、FPS、片長、alpha、可播放性；人眼另評角色身分、動作、
   故事因果和節奏。畫面相似度不能代替美感，技術 QA 不能代替理解與喜好。
7. 未量測的品質／GPU／VRAM／社群表現填未驗證；沒有受眾實驗，不能宣稱
   互動率或留存改善。成片與既有審查／發布封裝驗收充分後，
   才加入角色與排程器的隨機候選。

## 第一輪：分格逐步揭示漫畫（motion_comic）

入口是 `scripts/run_motion_comic_experiment.py`，沿用 `media.compose_edit`、
`media.video_qa` 和現有 CJK 字型載入。輸入 JSON 包含 `title`（1–12 字）、
三個 `panels`；每格具有 `image_path`、`caption`（1–24 字）。合成時會按實際字型、
欄寬與畫格高度換行，保留插畫空間並盡量維持畫布對應的字級，不固定行數。
及 `duration_seconds`。相對圖片路徑依 JSON 所在目錄解析。
`width`／`height`／`fps` 預設 576×1024／24，閱讀時間須對齊影格。

正式策略已登錄為 `--generation-type motion_comic`，Kirby 設定已有正隨機權重。
正式入口由 LLM 寫故事、Krea 生成三張圖並合成短片，無需既有 JSON；見
[自動生成使用指南](motion_comic.md)。本節的比較入口固定使用已完成圖片與
字幕，以隔離呈現方式對品質和效率的影響，仍要求 `--story`，不執行發布。

快速使用已得到正面回饋的修訂素材，從 repo 根目錄執行：

```powershell
python scripts/run_motion_comic_experiment.py --story output/motion_comic_story_revision_20261011/story.json --output-dir output/motion_comic_try_01 --repeats 1 --warmup 0
```

輸出目錄必須尚未存在。換故事時複製修訂版 `story.json`，更改標題、三個
`image_path`、`caption` 與 `duration_seconds`；先用相同參數生成及檢查素材。
需要正式測速時用 `--repeats 3 --warmup 1`，暖機排除、三組序列交錯測量。

輸出是 `slideshow.mp4`、`motion_comic.mp4`、對照接觸圖、EditPlan manifest
與 `report.json`。兩版標題、字幕、字重、每格閱讀時間、兩格淡入轉場、尺寸、
FPS 與來源固定，只比較每段單圖和逐步保留前格的漫畫呈現。
目前為分格揭示，角色身體仍來自靜態插畫；沒有宣稱完成局部肢體動畫。

暖機後交錯三次，各版分開計時。時間包含頁面組合、編碼、接觸圖及技術 QA；
共用插畫的生成、預檢、最後檔案複製不在該計時內。音軌沿用剪輯器產生的相同
靜音 stereo AAC，這一輪比較視覺，沒有新增配樂或旁白。

原始素材故事是「只吃一小口」：想吃蛋糕、奶油爆發、頂著奶油皇冠開心收尾。
Krea2 Turbo 同一 workflow、1024×768、8 steps，共生成五張、約 126 秒；初批
第二／三格未呈現要求的事件與結果，簡化提示後以原本各格 seed 重做。
所有嘗試保存在 `output/motion_comic_20261011/generation.json`，兩版使用
完全相同的最終三張插畫。這是單故事的格式試驗，不是跨模型品質比較。

使用者接受整體圖文比例，但否決故事：第一格沒有吃的一小口，奶油噴出缺少
觸發事件，奶油皇冠也無法支持「好像也不虧」；標題沒有得到內容回應。
原始素材保留為失敗案例，媒體 QA 與測速通過不改變敘事否決結果。

### 故事素材的驗收順序

**User Given** 一個標題與三格故事，**When** 生成素材並配字，**Then** 每句
字幕的事件有畫面證據，前後格呈現同一主體的狀態變化，最後回應標題的承諾；
缺少這些證據的素材須先修改故事或重做，不用新字幕補造未出現的動作。

先為每格寫「可見動作、關鍵物件的前後狀態、字幕、否決條件」。共用提示只
描述角色、畫風與場景；變動物件的狀態留在各格提示中，避免每格都要求完整
蛋糕，卻又在收尾要求空盤。生成後先逐圖、再並排、最後看編碼成片：

- 暫時遮住字幕，能否辨認發生什麼以及狀態如何改變？
- 每句文字是否由該格可見證據支持，是否描述了圖上沒有的動作或收益？
- 主體、道具與場景是否延續，標題的承諾是否在結尾被實現或形成反差？

這是目視敘事檢查，另留 review 記錄；尺寸、FPS、SHA-256 和本機測試不提供
語義證據。單一故事修訂也不證明整體品質或社群成效提升。

本次修訂的生成前驗收在
`output/motion_comic_story_revision_20261011/storyboard.md`：小口試吃 → 忍不住
續吃、蛋糕明顯變少 → 空盤與嘴邊奶油。版面、模型、步數、各格 seed、尺寸和
閱讀時間沿用；兩種剪輯仍共用同組修訂素材。

修訂成片在 `output/motion_comic_story_revision_20261011/comparison_serial/`，
第一格有一小塊的準備線索，第二格正在吃、蛋糕變少，第三格空盤與嘴邊奶油。
第一格三次仍未生成叉子接嘴，因此記為承諾起點，沒有把該動作目標改判通過；
逐格觀察與淘汰理由保存在 `narrative_review.json`。2026-10-11 使用者回饋
「我覺得還不錯唷」，記為這個修訂試片的正面觀感，尚非多故事品質驗證。
這輪七次生成約 179.74 秒，原版五次，不能宣稱等預算的生成效率提升。
修訂素材的後製交錯三組中位數：單圖 2.686 秒、漫畫 2.796 秒；兩版 QA 通過，
尺寸、FPS、片長與來源 hash 核對完成。渲染程式未改，沒有重跑本機完整回歸。

漫畫能留下前因與結果，但同屏多格會縮小插畫。接觸圖與成片用於檢查手機
文字、表情和事件可讀性。技術結果與觀感分開：`perceptual_review.status`
維持 `awaiting_review`，不因媒體 QA 通過就自動加入生成或排程候選。

2026-10-11 正式證據在 `output/motion_comic_20261011/comparison_serial/`。
暖機一組、交錯三組，後製中位數為圖片剪輯 2.852 秒、漫畫 2.878 秒，耗時接近；
沒有加速證據。兩版均為 9 秒、576×1024、24 FPS，QA 通過、來源 hash 未變。
較早的 `comparison/` 含細字重與並行測試診斷，排除於正式效能結論。

本輪完整本機非 integration 回歸：681 passed、1 skipped、2 deselected。
新增 CLI 的 12 個真實 Pillow／FFmpeg 驗收覆蓋成功成片、前格保留與失敗時不覆寫。
另外直接對 CLI 蒐集覆蓋，分支 47/52（90.4%），綜合行／分支為 95.8%。
未觸發的是長標題縮字、計量次數越界、QA 失敗、來源中途變更，以及 import
模式入口。前兩者未在本輪素材／參數觸發；後兩個失敗分支需要實際渲染錯誤或
外部改寫，沒有用 mocks 製造。入口僅以 CLI 執行驗收。覆蓋紀錄在
`output/motion_comic_20261011/coverage/`；沒有把 pytest 未捕捉 subprocess 的
空覆蓋報告算作成功。

真實 ComfyUI／Krea T2I 和本機封裝已執行；這輪沒有使用 H3／LLM 編排、重跑
Docker 排程或發布到社群，也沒有觀眾測試。下一個品質判斷需用更多配對故事和
觀看評分，不能由這一份試片推論互動率。

## 已實作：sprite 單次解碼取樣

`SpriteAdapter.video_to_sprite` 保留 16 個區間中點時間和匯出設定，以 FFmpeg
`split`／`trim` 一次解碼完成。單張取樣仍服務一般 review frame；sprite
不保留逐張解碼 fallback。拒絕不合法時間、重複目的路徑和缺少影格，先在
暫存輸出驗證全部存在，再寫入目的地；時間記錄於 manifest 的 `sampling`。

真實 FFmpeg／Pillow 驗收涵蓋定速／變速 PTS、重複時間戳、EOF，
atlas／GIF／WebP 與逐張取樣完全相同，未呼叫生成或社群 provider。

```powershell
python scripts/benchmark_sprite_sampling.py --video "D:/media/source.mp4" --output-dir "output/benchmarks/my_sprite_trial" --repeats 3 --warmup 1
```

目的目錄必須不存在。`--cell-size`、`--fps`、`--key-color`、`--threshold`
在兩組使用相同值。Report 保存環境、來源 hash、probe、時間、順序、
取樣／封裝耗時、像素一致數、匯出 hash、QA 和錯誤。暖機不計入中位數。
本輪結果在 `output/benchmarks/sprite_sampling_20261011/`，不把單機數字當 SLA。
此實驗只驗證匯出效率，沒有生成新的 H3 片或宣稱觀感提升。
三個來源各三次正式比較的中位數：取樣快 4.08–4.37 倍，完整封裝省 24–29%
時間；全部 16 張影格與 atlas／GIF／WebP 保持一致。原始 report、對照圖及
validation 在該目錄，序列執行的 `serial/` 才是正式計時證據。

取樣優化完成時的本機非 integration 測試為 669 passed、1 skipped、2 deselected。
新增 batch 方法量測的 18 個分支全部執行；工具整體分支覆蓋仍低於 90%
診斷目標（FFmpeg 62.0%、Sprite 68.1%）。既有音訊／overlay／concat 錯誤、
sprite one-shot、輸入與色彩解析分支未在這輪擴充，詳見實驗 validation。
