# Hypit × MediaOverload semantic-production research

研究日期：2026-09-21

## Scope

本研究只移植 Hypit 對 MediaOverload 有直接價值的 production 方法，不引入 Hypit 的整套 TypeScript/SVML runtime，也不把 Hypit 的模型服務或宣傳性成果當成 MediaOverload 的品質證據。

研究來源：

- [Hypit repository](https://github.com/hypit-ai/hypit)
- [Hypit Development Guide](https://hypit.ai/guide/develop/)
- 本機 clone：`9c9918d0cedf2f06574ab0d517b1b6b0afb56a66`

## 可移植的方法

### 1. Semantic source，而不是一次性 render

Hypit 把影片描述成 Brief、Treatment、Script、Timeline、Source、Recipe、Run 與 Result 的可重跑 production source；時間關係以語意事件與文字對齊，而不是把每個物件硬綁在固定秒數。這對 MediaOverload 的直接對應是：保留現有 planner/manifest/RunRecorder，增加一個可序列化的 semantic cue timeline，讓同一份因果故事能被不同 variant 重跑與比較。

### 2. Reusable composition，variant 只替換一個變因

Hypit 的 composition/component/recipe 讓 host、topic、語言、B-roll 或其他素材可以替換而不重寫整個 composition。MediaOverload 不需要複製這個 DSL；本次用 paired benchmark manifest 固定 prompt、route、duration、seed、canvas、模型與 QA，只替換 `semantic_cue_mode`。

### 3. Inspectable preview/review artifact

Hypit 把 preview、snapshot、Run、Result 和人類觀看分開；本 repo 會記錄 deterministic media DQ、GIF/contact sheet 與 Discord review，所以本次只補齊每個 A/B variant 的 `variant.json`、semantic timeline、QA 結果、影片與 contact-sheet 路徑。DQ 結果是檢查證據，不會代替人類的創意判斷；creative winner 仍必須人工觀看，不能由 technical QA 或 planner contract 自動宣稱。

## MediaOverload 實作

- `visual_action_contract.py` 新增 `semantic_cue_timeline()` 與 `semantic_cue_timeline_prompt()`。
- `CharacterGenerationOptions.semantic_cue_mode` 將 B 變因傳到 goal constraints、run manifest 與 prompt builders。
- `hypit_v1` 以 hook → mechanism → consequence → reaction → payoff 的語意窗口注入 video prompt；opening still 不塞入後續鏡頭。
- `scripts/run_hypit_semantic_ab.py` 沿用既有 `text2image2video → Krea2 → MiniMax H3 I2V → deterministic media QA`，建立至少五組 matched A/B pairs。QA 結果是記錄的證據，不是 blocking gate。輸出預設放在 `E:\comfyui\_extra\benchmarks\hypit_semantic_ab`，避免把生成媒體放進 repo。

## 實驗控制與判定

每一組固定：

- 同一個 creative prompt、single-protagonist contract、generation route、duration、ComfyUI endpoint、seed 與模型/工作流設定。
- A：現行 prompt contract。
- B：相同輸入加 `semantic_cue_mode=hypit_v1`。
- 每支影片記錄 deterministic media QA 結果，再用 contact sheet / 實際影片人工檢查 action readability、semantic order/cause-effect、identity/geography、payoff、overall visual quality。這是實驗分類規則，不代表工作流內存在 blocking gate。

只有在 B 在多數 matched pairs 中呈現穩定且使用者可見的改善，才考慮提升為 production default；若沒有穩定優勢，就保留 opt-in 能力與研究證據，不改變現行預設。

## A/B 結果

正式 run：`E:\comfyui\_extra\benchmarks\hypit_semantic_ab\20260921_234621`。這個 run 使用 6 秒、512×512、固定 case seed、single-subject override，A/B 各跑同一個 `text2image2video → Krea2 → MiniMax H3 I2V → deterministic media QA` 路由；B 只增加 `semantic_cue_mode=hypit_v1`。

第一個 run `20260921_232955` 不列入統計：lifecycle 顯示 YAML 的 random interaction selection 仍產生兩個 subject，和 benchmark 的 single-subject 控制不一致。這暴露了 request override 沒有在 selection entry point 生效；已修正 `resolve_character_selection()` 與 workflow selection logging，並以 focused test 和新的 lifecycle log 驗證後重跑。

### 統計規則

- 必須有 A/B 兩支影片、`video-qa.passed=true` 與可讀 contact sheet 才算 technical pair pass。
- `action_readability`、`semantic_order_and_cause_effect`、`subject_identity_and_geography`、`payoff_readability`、`overall_visual_quality` 由 contact sheet/影片觀看填寫；hard QA 不代替 creative review。
- 若 B 只改善 prompt/manifest 可追蹤性，不能稱為影片品質提升。

### 配對結果

正式 run 已完成 5/5 paired E2E；10 支影片的 QA 紀錄皆為 `passed=true`，且每支都有可讀 contact sheet。這些 QA 紀錄是觀察結果，不是阻擋工作流的 gate。原始 H3 輸出均約 6.58 秒、24 fps；下游既有的 2× speed artifact 約 3.25–3.29 秒，這是共同 pipeline 後處理，不是 A/B 變因。

| Case | Seed | Technical | 人工影格觀察 | provisional winner |
|---|---:|---|---|---|
| `lunchbox_snap` | 20303178 | A/B pass | A 還保留 lunchbox→snack 的部分因果，但物件會變形；B 的第一組影格漂移成 star-cookie/另一個廚房構圖，雖有動作，卻不再是同一個故事。 | A |
| `noodle_scarf` | 20311425 | A/B pass | A 有拉伸與最後圍成 scarf 的 payoff；B 的 pull→release→tumble→scarf 狀態交接更容易讀，雖然場景由廚房改成戶外。 | B（動作） |
| `bubble_pop` | 20340426 | A/B pass | A 的泡泡從背景包覆到破裂，但中段形變較重；B 從包覆、泡泡碎散到 Kirby 反應的狀態序列較完整。 | B |
| `mochi_bounce` | 20315408 | A/B pass | A、B 都保留球與槌子的核心互動；B 的推進、碰撞、落地與頭頂 payoff 較連貫，A 較像已從中段開始。 | B（輕微） |
| `pebble_puff` | 20299535 | A/B pass | A 的木槌與石頭畫面乾淨，但擊打結果不清楚；B 清楚呈現擊打、石頭飛出、Kirby 反應與收束，代價是多了白色 puff 且構圖/色調漂移。 | B（動作） |

這是本次本機 frame audit 的 provisional 判定，不取代 Discord/人類創意審查。以 action readability 與 cause-effect 來看，B 在 4/5 組較有利；以 subject identity/geography 與「是否仍是原 prompt 的同一個故事」來看，B 在 `lunchbox_snap` 明顯退步，其他組也常改變場景或構圖。因此不能把 4/5 的動作改善直接宣稱成全面品質提升。

### 結論與下一步

本次採用的 Hypit 方法值得保留：把故事拆成可序列化的 semantic source，使用 hook → mechanism → consequence → reaction → payoff 的事件窗口，並把 timeline、variant、run、QA 與 contact sheet 一起保存。它對簡單、單一主角、單一物理機制的短片，確實提高了「中間發生什麼、最後如何收束」的可讀性。

但本次正式 run 的 `AGENTIC_LLM_MODE=llm` 仍讓 prompt expansion 有非 deterministic 影響；`source_signature` 固定了輸入控制，卻不能保證 LLM 產出的 opening keyframe、場景與構圖完全相同。這是配對的限制，也是 `lunchbox_snap` 漂移的警示，不應把 B 的整支片視為只由 timeline 造成。

決策：保留 `hypit_v1` 為 opt-in production capability，不改現行 default route。建議只在「單一主角 + 單一主導機制 + 明確 payoff」的 motion brief 啟用，並在 Discord 人工審查時優先檢查 subject identity/geography 與 opening-story fidelity；若要升級為 default，下一輪應先增加 deterministic/template-mode matched benchmark，並把 opening keyframe 的變更隔離成獨立變因。

正式證據位置：`E:\comfyui\_extra\benchmarks\hypit_semantic_ab\20260921_234621\benchmark_summary.json`，各 case 的 A/B `contact_sheet.jpg` 與影片路徑均由 summary 引用。
