# Pixelle-Video 研究

研究日期：2026-11-23。對象：[ATH-MaaS/Pixelle-Video](https://github.com/ATH-MaaS/Pixelle-Video)（Apache-2.0，約 3 萬星；非 fork，README 連結指向 AIDC-AI，是同一專案；最後更新 2026-06-14）。

方法：只做靜態閱讀（README、pipelines、services、prompts、templates），**Pixelle 沒有實際執行**；mediaoverload 側的缺口也只來自程式與文件閱讀，ComfyUI、GPU、LLM、發布端皆未實機驗證。

## 結論

H3 策略的對白與音樂由模型自己產生，固定 10 秒一段。Pixelle 的價值多半在「生成後再補音訊」，貼到 H3 輸出上會和模型音訊衝突。因此只有兩項值得進一步驗證，且都限於非 H3 輸出或素材導向策略。

## Pixelle 的流程

題目 → LLM 寫旁白 → 逐句寫圖片提示 → TTS 配音 → ComfyUI 或雲端 API 生成圖／影片 → HTML 模板（Playwright 渲染）疊標題與字幕 → ffmpeg 串接並混入配樂。

與 mediaoverload 不同的設計：

- 語音先行：每段影片長度由 TTS 音訊長度決定（`services/frame_processor.py`）。
- 排版走 HTML 模板：`templates/<寬>x<高>/` 內 22 個以上直式版面，檔名前綴 `static_`／`image_`／`video_` 決定是否需要生成媒體。
- 素材導向流程：使用者提供素材，VLM 描述，LLM 為每個場景指派一個素材並寫 1–3 句旁白（`pipelines/asset_based.py`、`prompts/asset_script_generation.py`）。

## 值得驗證

### 1. 素材導向流程（對應 greeting-video 設計）

結構：VLM 描述素材 → LLM 依意圖與時長把素材指派到場景並寫旁白 → 逐段生成，段間以前一段尾幀當首幀參考。

mediaoverload 目前沒有使用者素材流程。greeting-video 的照片轉 Q 版、多照片融合與角色清單設計可借這個「描述 → 指派 → 腳本」的順序。借結構，不借程式碼。

驗收方向：給定 N 張素材，每張都被分配到至少一個場景，腳本不引用不存在的素材，總時長落在目標範圍內。

### 2. 非 H3 輸出的語音先行、配樂與疊字

適用輸出：`motion_comic`、`story_card` 轉影片等沒有 H3 音軌的成品。

現況缺口：

- `motion_comic` adapter 沒有音訊處理。
- edge-tts 只有 `video_narrate` 計畫使用，混流是 `-c copy -t <影片長度>`，影片長度勝出，語音長度不影響畫面。
- 沒有字幕燒錄，也沒有配樂混音。

可借的三件事，皆可用現有依賴（ffmpeg、edge-tts）完成：

- 以旁白長度決定畫面停留時間。
- 以 ffmpeg `amix` 混入低音量配樂。
- 把旁白文字疊在畫面上。

驗收方向：成品音軌長度與畫面長度一致，配樂不蓋過旁白，文字不被裁切。

### 3. 有條件：HTML 模板

`story_card` 目前只有一種 Pillow 版面。只有在確定需要多種視覺主題時才考慮 HTML 模板。代價是新增 Chromium 依賴與中文字型設定。Pillow 版面現在可用，不主動做。

## 不借的項目

| 項目 | 原因 |
|---|---|
| 結構化輸出（schema 附在 prompt、Pydantic 解析） | `chat_json` 已有嚴格 json_schema、修復迴圈與重試，較強 |
| `LinearVideoPipeline` 模板方法基底類別 | 相對於 `WorkflowRunner` 是多餘抽象，違反 AGENTS.md |
| 內容審查失敗後的提示詞中性化重試 | 只對雲端 API 有意義，這裡用本地 ComfyUI |
| RunningHub 並行信號量 | 單張本地 GPU 本來就序列執行 |
| 尾幀接力 | 已有 `media.video.extract_last_frame`；Pixelle 的兩段式備援只是小幅強化 |
| `topic_narration` 提示詞 | 寫死引用名單與開頭規則，屬於寫死變化清單 |
| 對 H3 影片燒錄字幕 | H3 音訊沒有時間軸，需先做語音辨識，成本大於收益 |

## 後續

兩項值得驗證的內容尚未排入實作。落地時依 AGENTS.md：先寫 `User` 開頭的 Given / When / Then 驗收情境，再實作，並把事實移進 architecture 後刪除本文件對應段落。
