# MediaOverload 策略執行流程說明

## 目錄

- [整體架構](#整體架構)
- [批量生成功能](#批量生成功能)
- [重要定義](#重要定義)
- [執行流程（OrchestrationService）](#執行流程orchestrationservice)
- [各策略詳細流程](#各策略詳細流程)
  - [Text2ImageStrategy](#1-text2imagestrategy文字生成圖片)
  - [Text2Image2ImageStrategy](#2-text2image2imagestrategy文字圖片圖片)
  - [Text2Image2VideoStrategy](#3-text2image2videostrategy文字圖片影片)
  - [Text2VideoStrategy](#4-text2videostrategy文字生成影片)
  - [Text2LongVideoStrategy](#5-text2longvideostrategy文字生成長影片)
  - [StickerPackStrategy](#7-stickerpackstrategy貼圖包生成)
- [審核流程詳解](#審核流程詳解)
- [文章內容生成時機](#文章內容生成時機)
- [後處理媒體](#後處理媒體)
- [配置參數優先級](#配置參數優先級)
- [社群媒體格式支援](#社群媒體格式支援)
- [加權隨機選擇功能](#加權隨機選擇功能)
- [Two Character Interaction 功能](#two-character-interaction-功能)
- [Seed 管理功能](#seed-管理功能)
- [Style Weights 功能](#style-weights-功能)
- [常見問題](#常見問題)
- [錯誤修復記錄](#錯誤修復記錄)

---

## 整體架構

系統採用策略模式（Strategy Pattern），每個策略都繼承自 `ContentStrategy` 基類，實現以下核心方法：

- `generate_description()` — 生成描述 / 提示詞
- `generate_media()` — 生成媒體（圖片 / 影片）
- `analyze_media_text_match()` — 分析媒體與文本匹配度（LLM 自動分析）
- `needs_user_review()` — 判斷是否需要使用者審核
- `get_review_items()` — 獲取需要審核的項目
- `handle_review_result()` — 處理審核結果
- `should_generate_article_now()` — 判斷是否現在生成文章內容

---

## 批量生成功能

**支援從資料庫批量取 news 或使用自定義 prompts 批量生成。**

主要功能：

1. **從資料庫批量生成** — 自動從資料庫取多個 news，批量產生指定數量的內容
2. **自定義 prompts 批量生成** — 使用提供的 prompts 列表批量生成
3. **支援所有策略** — Text2Image、Text2Video、Text2LongVideo、StickerPack 等
4. **長影片直接模式** — 不保存中間圖片，只輸出最終完整影片（含 TTS）

### 使用範例（Jupyter Notebook）

```python
# 批量生成 30 張圖片
results = batch_generate_by_count(
    strategy_type='text2image',
    num_total=30,
    use_news=True,
    character="kirby",
    num_images=4
)

# 批量生成長影片（直接模式）
results = batch_generate_by_count(
    strategy_type='text2longvideo',
    num_total=2,
    use_news=True,
    character="kirby",
    skip_candidate_stage=True,  # 不保存中間圖片
    segment_count=3,
    use_tts=True
)
```

完整範例請參考 `examples/all_strategies_examples.ipynb`。

---

## 重要定義

### 審核（Review）

**審核 = 上傳媒體到 Discord，讓使用者人工選擇要使用的媒體。**

審核流程：

1. 系統將媒體文件（圖片 / 影片）上傳到 Discord 頻道
2. 使用者透過 Discord 介面查看媒體
3. 使用者選擇要使用的媒體（透過 Discord 反應或指令）
4. 系統接收使用者的選擇，繼續後續流程

**注意：**

- 審核是**人工審核**，不是自動選擇
- 審核發生在**發布到社群媒體之前**
- 使用者可在審核時編輯文章內容
- 最多可選擇 10 個媒體（符合 Discord API 限制）

### LLM 分析與人工審核的區別

| 方法 | 說明 | 用途 |
|------|------|------|
| `analyze_media_text_match()` | LLM 自動分析媒體與文本匹配度 | 初步篩選 |
| `review_content()` | 人工審核 | 最終確認要發布的媒體 |

---

## 執行流程（OrchestrationService）

### 主要流程步驟

1. **角色選擇** — 從群組中隨機選擇角色（特殊情況：Kirby 群組的長影片直接使用 kirby）
2. **生成提示詞** — 使用 PromptService 生成初始提示詞
3. **準備配置** — 創建 GenerationConfig
4. **生成內容（第一階段）** — 調用 `ContentGenerationService.generate_content()`
5. **檢查是否需要審核** — 調用 `strategy.needs_user_review()`
6. **Discord 人工審核流程**（如果需要）：
   - 獲取需要審核的媒體項目（最多 10 個）
   - 上傳媒體到 Discord 頻道，讓使用者查看
   - 使用者透過 Discord 選擇要使用的媒體
   - 使用者可以編輯文章內容（可選）
   - 系統接收使用者的選擇（`selected_indices`）
   - 處理審核結果（調用 `strategy.handle_review_result()`）
   - 重新分析結果（獲取最終的媒體）
   - 檢查是否需要再次審核（例如：影片生成後）
7. **後處理媒體** — 調用 `strategy.post_process_media()`（例如：圖片放大）
8. **處理媒體格式** — 轉換格式等
9. **發布到社群媒體** — Instagram、Twitter、Facebook 粉絲專頁等
10. **發送通知**
11. **清理資源**

---

## 各策略詳細流程

### 1. Text2ImageStrategy（文字生成圖片）

#### 執行流程

```
generate_description()
  → 根據 image_system_prompt_weights 隨機選擇 system prompt
   （支援：stable_diffusion_prompt, warm_scene_description_system_prompt,
    sticker_prompt_system_prompt, two_character_interaction_generate_system_prompt）
  → 使用 VisionManager 生成圖片描述

generate_media()
  → 根據描述生成多張圖片（預設每個描述 4 張）

analyze_media_text_match()
  → 使用 VisionManager 分析圖片與描述的匹配度

generate_article_content()
  → 基於 filter_results 生成文章內容（hashtags）
```

#### 特點

- **需要使用者審核** — `needs_user_review()` 在有 `filter_results` 時返回 True
- 使用者選擇最終要發布的圖片
- 支援圖片放大（`post_process_media()` 可選）
- 直接生成文章內容

#### 配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `images_per_description` | 每個描述生成幾張圖片 | 4 |
| `enable_upscale` | 是否啟用圖片放大 | — |
| `upscale_workflow_path` | 放大工作流路徑 | — |

---

### 2. Text2Image2ImageStrategy（文字→圖片→圖片）

#### 執行流程

```
generate_description()
  → 生成第一階段圖片描述

generate_media()
  → 【第一階段：Text to Image】
    生成多張候選圖片（預設每個描述 4 張）
  → 使用 VisionManager 篩選最佳圖片（similarity_threshold=0.0）
  → 【第二階段：Image to Image】
    對每張選中的圖片進行 I2I 處理（預設每張 1 個變體）

analyze_media_text_match()
  → 分析第二階段生成的圖片

generate_article_content()
  → 生成文章內容
```

#### 特點

- **需要使用者審核** — `needs_user_review()` 在有 `filter_results` 時返回 True
- 兩階段生成：先 T2I，再 I2I 精煉
- 自動篩選最佳圖片進入第二階段

#### 配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `first_stage.images_per_description` | 第一階段每個描述生成幾張 | 4 |
| `second_stage.images_per_input` | 第二階段每張輸入圖片生成幾個變體 | 1 |

---

### 3. Text2Image2VideoStrategy（文字→圖片→影片）

#### 執行流程

```
generate_description()
  → 生成圖片描述

generate_media()
  → 【第一階段：Text to Image】
    生成多張候選圖片（預設每個描述 4 張）

needs_user_review() → True（圖片已生成，影片未生成）

【Discord 審核：選擇圖片】

handle_review_result()
  → 【第二階段：Image to Video】
    對每張選中的圖片：
      1. 提取圖片內容
      2. 生成影片描述
      3. 生成音訊描述
      4. 使用 I2V workflow 生成影片（預設每張圖片 1 個影片）

needs_user_review() → True（影片已生成，未審核）

【Discord 審核：選擇影片】

handle_review_result()

analyze_media_text_match()
  → 分析影片文件（使用影片描述）

should_generate_article_now() → True（影片已生成）

generate_article_content()
  → 基於影片描述生成文章內容
```

#### 特點

- **需要兩次使用者審核**：
  1. 第一次：選擇要生成影片的圖片
  2. 第二次：選擇最終要發布的影片
- 延遲生成文章內容（直到影片生成後）
- 自動生成影片描述和音訊描述

#### 狀態管理

| 狀態 | 說明 |
|------|------|
| `_videos_generated` | 標記影片是否已生成 |
| `_videos_reviewed` | 標記影片是否已審核 |
| `video_descriptions` | 儲存每張圖片對應的影片描述 |
| `audio_descriptions` | 儲存每張圖片對應的音訊描述 |

#### 配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `first_stage.images_per_description` | 第一階段每個描述生成幾張圖片 | 4 |
| `video.videos_per_image` | 每張圖片生成幾個影片 | 1 |
| `video.i2v_workflow_path` | I2V 工作流路徑 | — |

---

### 4. Text2VideoStrategy（文字生成影片）

#### 執行流程

```
generate_description()
  → 兩階段描述生成：
    1. 生成角色描述
    2. 基於角色描述生成影片描述

generate_media()
  → 直接生成影片（預設每個描述 2 個影片）

analyze_media_text_match()
  → 簡化分析：返回所有影片（similarity=1.0）

generate_article_content()
  → 生成文章內容
```

#### 特點

- **需要使用者審核** — `needs_user_review()` 在有 `filter_results` 時返回 True
- 直接從文字生成影片（不經過圖片階段）
- 兩階段描述生成：先角色描述，再影片描述

#### 配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `videos_per_description` | 每個描述生成幾個影片 | 2 |

---

### 5. Text2LongVideoStrategy（文字生成長影片 - 尾幀驅動）

#### 最新優化（2025-12-17）

1. **I2I 幀轉換** — 使用 nova-anime-xl 的 I2I workflow 重新生成高質量第一幀
   - 問題：wan2.2 沒有角色知識，直接使用最後一幀會導致角色崩壞
   - 方案：在每個段落之間使用 I2I 重新生成，確保角色準確性
   - 配置：`frame_transition.enabled = true`

2. **劇情推進優化** — 基於新腳本描述進行 I2I 轉換
   - 問題：劇情停滯，看起來像重複的 6 秒影片
   - 方案：I2I 轉換時使用下一段的視覺描述，確保劇情推進

3. **TTS 音訊生成改進** — 增強錯誤處理和日誌記錄，確保音訊文件正確生成和合併

#### 執行流程

**模式 1：候選圖片模式（預設）**

```
generate_description()
  → 生成第一個段落的腳本（包含視覺描述和旁白）

generate_media()
  → 生成第一個段落的候選圖片（預設 3 張）

needs_user_review() → True

【Discord 審核：選擇第一幀圖片】

handle_review_result()

_generate_full_video_loop()
  → 循環生成多個段落（預設 5 個段落）：
    對每個段落：
      1. 如果不是第一段，基於上一段最後一幀生成腳本
      2. 上傳當前幀圖片
      3. 使用 I2V 生成影片
      4. 從影片提取最後一幀
      5. 如果不是最後一段，使用 I2I（nova-anime-xl）重新生成高質量第一幀
         - 避免角色崩壞（wan2.2 沒有角色知識）
         - 確保劇情推進（基於新腳本描述）
         - 保持視覺連續性

  → 後處理：
    1. 合併所有段落影片為一個完整影片
    2. 如果啟用 TTS：
       - 為每個段落生成 TTS 音訊
       - 合併所有音訊
       - 將音訊與影片合併

_generate_final_article_content()
  → 基於所有段落的腳本生成最終文章內容
```

**模式 2：直接生成模式（skip_candidate_stage=True）**

```
generate_description()
  → 生成第一個段落的腳本

generate_media()
  → 跳過候選圖片階段

_generate_full_video_direct()
  → 1. 直接生成第一幀圖片
    2. 使用第一幀開始完整影片循環
    3. 生成所有段落影片
    4. 合併段落為完整影片
    5. 如果啟用 TTS，添加旁白
    6. 不保存中間圖片，只輸出最終完整影片
    7. 清理臨時檔案

_generate_final_article_content()
  → 基於所有段落的腳本生成最終文章內容
```

#### 特點

- **兩種生成模式**：
  - 候選圖片模式 — 需要兩次使用者審核
  - 直接生成模式 — 無需審核，自動生成完整影片
- 多段落生成：每個段落基於上一段的最後一幀
- 支援 TTS 旁白
- 自動合併段落為完整影片
- **直接模式優勢**：不保存中間圖片，只輸出最終完整影片

#### 狀態管理

| 狀態 | 說明 |
|------|------|
| `script_segments` | 儲存所有段落的腳本 |
| `generated_media_paths` | 儲存生成的媒體路徑（最終為合併後的影片） |

#### 配置參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `longvideo_config.skip_candidate_stage` | 是否跳過候選圖片階段 | False |
| `longvideo_config.segment_count` | 段落數量 | 5 |
| `longvideo_config.segment_duration` | 每個段落時長 | 5 秒 |
| `longvideo_config.use_tts` | 是否使用 TTS | True |
| `longvideo_config.tts_voice` | TTS 語音 | `en-US-AriaNeural` |
| `first_stage.batch_size` | 第一幀候選圖片數量（僅候選模式） | 3 |
| `frame_transition.enabled` | 是否啟用 I2I 幀轉換 | true |
| `frame_transition.workflow_path` | I2I 工作流路徑 | `image_to_image.json` |
| `frame_transition.denoise` | I2I denoise 強度（推薦 0.5–0.7） | 0.6 |
| `frame_transition.style_continuity_prompt` | 風格連續性提示詞 | — |

---

### 7. StickerPackStrategy（貼圖包生成）

#### 執行流程

```
generate_description()
  → 使用 OpenRouter LLM 自動生成 8 種表情描述
   （happy, sad, angry, surprised, love, sleepy, confused, excited 等）

generate_media()
  → 批量生成 8 張靜態貼圖

needs_user_review() → True

【Discord 審核：選擇要使用的圖片】

handle_review_result()
  → 隨機決定（根據 gif_probability）：
    ├─ 生成動態 GIF（機率：gif_probability）
    │   → _generate_animated_stickers()
    │   → 對選中的圖片：
    │       1. 使用 I2V 生成短影片
    │       2. 使用 FFmpeg 轉換為優化 GIF
    │   → needs_user_review() → True
    │   → 【Discord 審核：選擇最終要發布的 GIF】
    │
    └─ 使用靜態貼圖（機率：1 - gif_probability）
        → 直接跳到生成文章內容

生成文章內容
```

#### 特點

- **LLM 自動生成表情** — 使用 OpenRouter 隨機模型生成多樣化表情
- **隨機動態 / 靜態** — 根據配置的機率隨機決定生成動態 GIF 或使用靜態貼圖
- **統一風格** — 所有貼圖保持一致的 LINE 貼圖風格
- **GIF 優化** — 使用 FFmpeg 二階段轉換，最佳化檔案大小
- **動作補幀** — 使用 minterpolate 進行幀插值，讓動畫更流暢自然
- **Instagram 相容** — 自動將 GIF 轉換為 MP4 格式以符合 Instagram 上傳要求

#### 配置參數

```yaml
sticker_pack:
  style: "LINE sticker style, chibi proportions, white outline, simple background"

  static_config:
    workflow_path: /app/configs/workflow/nova-anime-xl.json
    images_per_expression: 1

  animated_config:
    enabled: true
    gif_probability: 0.5        # 生成 GIF 的機率（0.0–1.0），預設 50%
    i2v_workflow_path: /app/configs/workflow/wan2.2_gguf_i2v.json
    # 短動畫參數
    total_frames: 33            # 總幀數（33 frames / 12 fps ≈ 2.75 秒）
    video_fps: 12
    # GIF 轉換參數
    gif_fps: 10
    gif_max_colors: 256
    gif_scale_width: 512
```

---

## 審核流程詳解

### 審核觸發條件

策略通過 `needs_user_review()` 方法決定是否需要上傳到 Discord 讓使用者選擇：

| 策略 | 觸發條件 |
|------|----------|
| **Text2ImageStrategy** | `len(filter_results) > 0` — 有 LLM 分析結果時 |
| **Text2Image2ImageStrategy** | `len(filter_results) > 0` — 有 LLM 分析結果時 |
| **Text2VideoStrategy** | `len(filter_results) > 0` — 有 LLM 分析結果時 |
| **Text2Image2VideoStrategy** | 第一次：`len(first_stage_images) > 0 and not _videos_generated`（選擇圖片）<br>第二次：`_videos_generated and not _videos_reviewed`（選擇影片） |
| **Text2LongVideoStrategy** | 第一次：`len(generated_media_paths) > 0`（選擇第一幀）<br>第二次：`len(generated_media_paths) > 0`（選擇最終影片） |

### 審核項目獲取

`get_review_items(max_items=10)` 返回需要上傳到 Discord 的媒體項目：

- **限制**：最多 10 個項目（符合 Discord API 限制）
- **格式**：`[{'media_path': '...', 'similarity': ...}, ...]`
- **用途**：媒體會被上傳到 Discord，讓使用者選擇

### 審核結果處理

`handle_review_result(selected_indices, output_dir)` 處理使用者在 Discord 中的選擇：

- `selected_indices` — 使用者在 Discord 中選擇的索引列表（相對於 `get_review_items()` 返回的列表）
- 根據策略不同，可能觸發：
  - 生成影片（Text2Image2VideoStrategy）
  - 生成後續段落（Text2LongVideoStrategy）
  - 圖片放大（Text2ImageStrategy，如果啟用）
  - 直接使用選擇的媒體進行後續處理

### 審核後的媒體路徑提取

在 `orchestration_service.py` 中，媒體路徑提取邏輯分為兩種情況：

1. **`selected_result_already_filtered = True`** — `selected_result` 已根據 `selected_indices` 過濾，直接提取所有媒體路徑
2. **`selected_result_already_filtered = False`** — `selected_result` 尚未過濾，使用 `selected_indices` 索引提取媒體路徑

無論審核流程如何，上述邏輯都能正確提取使用者選擇的媒體。

---

## 文章內容生成時機

### 立即生成（`should_generate_article_now() = True`）

| 策略 | 時機 |
|------|------|
| Text2ImageStrategy | 審核後立即生成 |
| Text2Image2ImageStrategy | 審核後立即生成 |
| Text2VideoStrategy | 審核後立即生成 |
| Text2LongVideoStrategy | 最終影片生成後立即生成 |

### 延遲生成（`should_generate_article_now() = False`）

| 策略 | 說明 |
|------|------|
| Text2Image2VideoStrategy | 圖片階段不生成文章內容；影片生成後才生成基於影片描述的文章內容 |

---

## 後處理媒體

### 圖片放大（Upscale）

**Text2ImageStrategy**

- 配置：`enable_upscale = True`
- 工作流：`upscale_workflow_path`
- 處理：對每張選中的圖片進行放大處理

**Text2Image2VideoStrategy**

- 配置：`first_stage.enable_upscale = True`
- 工作流：`first_stage.upscale_workflow_path`
- 處理流程：
  1. 使用者選擇圖片後，先對選中的圖片進行放大
  2. 使用放大後的圖片生成影片
  3. 使用者再選擇最終要發布的影片

**Text2LongVideoStrategy**

- 配置：`first_stage.enable_upscale = True`
- 工作流：`first_stage.upscale_workflow_path`
- 處理流程：
  1. 對第一幀進行放大後生成第一個影片段落
  2. 對每個影片段落的最後一幀進行放大
  3. 使用放大後的幀作為下一個段落的輸入

**不支援後處理的策略**：Text2Image2ImageStrategy、Text2VideoStrategy

---

## 配置參數優先級

配置參數的合併順序（優先級從高到低）：

1. **階段特定配置**（例如：`strategies.text2image2video.first_stage`）
2. **策略特定配置**（例如：`strategies.text2image2video`）
3. **通用配置**（`general`）
4. **Config 屬性**（`config.xxx`）
5. **預設值**

更細粒度的配置可以覆蓋更通用的配置。

---

## 社群媒體格式支援

### Instagram 格式轉換

**Instagram 不支援直接上傳 GIF 格式。** 系統會自動將 GIF 轉換為 MP4 格式後再上傳。

自動轉換流程：

1. 檢測到 GIF 檔案時，自動使用 FFmpeg 轉換為 MP4
2. 轉換後的 MP4 檔案在上傳完成後自動清理
3. 直接轉換 GIF 一次（不使用循環輸入，避免轉換卡住）

轉換參數：

| 參數 | 值 | 說明 |
|------|-----|------|
| FPS | 自動從 GIF 讀取 | 使用 ffprobe（30 秒超時），讀取失敗時預設 10 |
| Pixel Format | yuv420p | 確保相容性 |
| `-movflags` | faststart | 優化串流播放 |
| Timeout | 2 分鐘 | 超時保護，避免轉換過程卡住 |

**重要修復**：已移除 `-stream_loop` 參數。Instagram 會自動循環播放 MP4，不需要在轉換時循環輸入。使用 `-stream_loop` 是導致轉換卡住的根本原因。

### Facebook 粉絲專頁發布

**當 Instagram 有驗證問題時，可改用 Facebook 粉絲專頁發布。** 使用 Graph API，需具備 `pages_manage_posts`、`pages_read_engagement` 權限。

所需資訊：

- `FB_PAGE_ID` — 粉絲專頁 ID（專頁「關於」或 API 查詢取得）
- `FB_PAGE_ACCESS_TOKEN` — 粉絲專頁長期 Access Token

設定步驟：

1. 在 `configs/social_media/credentials/{character_name}/` 建立 `facebook.env`
2. 複製 `facebook.env.example` 並填入憑證
3. 在角色 YAML 的 `social_media.platforms` 新增：

```yaml
social_media:
  platforms:
    facebook:
      config_folder_path: /app/configs/social_media/credentials/kirby
      prefix: ""
      enabled: true
```

取得 Token：前往 [developers.facebook.com](https://developers.facebook.com/) 建立應用程式，取得長期 Page Access Token。詳見 `configs/social_media/credentials/facebook.env.example`。

重新產生長期 Token：當 Page Token 過期時，執行 `python utils/generate_fb_token.py`，依提示選擇角色並貼上短期 User Token（可從 [Graph API Explorer](https://developers.facebook.com/tools/explorer/) 取得，需具備 `pages_manage_posts`、`pages_read_engagement` 權限），腳本會交換為長期 Page Token 並可選擇寫入 `facebook.env`。

測試發布：`python utils/test_fb_post.py` 可測試純文字；加 `--image` 或 `--video` 可測試圖片 / 影片。

### Instagram Graph API 官方發布

**使用 [Instagram API with Instagram Login](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/content-publishing) 發布至 IG 商業帳號。** 無需 Facebook 粉絲專頁，與 instagrapi（`instagram` 平台）並存。

權限：`instagram_business_basic`、`instagram_business_content_publish`

所需資訊：

- `IG_GRAPH_ACCESS_TOKEN` — Instagram User access token（從 App Dashboard 或 Business Login 取得）
- 媒體 URL（二選一）：Cloudinary 或 `IG_GRAPH_MEDIA_BASE_URL`
- `IG_USER_ID` — 可選，未設定時從 `/me` 端點自動取得

**媒體 URL 說明**：graph.instagram.com 使用 `image_url` / `video_url`，API 會 cURL 媒體。有兩種方式：

1. **Cloudinary** — 在 `media_overload.env` 設定 `CLOUDINARY_CLOUD_NAME`、`CLOUDINARY_API_KEY`、`CLOUDINARY_API_SECRET`（或 `cloudinary_token`），系統會自動上傳取得 URL
2. **靜態基底** — 設定 `IG_GRAPH_MEDIA_BASE_URL`，媒體需已可透過 URL 存取（nginx、S3、CDN）

設定步驟：

1. 在 `configs/social_media/credentials/{character_name}/` 建立 `instagram_graph.env`
2. 複製 `instagram_graph.env.example` 並填入憑證
3. 設定 Cloudinary（`media_overload.env`）或 `IG_GRAPH_MEDIA_BASE_URL`（`instagram_graph.env`）
4. 在角色 YAML 的 `social_media.platforms` 新增：

```yaml
social_media:
  platforms:
    instagram_graph:
      config_folder_path: /app/configs/social_media/credentials/kirby
      prefix: ""
      enabled: true
```

**與 instagrapi 的差異**：`instagram` 使用 instagrapi（帳密登入），`instagram_graph` 使用官方 Instagram API（Token 認證，無需 FB 專頁）。兩者可同時啟用。

快速連線測試：`python utils/test_instagram_graph.py` 或 `python utils/test_instagram_graph.py kirby`

IG / FB 上傳除錯：`python utils/test_ig_fb_upload.py [角色名] [媒體1] [媒體2] ...`，需至少 2 個媒體檔案測試輪播，不帶媒體時僅測試連線。

### GIF 優化功能

系統在生成 LINE 貼圖 GIF 時會自動進行優化。

優化項目：

- **動作補幀** — 使用 `minterpolate` 進行幀插值，讓動畫更流暢自然
- **調色板優化** — 使用二階段轉換，最佳化檔案大小
- **解析度調整** — 自動縮放以符合貼圖標準

配置參數（在 `sticker_pack.animated_config` 中）：

```yaml
animated_config:
  gif_fps: 10             # GIF 播放幀率
  gif_max_colors: 256     # 最大顏色數
  gif_scale_width: 512    # 寬度（高度自動計算）
```

---

## 加權隨機選擇功能

系統支援兩個重要的加權隨機選擇功能：

1. **Image System Prompt Weights** — 描述生成系統提示詞的加權選擇
2. **Style Weights** — 視覺風格的加權選擇

兩個功能使得內容生成更加多樣化和可控。

---

## Two Character Interaction 功能

### 概述

Two Character Interaction 是一個特殊的圖片描述生成系統，用於生成兩個角色互動的場景。

### 支援策略

**已支援**（2025-12-17 修復）：

- Text2ImageStrategy — 文字生成圖片
- Text2Image2ImageStrategy — 文字→圖片→圖片
- Text2Image2VideoStrategy — 文字→圖片→影片
- Text2VideoStrategy — 文字生成影片（角色描述階段）

**不支援**：

- Text2LongVideoStrategy — 長影片（使用腳本生成系統）
- Text2LongVideoFirstFrameStrategy — 長影片首幀模式（使用腳本生成系統）
- Image2ImageStrategy — 圖片→圖片（從現有圖片提取內容）
- StickerPackStrategy — 貼圖包（使用專用表情系統）

### 使用方式

在角色配置檔案中，使用 `image_system_prompt_weights` 設定 two character interaction 的機率：

```yaml
generation:
  # 全域設定（所有策略）
  image_system_prompt_weights:
    stable_diffusion_prompt: 0.3
    warm_scene_description_system_prompt: 0.3
    sticker_prompt_system_prompt: 0.4
    two_character_interaction_generate_system_prompt: 0.3  # 30% 機率
```

或針對特定策略設定：

```yaml
additional_params:
  strategies:
    text2image2video:
      first_stage:
        image_system_prompt_weights:
          stable_diffusion_prompt: 0.3
          two_character_interaction_generate_system_prompt: 0.7  # 70% 機率
```

### Secondary Character 來源

當系統選擇使用 two character interaction 時，會自動選擇第二角色：

1. **優先**：從 `config.secondary_character` 獲取（如果有指定）
2. **次要**：從資料庫中隨機選擇與主角色不同的角色
3. **回退**：如果無法獲取，則使用預設的圖片生成方法

### 實現細節

- 基類 `ContentStrategy` 提供 `_get_system_prompt()` 方法，支援加權隨機選擇
- 所有策略繼承此方法，特殊需求的策略可以覆寫
- 系統根據權重隨機選擇 system prompt
- 當選中 `two_character_interaction_generate_system_prompt` 時：
  - 調用 `_generate_two_character_interaction_description()` 方法
  - 獲取 secondary character
  - 使用 `vision_manager.generate_two_character_interaction_prompt()` 生成場景描述
- 生成的描述包含兩個角色的互動細節（動作、表情、環境等）

### 配置優先級

配置查找順序（從高到低）：

1. 階段特定配置：`strategies.{strategy_name}.{stage}.image_system_prompt_weights`
2. 策略特定配置：`strategies.{strategy_name}.image_system_prompt_weights`
3. 全域配置：`generation.image_system_prompt_weights`
4. 單一值：`image_system_prompt`（不支援加權選擇）

---

## Seed 管理功能

### exclude_sampler_node_ids 參數（推薦）

**當 workflow 中包含多個 KSampler 節點時，可在 `configs/workflow_config.yaml` 中將 workflow 路徑映射到對應的 exclude 配置，避免 seed 被自動更新。**

優點：

- 配置與 workflow 綁定，更換 workflow 時不會遺忘配置
- 不修改 workflow JSON，保持 ComfyUI 兼容性
- 使用 node_id 精確指定，不依賴節點順序
- 配置集中管理，易於維護

使用場景：

- workflow 中有多個採樣器（例如 z-image + nova model）
- 希望某些採樣器的 seed 保持固定，而其他採樣器的 seed 隨機

### 推薦配置方式（workflow_config.yaml）

在 `configs/workflow_config.yaml` 中添加 workflow 路徑映射：

```yaml
workflows:
  # 完整路徑匹配
  "/app/configs/workflow/z_image_plus_nova_model.json":
    exclude_sampler_node_ids: ["80:44"]
    description: "z-image 的 KSampler (80:44) 保持 seed 固定，nova 的 KSampler (81:76) seed 隨機"

  # 相對路徑匹配（匹配所有以此路徑結尾的 workflow）
  "configs/workflow/z_image_plus_nova_model.json":
    exclude_sampler_node_ids: ["80:44"]

  # 文件名匹配（匹配所有包含此文件名的 workflow）
  "z_image_plus_nova_model.json":
    exclude_sampler_node_ids: ["80:44"]
```

配置參數說明：

| 參數 | 說明 |
|------|------|
| `exclude_sampler_node_ids` | 要排除的節點 ID 列表（推薦使用，精確指定） |
| `exclude_sampler_indices` | 要排除的節點索引列表（備選方案，基於節點順序） |
| `description` | 可選的描述文字，說明配置用途 |

匹配優先級：

1. 完整路徑匹配
2. 相對路徑匹配（路徑結尾匹配）
3. 文件名匹配

### 備選配置方式（YAML 配置文件中）

如果無法修改 `workflow_config.yaml`，也可以在策略配置中使用 `exclude_sampler_indices`：

```yaml
strategies:
  text2img:
    exclude_sampler_indices: [0]  # 排除第一個 KSampler，保持 seed 固定
```

**注意：**

- 優先級：`workflow_config.yaml` 中的配置 > YAML 配置文件中的配置
- 推薦使用 `exclude_sampler_node_ids`，因為不依賴節點順序，更穩定
- 只有未被排除的 KSampler 會自動更新 seed
- 系統會自動根據 workflow 路徑查找對應的配置

---

## Style Weights 功能

### 概述

**Style Weights 允許系統根據權重隨機選擇視覺風格，讓生成的內容更加多樣化。**

### 支援策略

所有策略均已支援（2025-12-17 修復）：

- Text2ImageStrategy、Text2Image2ImageStrategy、Text2Image2VideoStrategy
- Text2VideoStrategy、Text2LongVideoStrategy、Text2LongVideoFirstFrameStrategy
- Image2ImageStrategy、StickerPackStrategy

### 使用方式

在角色配置檔案中，使用 `style_weights` 設定不同風格的機率：

```yaml
generation:
  # 全域風格設定（所有策略）
  style_weights:
    "minimalism style with pure background": 0.2
    "watercolor painting style with pure background": 0.1
    "A highly saturated, dreamy-colored digital illustration style": 0.3
    "": 0.4  # 空字串表示不加風格提示詞
```

或針對特定策略設定：

```yaml
additional_params:
  strategies:
    text2image2video:
      first_stage:
        style_weights:
          "minimalism style with pure background": 0.8
          "": 0.2
```

### 實現細節

- 基類 `ContentStrategy` 提供 `_get_style()` 方法，支援加權隨機選擇
- 所有策略繼承此方法，特殊需求的策略可以覆寫（如 `StickerPackStrategy`）
- 系統根據權重隨機選擇風格
- 選中的風格會被添加到 prompt 中：`{prompt}\nstyle: {style}`
- 選中空字串 `""` 時，不添加風格提示詞
- 每次生成時都會重新隨機選擇

### 代碼架構

**基類實現**（`base_strategy.py`）：

```python
def _get_style(self, stage_config: Dict[str, Any], default: str = '') -> str:
    """獲取視覺風格，支援加權隨機選擇"""
    weights = stage_config.get('style_weights')
    if weights:
        choices = list(weights.keys())
        probs = list(weights.values())
        total = sum(probs)
        if total > 0:
            probs = [p/total for p in probs]
            return np.random.choice(choices, p=probs)
    return self._get_config_value(stage_config, 'style', default)
```

**特殊策略覆寫**（例如 `sticker_pack.py`）：

```python
def _get_style(self, stage_config):
    """覆寫基類方法以使用不同的默認值"""
    return super()._get_style(stage_config,
        default='LINE sticker style, chibi proportions, white outline...')
```

### 配置優先級

配置查找順序（從高到低）：

1. 階段特定配置：`strategies.{strategy_name}.{stage}.style_weights`
2. 策略特定配置：`strategies.{strategy_name}.style_weights`
3. 全域配置：`generation.style_weights`
4. 單一值：`style`（不支援加權選擇）

### 與 image_system_prompt_weights 的關係

**兩個功能是獨立的：**

- `image_system_prompt_weights` 控制**如何生成描述**（使用哪個系統提示詞）
- `style_weights` 控制**生成什麼風格的描述**（視覺風格）

同時使用時，系統會：

1. 先根據 `image_system_prompt_weights` 選擇系統提示詞
2. 再根據 `style_weights` 選擇風格
3. 將風格添加到 prompt 中
4. 使用選定的系統提示詞和 prompt 生成描述

---

## 常見問題

### Jupyter Notebook 中出現 `NameError: name '__file__' is not defined`

在 Jupyter Notebook 環境中，`__file__` 變數不存在。`all_strategies_examples.ipynb` 已修正：

- Cell 2 中定義了全局 `project_root` 變數
- `build_config_for_strategy` 函數使用 `global project_root` 來存取專案路徑
- 請按順序執行 Cell 2（環境初始化）後再執行其他 Cell

### 什麼是「審核」？

**審核 = 上傳媒體到 Discord，讓使用者人工選擇要使用的媒體。** 審核不是自動選擇或 LLM 分析，而是人工審核流程，確保最終發布的內容都經過使用者確認。

### 為什麼所有策略都需要審核？

每個策略在生成媒體後，都會上傳到 Discord 讓使用者選擇最終要發布的內容，確保發布到社群媒體的內容都經過使用者確認。

### 為什麼 Text2Image2VideoStrategy 需要兩次審核？

因為分為兩個階段：

1. 第一階段生成多張候選圖片，使用者選擇要生成影片的圖片
2. 第二階段生成影片後，使用者選擇最終要發布的影片

### 為什麼 Text2LongVideoStrategy 需要兩次審核？

同樣分為兩個階段：

1. 第一階段生成候選圖片，使用者選擇第一幀圖片
2. 第二階段生成完整影片後，使用者選擇最終要發布的影片

### 審核時選擇的索引如何使用？

`selected_indices` 是相對於 `get_review_items()` 返回列表的索引。在 `handle_review_result()` 中，策略會將索引映射到實際的媒體路徑。

### 為什麼有些策略延遲生成文章內容？

文章內容應該基於最終的媒體（例如：影片）生成，而不是中間產物（例如：圖片）。Text2Image2VideoStrategy 在影片生成後才生成文章內容，確保內容與最終媒體匹配。

### 如何確保審核後正確提取媒體路徑？

使用 `selected_result_already_filtered` 標記來區分兩種情況：

- 已過濾：直接從 `selected_result` 提取
- 未過濾：使用 `selected_indices` 索引 `selected_result`

### Instagram 登入時出現「找不到帳號」錯誤

如果出現 `We can't find an account with {username}` 錯誤，可能原因和解決方案：

**`IG_USERNAME` 欄位可接受的格式**：

- 使用者名稱（例如：`wobbuffet_mao_66666`）
- Email（例如：`your_email@example.com`）
- 電話號碼（例如：`+886912345678`）

**解決步驟**：

1. 如果使用使用者名稱登入失敗，嘗試使用 **email** 登入
2. 檢查 `configs/social_media/credentials/{character_name}/ig.env` 中的 `IG_USERNAME` 設定
3. 確認帳號是否需要驗證（檢查電子郵件或簡訊）
4. 確認帳號未被停用或刪除

配置範例：

```env
# 使用 email 登入（推薦）
IG_USERNAME=your_email@example.com
IG_PASSWORD=your_password

# 或使用電話號碼
IG_USERNAME=+886912345678
IG_PASSWORD=your_password
```

**注意**：如果能在 UI 介面登入但程式碼無法登入，通常是因為 UI 可能使用 email 登入，而配置檔案使用的是使用者名稱。Instagram API 對某些使用者名稱格式的支援可能有限制。

### 如何更新或新增 Instagram Session？

使用專用的 Session 產生器腳本：

```bash
python utils/generate_ig_session.py
```

腳本會引導使用者：

1. 選擇現有角色或新增角色
2. 確認或輸入 `ig.env` 中的憑證（使用者名稱、密碼、Proxy）
3. 執行登入並自動將 Session 儲存到對應的角色目錄下（`instagram_session.json`）

### 如何重新產生 Facebook 長期 Page Token？

執行 `python utils/generate_fb_token.py`，選擇角色後貼上短期 User Token（從 [Graph API Explorer](https://developers.facebook.com/tools/explorer/) 取得，需勾選 `pages_manage_posts`、`pages_read_engagement`），腳本會交換為長期 Page Token 並可選擇寫入 `facebook.env`。

---

## 錯誤修復記錄

### 雙角色互動獲取 Secondary Role 返回 None（2025-12-17）

**問題**：使用雙角色互動系統提示詞時，`_get_random_secondary_character` 方法返回 `None`，導致無法生成雙角色互動描述。

**根本原因**：系統在查詢同群組的其他角色時，使用了隨機選出的角色（如 `metaknight`）而不是群組代表角色（如 `kirby`）。

**執行流程說明**：

1. `orchestration_service` 從群組 `Kirby` 中隨機選擇了 `MetaKnight` 作為主角色
2. 在 `_get_random_secondary_character` 中，使用 `metaknight` 去查詢同群組的其他角色
3. 但資料庫查詢應該使用群組代表角色（`kirby`）來查詢，才能找到同群組的所有角色

**修復方案**：

1. 在 `orchestration_service` 中，保存原始的群組代表角色（`group_representative_character`）
2. 通過 `config_dict` 傳遞給 `GenerationConfig`
3. 在 `_get_random_secondary_character` 中，優先使用 `group_representative_character` 進行資料庫查詢
4. 過濾角色時，排除當前主角色（隨機選出的角色）

**影響檔案**：

- `lib/services/implementations/orchestration_service.py`
- `lib/media_auto/strategies/base_strategy.py`

**範例**：

- 群組代表角色：`kirby`
- 隨機選出的主角色：`metaknight`
- 查詢使用：`kirby`（群組代表角色）
- 過濾排除：`metaknight`（當前主角色）
- 可能的 Secondary Role：`waddle dee`、`king dedede` 等

---

### TTS 音訊生成 asyncio 錯誤（2025-12-17）

**問題**：Text2LongVideo 策略在生成 TTS 音訊時出現 `asyncio.run() cannot be called from a running event loop` 錯誤。

**原因**：`TTSService.generate_speech_sync()` 方法在檢測到運行中的 event loop 時，會在新線程中執行 async 任務。但在異常處理的 `except RuntimeError` 區塊中，仍然使用 `asyncio.run()`，導致當沒有運行中的 event loop 時也會失敗。

**修復**：重構 `generate_speech_sync()` 方法，將在新線程中執行的邏輯提取為 `run_in_thread()` 函數，並在兩種情況下都使用：

- 有運行中的 event loop：在線程池中執行
- 沒有運行中的 event loop：直接執行

**影響檔案**：`lib/services/implementations/tts_service.py`
