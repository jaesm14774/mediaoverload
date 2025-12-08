# MediaOverload 策略執行流程說明

## 整體架構

系統採用策略模式（Strategy Pattern），每個策略都繼承自 `ContentStrategy` 基類，實現以下核心方法：
- `generate_description()` - 生成描述/提示詞
- `generate_media()` - 生成媒體（圖片/影片）
- `analyze_media_text_match()` - 分析媒體與文本匹配度（LLM 自動分析）
- `needs_user_review()` - 是否需要使用者審核
- `get_review_items()` - 獲取需要審核的項目
- `handle_review_result()` - 處理審核結果
- `should_generate_article_now()` - 是否現在生成文章內容

## 重要定義

### 審核（Review）的定義

**審核 = 上傳媒體到 Discord，讓使用者人工選擇要使用的媒體**

審核流程：
1. 系統將媒體文件（圖片/影片）上傳到 Discord 頻道
2. 使用者透過 Discord 介面查看媒體
3. 使用者選擇要使用的媒體（透過 Discord 反應或指令）
4. 系統接收使用者的選擇，繼續後續流程

**注意**：
- 審核是**人工審核**，不是自動選擇
- 審核發生在**發布到社群媒體之前**
- 審核可以讓使用者編輯文章內容
- 審核可以選擇多個媒體（最多 10 個，符合 Discord API 限制）

**與 LLM 分析的區別**：
- `analyze_media_text_match()` - LLM 自動分析媒體與文本匹配度，用於**初步篩選**
- `review_content()` - 人工審核，用於**最終確認**要發布的媒體

## 執行流程（OrchestrationService）

### 主要流程步驟

1. **角色選擇** - 從群組中隨機選擇角色（特殊情況：Kirby 群組的長影片直接使用 kirby）
2. **生成提示詞** - 使用 PromptService 生成初始提示詞
3. **準備配置** - 創建 GenerationConfig
4. **生成內容（第一階段）** - 調用 ContentGenerationService.generate_content()
5. **檢查是否需要審核** - 調用 strategy.needs_user_review()
6. **Discord 人工審核流程**（如果需要）
   - 獲取需要審核的媒體項目（最多 10 個，符合 Discord API 限制）
   - **上傳媒體到 Discord 頻道**，讓使用者查看
   - **使用者透過 Discord 選擇**要使用的媒體
   - **使用者可以編輯文章內容**（可選）
   - 系統接收使用者的選擇（`selected_indices`）
   - 處理審核結果（調用 strategy.handle_review_result()）
   - 重新分析結果（獲取最終的媒體）
   - 檢查是否需要再次審核（例如：影片生成後）
7. **後處理媒體** - 調用 strategy.post_process_media()（例如：圖片放大）
8. **處理媒體格式** - 轉換格式等
9. **發布到社群媒體** - Instagram、Twitter 等
10. **發送通知**
11. **清理資源**

---

## 各策略詳細流程

### 1. Text2ImageStrategy（文字生成圖片）

#### 執行流程

```
generate_description()
  ↓
  使用 VisionManager 生成圖片描述（支援雙角色互動）
  ↓
generate_media()
  ↓
  根據描述生成多張圖片（預設每個描述 4 張）
  ↓
analyze_media_text_match()
  ↓
  使用 VisionManager 分析圖片與描述的匹配度
  ↓
generate_article_content()
  ↓
  基於 filter_results 生成文章內容（hashtags）
```

#### 特點
- **需要使用者審核**（`needs_user_review()` 返回 True，當有 filter_results 時）
- 使用者選擇最終要發布的圖片
- 支援圖片放大（`post_process_media()` 可選）
- 直接生成文章內容

#### 配置參數
- `images_per_description`: 每個描述生成幾張圖片（預設 4）
- `enable_upscale`: 是否啟用圖片放大
- `upscale_workflow_path`: 放大工作流路徑

---

### 2. Text2Image2ImageStrategy（文字→圖片→圖片）

#### 執行流程

```
generate_description()
  ↓
  生成第一階段圖片描述
  ↓
generate_media()
  ↓
【第一階段：Text to Image】
  生成多張候選圖片（預設每個描述 4 張）
  ↓
  使用 VisionManager 篩選最佳圖片（similarity_threshold=0.0）
  ↓
【第二階段：Image to Image】
  對每張選中的圖片進行 I2I 處理（預設每張 1 個變體）
  ↓
analyze_media_text_match()
  ↓
  分析第二階段生成的圖片
  ↓
generate_article_content()
  ↓
  生成文章內容
```

#### 特點
- **需要使用者審核**（`needs_user_review()` 返回 True，當有 filter_results 時）
- 使用者選擇最終要發布的圖片
- 兩階段生成：先 T2I，再 I2I 精煉
- 自動篩選最佳圖片進入第二階段

#### 配置參數
- `first_stage.images_per_description`: 第一階段每個描述生成幾張（預設 4）
- `second_stage.images_per_input`: 第二階段每張輸入圖片生成幾個變體（預設 1）

---

### 3. Text2Image2VideoStrategy（文字→圖片→影片）

#### 執行流程

```
generate_description()
  ↓
  生成圖片描述
  ↓
generate_media()
  ↓
【第一階段：Text to Image】
  生成多張候選圖片（預設每個描述 4 張）
  ↓
needs_user_review() → True（圖片已生成，影片未生成）
  ↓
【Discord 審核：選擇圖片】
  ↓
handle_review_result()
  ↓
【第二階段：Image to Video】
  對每張選中的圖片：
    1. 提取圖片內容
    2. 生成影片描述
    3. 生成音訊描述
    4. 使用 I2V workflow 生成影片（預設每張圖片 1 個影片）
  ↓
needs_user_review() → True（影片已生成，未審核）
  ↓
【Discord 審核：選擇影片】
  ↓
handle_review_result()
  ↓
analyze_media_text_match()
  ↓
  分析影片文件（使用影片描述）
  ↓
should_generate_article_now() → True（影片已生成）
  ↓
generate_article_content()
  ↓
  基於影片描述生成文章內容
```

#### 特點
- **需要兩次使用者審核**
  1. 第一次：選擇要生成影片的圖片
  2. 第二次：選擇最終要發布的影片
- 延遲生成文章內容（直到影片生成後）
- 自動生成影片描述和音訊描述

#### 狀態管理
- `_videos_generated`: 標記影片是否已生成
- `_videos_reviewed`: 標記影片是否已審核
- `video_descriptions`: 儲存每張圖片對應的影片描述
- `audio_descriptions`: 儲存每張圖片對應的音訊描述

#### 配置參數
- `first_stage.images_per_description`: 第一階段每個描述生成幾張圖片（預設 4）
- `video.videos_per_image`: 每張圖片生成幾個影片（預設 1）
- `video.i2v_workflow_path`: I2V 工作流路徑

---

### 4. Text2VideoStrategy（文字生成影片）

#### 執行流程

```
generate_description()
  ↓
  兩階段描述生成：
    1. 生成角色描述
    2. 基於角色描述生成影片描述
  ↓
generate_media()
  ↓
  直接生成影片（預設每個描述 2 個影片）
  ↓
analyze_media_text_match()
  ↓
  簡化分析：返回所有影片（similarity=1.0）
  ↓
generate_article_content()
  ↓
  生成文章內容
```

#### 特點
- **需要使用者審核**（`needs_user_review()` 返回 True，當有 filter_results 時）
- 使用者選擇最終要發布的影片
- 直接從文字生成影片（不經過圖片階段）
- 兩階段描述生成：先角色描述，再影片描述

#### 配置參數
- `videos_per_description`: 每個描述生成幾個影片（預設 2）

---

### 5. Text2LongVideoStrategy（文字生成長影片 - 尾幀驅動）

#### 執行流程

```
generate_description()
  ↓
  生成第一個段落的腳本（包含視覺描述和旁白）
  ↓
generate_media()
  ↓
  生成第一個段落的候選圖片（預設 3 張）
  ↓
needs_user_review() → True
  ↓
【Discord 審核：選擇第一幀圖片】
  ↓
handle_review_result()
  ↓
_generate_full_video_loop()
  ↓
【循環生成多個段落】
  對每個段落（預設 5 個段落）：
    1. 如果不是第一段，基於上一段最後一幀生成腳本
    2. 上傳當前幀圖片
    3. 使用 I2V 生成該段落的影片
    4. 從影片提取最後一幀作為下一段的輸入
  ↓
【後處理】
  1. 合併所有段落影片為一個完整影片
  2. 如果啟用 TTS：
     - 為每個段落生成 TTS 音訊
     - 合併所有音訊
     - 將音訊與影片合併
  ↓
_generate_final_article_content()
  ↓
  基於所有段落的腳本生成最終文章內容
```

#### 特點
- **需要兩次使用者審核**
  1. 第一次：選擇第一幀候選圖片
  2. 第二次：選擇最終要發布的影片
- 多段落生成：每個段落基於上一段的最後一幀
- 支援 TTS 旁白
- 自動合併段落為完整影片

#### 狀態管理
- `script_segments`: 儲存所有段落的腳本
- `generated_media_paths`: 儲存生成的媒體路徑（最終為合併後的影片）

#### 配置參數
- `longvideo_config.segment_count`: 段落數量（預設 5）
- `longvideo_config.segment_duration`: 每個段落時長（預設 5 秒）
- `longvideo_config.use_tts`: 是否使用 TTS（預設 True）
- `longvideo_config.tts_voice`: TTS 語音（預設 'en-US-AriaNeural'）
- `first_stage.batch_size`: 第一幀候選圖片數量（預設 3）

---

### 6. Text2LongVideoFirstFrameStrategy（文字生成長影片 - 首幀驅動）

#### 執行流程

```
generate_description()
  ↓
  生成第一個段落的腳本
  ↓
generate_media()
  ↓
  生成第一個段落的候選圖片
  ↓
needs_user_review() → True
  ↓
【Discord 審核：選擇第一幀圖片】
  ↓
handle_review_result()
  ↓
_generate_full_video_with_firstframe_transitions()
  ↓
【循環生成多個段落】
  對每個段落：
    1. 使用當前首幀生成影片
    2. 從影片提取最後一幀
    3. 生成下一段的腳本
    4. ✨關鍵：使用 I2I 將尾幀 + 新分鏡描述 → 新首幀
    5. 使用新首幀作為下一段的輸入
  ↓
【後處理】
  1. 合併所有段落影片
  2. 如果啟用 TTS，添加旁白
  ↓
生成文章內容
```

#### 與 Text2LongVideoStrategy 的關鍵差異

| 項目 | 尾幀驅動 | 首幀驅動 |
|------|----------|----------|
| 段落銜接 | 尾幀 → 下一段 | 尾幀 → I2I → 新首幀 → 下一段 |
| I2I 用途 | 無 | 場景+風格延續，符合新分鏡 |
| Denoise | - | 0.5~0.6（保持風格，允許變化）|
| 適用場景 | 動作連貫的短分鏡 | 場景變化較大的長分鏡 |

#### 特點
- **首幀驅動**：I2V 模型對首幀控制力最強，生成更自然
- **I2I 過場**：透過低 denoise (0.5~0.6) 保持風格，同時適應新場景
- **場景適應**：允許更大的場景變化，適合長敘事影片

#### 配置參數
```yaml
text2longvideo_firstframe:
  longvideo_config:
    segment_count: 5
    segment_duration: 5
    use_tts: true
    tts_voice: "en-US-AriaNeural"
  
  first_stage:
    workflow_path: /app/configs/workflow/nova-anime-xl.json
    batch_size: 3
    style: "minimalism style"
    enable_upscale: true
  
  # 關鍵：首幀轉換配置
  frame_transition:
    enabled: true
    workflow_path: /app/configs/workflow/image_to_image.json
    denoise: 0.55  # 保持風格，允許場景變化
    style_continuity_prompt: "maintain visual style and color palette"
  
  video_generation:
    workflow_path: /app/configs/workflow/wan2.2_gguf_i2v.json
```

---

### 7. StickerPackStrategy（貼圖包生成）

#### 執行流程

```
generate_description()
  ↓
  使用 OpenRouter LLM 自動生成 8 種表情描述
  （happy, sad, angry, surprised, love, sleepy, confused, excited 等）
  ↓
generate_media()
  ↓
  批量生成 8 張靜態貼圖
  ↓
needs_user_review() → True
  ↓
【Discord 審核：選擇要轉為動態 GIF 的圖片】
  ↓
handle_review_result()
  ↓
_generate_animated_stickers()
  ↓
  對選中的圖片：
    1. 使用 I2V 生成短影片
    2. 使用 FFmpeg 轉換為優化 GIF
  ↓
needs_user_review() → True
  ↓
【Discord 審核：選擇最終要發布的 GIF】
  ↓
生成文章內容
```

#### 特點
- **LLM 自動生成表情**：使用 OpenRouter 隨機模型生成多樣化表情
- **支援動態 GIF**：將選中的靜態貼圖轉為動態 GIF
- **統一風格**：所有貼圖保持一致的 LINE 貼圖風格
- **GIF 優化**：使用 FFmpeg 二階段轉換，最佳化檔案大小
- **動作補幀**：使用 minterpolate 進行幀插值，讓動畫更流暢自然
- **Instagram 相容**：自動將 GIF 轉換為 MP4 格式以符合 Instagram 上傳要求

#### 配置參數
```yaml
sticker_pack:
  style: "LINE sticker style, chibi proportions, white outline, simple background"
  
  static_config:
    workflow_path: /app/configs/workflow/nova-anime-xl.json
    images_per_expression: 1
  
  animated_config:
    enabled: true
    i2v_workflow_path: /app/configs/workflow/wan2.2_gguf_i2v.json
    # 短動畫參數（控制 GIF 長度）
    total_frames: 33        # 總幀數（33 frames / 12 fps ≈ 2.75 秒）
    video_fps: 12           # 影片 fps
    # GIF 轉換參數
    gif_fps: 10             # GIF fps
    gif_max_colors: 256
    gif_scale_width: 512
```

---

## 審核流程詳解

### 審核觸發條件

策略通過 `needs_user_review()` 方法決定是否需要**上傳到 Discord 讓使用者選擇**：

1. **Text2ImageStrategy**
   - `len(filter_results) > 0`（有 LLM 分析結果時，需要人工審核確認）

2. **Text2Image2ImageStrategy**
   - `len(filter_results) > 0`（有 LLM 分析結果時，需要人工審核確認）

3. **Text2VideoStrategy**
   - `len(filter_results) > 0`（有 LLM 分析結果時，需要人工審核確認）

4. **Text2Image2VideoStrategy**
   - 第一次：`len(first_stage_images) > 0 and not _videos_generated`（圖片已生成，需要選擇要生成影片的圖片）
   - 第二次：`_videos_generated and not _videos_reviewed`（影片已生成，需要選擇最終要發布的影片）

5. **Text2LongVideoStrategy**
   - 第一次：`len(generated_media_paths) > 0`（候選圖片已生成，需要選擇第一幀）
   - 第二次：`len(generated_media_paths) > 0`（最終影片已生成，需要選擇最終要發布的影片）

### 審核項目獲取

`get_review_items(max_items=10)` 方法返回需要**上傳到 Discord** 的媒體項目：
- **限制**：最多 10 個項目（符合 Discord API 限制）
- **格式**：`[{'media_path': '...', 'similarity': ...}, ...]`
- **用途**：這些媒體會被上傳到 Discord，讓使用者選擇

### 審核結果處理

`handle_review_result(selected_indices, output_dir)` 方法處理**使用者在 Discord 中的選擇**：
- `selected_indices`: 使用者在 Discord 中選擇的索引列表（相對於 `get_review_items()` 返回的列表）
- 根據策略不同，可能觸發：
  - 生成影片（Text2Image2VideoStrategy）
  - 生成後續段落（Text2LongVideoStrategy）
  - 圖片放大（Text2ImageStrategy，如果啟用）
  - 或者直接使用選擇的媒體進行後續處理

### 審核後的媒體路徑提取

**重要**：在 `orchestration_service.py` 中，媒體路徑提取邏輯：

1. **如果 `selected_result_already_filtered = True`**
   - `selected_result` 已經根據 `selected_indices` 過濾過
   - 直接從 `selected_result` 提取所有媒體路徑

2. **如果 `selected_result_already_filtered = False`**
   - `selected_result` 尚未過濾
   - 使用 `selected_indices` 索引 `selected_result` 來提取媒體路徑

這確保了無論審核流程如何，都能正確提取使用者選擇的媒體。

---

## 文章內容生成時機

### 立即生成（`should_generate_article_now() = True`）

- **Text2ImageStrategy** - 審核後立即生成
- **Text2Image2ImageStrategy** - 審核後立即生成
- **Text2VideoStrategy** - 審核後立即生成
- **Text2LongVideoStrategy** - 最終影片生成後立即生成

### 延遲生成（`should_generate_article_now() = False`）

- **Text2Image2VideoStrategy**
  - 在圖片階段：不生成文章內容
  - 在影片生成後：生成基於影片描述的文章內容

---

## 後處理媒體

### 圖片放大（Upscale）

#### Text2ImageStrategy
支援可選的圖片放大：
- 配置：`enable_upscale = True`
- 工作流：`upscale_workflow_path`
- 處理：對每張選中的圖片進行放大處理

#### Text2Image2VideoStrategy
支援可選的圖片放大：
- 配置：`first_stage.enable_upscale = True`
- 工作流：`first_stage.upscale_workflow_path`
- 處理流程：
  1. 使用者選擇圖片後，先對選中的圖片進行放大
  2. 使用放大後的圖片生成影片
  3. 使用者再選擇最終要發布的影片

#### Text2LongVideoStrategy
支援可選的圖片放大：
- 配置：`first_stage.enable_upscale = True`
- 工作流：`first_stage.upscale_workflow_path`
- 處理流程：
  1. 對第一幀進行放大後生成第一個影片段落
  2. 對每個影片段落的最後一幀進行放大
  3. 使用放大後的幀作為下一個段落的輸入

### 其他策略

- **Text2Image2ImageStrategy**: 無後處理
- **Text2VideoStrategy**: 無後處理

---

## 配置參數優先級

配置參數的合併順序（優先級從高到低）：

1. **階段特定配置**（例如：`strategies.text2image2video.first_stage`）
2. **策略特定配置**（例如：`strategies.text2image2video`）
3. **通用配置**（`general`）
4. **Config 屬性**（`config.xxx`）
5. **預設值**

這確保了更細粒度的配置可以覆蓋更通用的配置。

---

## 社群媒體格式支援

### Instagram 格式轉換

Instagram 不支援直接上傳 GIF 格式，系統會自動將 GIF 轉換為 MP4 格式後再上傳。

**自動轉換流程**：
1. 檢測到 GIF 檔案時，自動使用 FFmpeg 轉換為 MP4
2. 轉換後的 MP4 檔案會在上傳完成後自動清理
3. 直接轉換 GIF 一次（不使用循環輸入，避免轉換卡住）

**轉換參數**：
- FPS: 自動從 GIF 檔案讀取（使用 ffprobe，30 秒超時），讀取失敗時使用預設值 10
- Pixel Format: yuv420p（確保相容性）
- 不使用 `-stream_loop`：避免無限循環導致轉換卡住，直接轉換 GIF 一次即可
- `-movflags faststart`：優化串流播放
- Timeout: 2 分鐘超時保護，避免轉換過程卡住

**重要修復**：
- 移除了 `-stream_loop` 參數，這是導致轉換卡住的根本原因
- Instagram 會自動循環播放 MP4，不需要在轉換時循環輸入

### GIF 優化功能

系統在生成 LINE 貼圖 GIF 時會自動進行優化：

**優化項目**：
- **動作補幀**：使用 `minterpolate` 進行幀插值，讓動畫更流暢自然
- **調色板優化**：使用二階段轉換，最佳化檔案大小
- **解析度調整**：自動縮放以符合貼圖標準

**配置參數**（在 `sticker_pack.animated_config` 中）：
```yaml
animated_config:
  gif_fps: 10             # GIF 播放幀率
  gif_max_colors: 256     # 最大顏色數
  gif_scale_width: 512    # 寬度（高度自動計算）
```

---

## 常見問題

### Q: 什麼是「審核」？

A: **審核 = 上傳媒體到 Discord，讓使用者人工選擇要使用的媒體**。這不是自動選擇或 LLM 分析，而是人工審核流程，確保最終發布的內容都經過使用者確認。

### Q: 為什麼所有策略都需要審核？

A: 為了確保最終發布到社群媒體的內容都經過使用者確認。每個策略在生成媒體後，都會**上傳到 Discord 讓使用者選擇**最終要發布的內容。

### Q: 為什麼 Text2Image2VideoStrategy 需要兩次審核？

A: 因為它分為兩個階段：
1. 第一階段生成多張候選圖片，使用者選擇要生成影片的圖片
2. 第二階段生成影片後，使用者選擇最終要發布的影片

### Q: 為什麼 Text2LongVideoStrategy 需要兩次審核？

A: 因為它分為兩個階段：
1. 第一階段生成候選圖片，使用者選擇第一幀圖片
2. 第二階段生成完整影片後，使用者選擇最終要發布的影片

### Q: 審核時選擇的索引是如何使用的？

A: `selected_indices` 是相對於 `get_review_items()` 返回列表的索引。在 `handle_review_result()` 中，策略會將這些索引映射到實際的媒體路徑。

### Q: 為什麼有些策略延遲生成文章內容？

A: 因為文章內容應該基於最終的媒體（例如：影片）生成，而不是中間產物（例如：圖片）。Text2Image2VideoStrategy 在影片生成後才生成文章內容，確保內容與最終媒體匹配。

### Q: 如何確保審核後正確提取媒體路徑？

A: 使用 `selected_result_already_filtered` 標記來區分兩種情況：
- 如果已過濾：直接從 `selected_result` 提取
- 如果未過濾：使用 `selected_indices` 索引 `selected_result`

這確保了無論審核流程如何，都能正確提取使用者選擇的媒體。
