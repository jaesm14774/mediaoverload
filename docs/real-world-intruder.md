# 誤闖真實世界（real_world_intruder）

一則新聞給一個真實場景，一個畫風不同的角色（例如 Kirby）誤闖進去，做一件可愛又合乎物理的小事。反差就是笑點：扁平的 2D 角色在一張普通的真實照片裡認真地忙。左上角會貼一張膠帶固定的小朋友馬克筆塗鴉，畫的是同一個角色，約前 4 秒可見，之後淡出。

這個策略依 AIMediaPrompt 的 `Sketch-to-Reality-Absurd-Faithfulness` 提示詞與七支範例影片設計：範例片長 5 到 6 秒，但本專案影片下限是 10 秒（`MIN_VIDEO_DURATION_SECONDS`），所以這裡是一支約 10 秒的單一鏡頭。

```powershell
python run_media_interface.py --character kirby --generation-type real_world_intruder --no-publish
```

## 設定

- 候選策略：`configs/routing.yaml` 的 `routing.strategy_candidates` 列出 `real_world_intruder`；工作流程階段為 `krea2_turbo`（圖片）與 `minimax_h3_i2va`（影片），各只有一個候選。
- 隨機抽選：Kirby 的 `generation.generation_type_weights.real_world_intruder` 為 1。沒列在權重表的角色抽不到它，但仍可用 `--generation-type real_world_intruder` 指定。
- 新聞：與 `news_fusion_story` 相同，隨機挑一則，排除政治、選舉、社會、國際與財經類別。場景變化來自新聞，不是寫死的清單。
- 人工審核：`require_human_review` 開啟時，關鍵幀核准後才渲染影片。

## 流程

1. 故事作者寫 `premise`、`real_scene`（新聞裡一個可拍照的普通場所或物件）、`clip_scene`（10 秒內兩三個連續的可愛動作）、`soundscape`，並引用一兩句新聞原文作為錨點。格式、錨點不合，或用到角色沒有的部位（例如沒嘴的角色寫了笑或吃）就停止，沒有模板備援。
2. 關鍵幀提示詞由程式組合，不由 LLM 寫：一張手機拍的真實照片（方向由畫布決定，直式、橫式或方形都可能），場景裡有一隻約畫面高度五分之一的扁平 2D 角色，保留自己的原畫風，帶接觸陰影，圖中只有角色是畫出來的。Krea2 以影片畫布的兩倍尺寸繪製。
3. 同一個角色的塗鴉另由 Krea2 畫：皺紙上的幼稚園程度黑色馬克筆畫，提示詞帶角色自己的外觀描述，所以看得出是同一個角色。
4. H3 I2VA 從關鍵幀渲染一支至少 10 秒的影片。提示詞固定寫死鏡頭（不移動、不變焦）、除角色與它碰到的東西外畫面不動、角色全程維持約五分之一畫面高度，並帶上角色的身體描述。
5. 塗鴉貼進影片左上角（`media.sketch_inset`）：紙寬約為影片短邊的 20%，貼四條膠帶，微傾斜並帶陰影，前 4 秒完整可見，之後 0.5 秒淡出。畫布、幀率、時長與音訊都不變，音訊直接複製。
6. 接著走原生 H3 的收尾節點：畫布、剪輯節奏、NVIDIA VFX、技術 QA、預覽 GIF 與打包。QA 與打包讀的是貼好塗鴉的影片。

## 為什麼不用 `native_story`

`native_story` 策略在沒有自己的設定區塊時，會悄悄繼承 `native_h3_story` 的 15 秒配方，且它的關鍵幀提示詞是固定模板，畫不出「真實照片加 2D 角色」。所以本策略仿照 `news_fusion_story` 自成一圖。

## 實測紀錄

- 2026-11-28 手動 PoC（Kirby、草莓、直式 448x768、10 秒）：鏡頭鎖定，角色全程維持小隻。第一版提示詞寫「超大草莓」時，H3 會把鏡頭拉近，角色和草莓都塞滿畫面，所以固定提示詞明寫不移動、不變焦、角色維持約五分之一畫面高度、物件維持日常大小。
- 2026-11-28 完整流程一次（`--no-publish --no-review`，抽到 Bandana Waddle Dee、辦公桌上的表單與筆、橫式 640x360 加 VFX 成 1920x1080）：故事通過錨點與身體部位驗證，技術 QA 通過，立體聲音訊保留，約 10.1 秒，塗鴉淡出時間在剪輯節奏與 VFX 後仍正確。只驗過這一次，沒有驗過其他角色、種子或畫布；塗鴉是黑色線稿，但「只描線不上色」的措辭只驗過一次。
