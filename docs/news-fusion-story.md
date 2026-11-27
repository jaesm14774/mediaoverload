# 新聞融合故事（news_fusion_story）

一則新聞加一個角色，融合成一支約 30 秒的短片。新聞只是入口：取一個小場景，讓角色用自己的個性把它變成新聞沒寫的情節，故事要保留一兩句可引用的新聞原文作為錨點。

```powershell
python run_media_interface.py --character kirby --generation-type news_fusion_story --no-publish
```

## 設定

- 候選策略：`configs/routing.yaml` 的 `routing.strategy_candidates` 列出 `news_fusion_story`；工作流程階段為 `krea2_turbo`（圖片）、`minimax_h3_i2va`（影片）、`flux3_image_klein_img2img`（Klein 編輯），三個階段各只有一個候選。
- 隨機抽選：不指定 `--generation-type` 時，入口依角色 YAML 的 `generation.generation_type_weights` 抽策略。Kirby 的 `news_fusion_story` 權重為 0.5；權重表有列其他策略但沒列它的角色不會抽到它，但仍可用 `--generation-type news_fusion_story` 指定；權重表沒有任何有效項目時，會在所有候選策略間平均抽選。
- 段落數：`news_fusion_beat_count`（2 到 5，預設 3）。

## 流程

N 個關鍵幀段落（預設 3，`news_fusion_beat_count` 可設 2 到 5），N 段 I2VA 影片，N-1 個銜接點：

1. 故事作者一次寫完 N 個段落和 N-1 個銜接，沒有模板備援，格式或錨點不合就停止。
2. 第 1 張關鍵幀由 Krea2 畫，之後每張由 FLUX Klein 以前一張編輯，角色因此不變。開啟人工審核時，所有關鍵幀核准後才開始渲染影片。
3. 每段 I2VA 從自己的關鍵幀出發（約 5.2 秒）。
4. 每個銜接點：取前一段實際的最後一幀，由 Klein 畫出「走到一半」的連接圖，再用兩段 FL2VA（各約 4.5 秒）分別走完「結尾到連接圖」和「連接圖到下一張關鍵幀」。每一半只需要跨過一半的落差。連接圖要等前一段渲染完才存在，所以無法事先審核，只能在成品裡判斷。
5. 所有片段依故事順序合併，做技術 QA 並產生預覽 GIF。三段的影片約 33 秒，其中銜接約 18 秒。

新聞用隨機挑選，排除政治、選舉、社會、國際與財經類別，因為可愛角色不適合演犯罪、事故或選戰。

## 角色身體設定

角色描述（例如 Waddle Dee 沒有嘴巴、Magolor 沒有腳）每個影片提示詞都會帶上，Klein 的連接圖也一樣。但是 2026-11-21 同圖同種子的渲染顯示，只靠描述擋不住：

| 提示詞 | 結果 |
| --- | --- |
| 劇情寫「驚呼、大笑、唱歌」，不帶描述 | 長出大嘴 |
| 同上，帶「no mouth」描述 | 仍長出嘴（略小） |
| 同上，改成「臉上只有兩隻眼睛和臉頰，從來沒有嘴」的正向說法 | 仍長出大嘴 |
| 劇情不含發聲動作，帶描述 | 整段沒有嘴 |

所以真正的防線是文字本身不要求嘴巴或腳：

- 故事作者會被告知角色缺什麼，並拿到驗證器實際會擋的完整詞表（由 `FEATURE_ACTIONS` 產生，請求與檢查同源，變化形逐一列出，`x*` 代表以 x 開頭的字）。詞表適用關鍵影格、動作、連接片段與音效欄位，也適用群眾、路人與其他角色；`premise` 和音樂欄位不檢查。2026-11-23 重放同一則新聞時，模型常因音效欄位寫了 cheering 或 gasp 被拒，補上詞表後 Groq 與 Cloudflare 的 gpt-oss-120b 都能通過。
- 請求明確要求 `premise` 與每個連接片段的 `soundscape`。不支援 JSON schema 的供應商（Groq、Cloudflare、Experiential、AMD、NVIDIA）收到的是「欄位僅供參考、選填可省略」的指示（`structured_chat.py`），請求沒點名的必填欄位會被漏掉，整份故事因此被拒。
- 故事回應若出現缺少的部位，或需要它的動作（笑、唱、喊、驚呼、說話、吃、走、跑、踩踏等），整份故事被拒絕重寫。音樂欄位不檢查。缺少的部位從角色描述裡的「no ○○」取得，詞表在 `FEATURE_ACTIONS`。

限制：詞表只擋字面用字，沒列到的說法仍可能讓影片長出嘴巴或腳；只做過一個角色、一個種子的對照。
