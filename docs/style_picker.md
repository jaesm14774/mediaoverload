# 挑選 Krea 2 隨機生成風格

啟動本機頁面：

```powershell
python scripts/style_picker.py
```

開啟 <http://127.0.0.1:8766>，取消不要的風格，再按「儲存並套用」。風格庫包含 288 種匯入風格，以及 6 種原有角色風格，共 294 種。可依分類、關鍵字或已取消狀態篩選；「取消目前結果」只修改篩選出的項目，「取消全部」會清空畫面選擇，但至少保留一種才能儲存。點圖片可以放大比較，切換八種場景不會改動勾選。

「原有角色風格」分類包含水彩繪本、柔和粉彩、手繪漫畫、電影感厚塗、3D 遊戲風與實景角色融合。這六種保留原本完整的描述、構圖、配色、動態與避用條件，並已加入勾選清單；原先 288 種的勾選結果保持不變。這六種沒有原圖庫的場景預覽，可點開查看描述。

完整風格描述存放在 `configs/styles/krea2_catalog.json`；你的勾選結果存放在 `configs/styles/krea2_selection.yaml`。Kirby 的 `generation.creative_profile.style` 使用 `catalog: krea2` 與 `mode: random`。每次建立新的生成請求時，會從已勾選項目等機率抽一種，並沿用既有 `visual_style_profile` 提示詞路徑。既有執行中的生成保留已選好的風格。

圖文卡的每頁背景也使用該次抽中的風格，包含構圖、配色與避用條件，不會加入影片專用動態指示。Kirby 的圖文卡設定已移除額外固定的水彩風格及兩個自造配角。

貼圖與遊戲 sprite 只把風格套用到角色的媒材、配色與動態，不套用完整場景構圖或可能禁止隔離素材的風格排除條件。透明、純色與色鍵背景要求優先；風格描述不會覆蓋去背用途的輸出契約。使用者另行指定的負面提示仍保留。

儲存會直接更新 selection YAML，不必匯出後手動搬檔。重開頁面會讀取 config；同一瀏覽器也會暫存尚未儲存的勾選。舊頁面若遇到設定已被其他頁面更新，會顯示衝突並拒絕覆寫。

圖片直接引用原圖庫的公開預覽，因此看圖需要連線；風格資料與已儲存的 random 清單可離線使用。預覽未載入時仍能查看風格文字與操作勾選。

來源：[matplinta / Krea 2 Style Library](https://github.com/matplinta/t2i-krea-2-style-library)。2026-10-15 匯入 288 筆描述及 2,304 筆預覽路徑，來源與原始資料 SHA-256 記錄於 catalog。預覽是原作者的生成範例，不代表本機已重跑。這是 prompt style library，並非載入 Style Reference LoRA 或 Krea 雲端 moodboard conditioning。
