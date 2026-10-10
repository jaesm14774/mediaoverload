# H3 進階策略：WanGP

MediaOverload H3 影片生成 **Powered by WanGP**。完整設定與安裝見 [WanGP H3 workflows](wan2gp_h3.md)。

- T2VA：只有 prompt，沒有首幀或尾幀。
- I2VA：Krea / 既有圖片經選擇後送入 `image_start`，flag `S`。
- FL2VA：經選擇的首尾圖送入 `image_start` / `image_end`，flag `SE`。既有 opening-conditioned img2img 尾幀策略仍由 Krea 執行。
- L2VA：只有 `image_end`，flag `E`。模型自由生成開頭，再收束到尾圖；不會自動補首圖。
- Ref2VA：使用專屬 Ref2VA checkpoint。圖片送入 `image_refs`，flag `I`；影片送入 `video_guide` / `video_guide2` / `video_guide3`，使用 reference-video flags。

原有 `native_h3_*` generation type、storyboard、候選圖選擇與人工審核契約仍是使用入口。正式影片工具統一為 `wan2gp.render_h3`，五個 template 位於 `configs/workflow/wan2gp_h3_*.json`。

Ref2VA 的參考圖描述外觀、角色、場景等特徵，不保證第一幀等於參考圖。若第一幀必須固定，使用 I2VA / FL2VA。圖片上限九張，影片上限三個；專案目前不接 reference audio，H3 自行生成音訊。

WanGP 上游還提供續接影片、指定位置插幀、控制影片與 V2V edit、音訊參考等能力。這些尚未成為本專案的正式策略。圖片／影片參考與首尾幀是已實作的五種策略。

實跑驗證：`python scripts/validate_wan2gp_h3.py`。Krea-to-H3 完整模式 runner：`python scripts/run_h3_modes_e2e.py --help`。日常介面仍為 `run_media_interface.py`；不需要另外啟動 WanGP WebUI。
