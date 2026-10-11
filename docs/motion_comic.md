# 分格逐步揭示漫畫（motion_comic）

`motion_comic` 是正式生成策略。輸入角色與題材，由既有 LLM 寫三格因果故事，
Krea 逐格生成插畫，再用 Pillow／FFmpeg 合成直式 MP4。第一段顯示起因，
第二段保留起因並揭示變化，第三段保留前兩格，以較大的結果格收尾。
角色圖像是靜態插畫，動態來自分格揭示與淡入；成片附靜音 AAC。

## 使用

從 repo 根目錄執行，不需 `--story` 或預先製作任何故事 JSON：

```powershell
python run_media_interface.py --character kirby --generation-type motion_comic --prompt "只想吃一小口蛋糕，卻忍不住一直吃" --no-publish
```

省略 `--prompt` 時，沿用角色流程的自主題材生成，再由漫畫寫作任務產生三格故事。
角色群組／多角色選取與新聞上下文仍使用現有服務，需有效的設定與憑證。
`--character kirby` 指設定群組；實際主角沿用該次已選定角色，不寫死為 Kirby。

預設 9 秒、576×1024、24 FPS；三格各 3 秒。圖片預設 Krea2 Turbo、
1024×768、8 steps。可用 `--duration-seconds` 改總閱讀時間，或修改角色設定的
`generation.motion_comic`。總片長乘 FPS 必須是整數，且至少有三個影格。
漫畫停用角色全域加速，避免縮短字幕閱讀時間。
LLM 的標題上限 12 字、每格字幕上限 24 字；插畫不生成文字，由合成器精確排字。

缺少 LLM、無法取得有效故事或圖片時，執行會失敗並保留紀錄，不替換為固定故事。
需要既有 ComfyUI、Krea 資產、LLM、Pillow、FFmpeg 與 CJK 字型；沒有新增套件。

## 隨機策略

`configs/routing.yaml` 已登錄候選、Krea 圖片階段與三張圖片的數量契約。
`configs/characters/kirby.yaml` 已加入正權重：

```yaml
generation:
  generation_type_weights:
    motion_comic: 1
```

角色 CLI／排程使用既有加權隨機選取。省略 `--generation-type` 即參與策略抽選；
權重相對於同一角色的其他正權重，`1` 並不表示固定百分比。
其他角色設定需自行給此策略正權重。明確指定策略或既有片長政策優先於抽選。

## 產物與審查

每次執行保存 `story_draft.json`（LLM 原創故事與模型資訊）、
`panel_generation.json`（各格 prompt、workflow、seed、steps、來源路徑）、
`story.json`（實際合成配方）、`motion_comic_summary.json`、MP4 與技術 QA 接觸圖。
發布素材選取只交付最終 MP4；發布文案使用實際生成故事及成片接觸圖。
沿用現有最終審查／發布流程，不走多張候選圖的前置選圖流程。

故事 schema 只驗證三格順序、必要文字與長度；技術 QA 只觀察媒體。
每格圖片使用自己的完整場景與狀態描述，共用已選角色身分及畫風；
LLM 的 `continuity` 是規劃／審查資料，不拼回圖片提示，以免原始道具狀態污染結尾。
兩者都不能保證圖片真的呈現字幕所說的事件。必須另看主題、因果變化、圖文一致性、
角色與道具延續，以及字幕閱讀節奏。生成摘要保持 `awaiting_review`，
不把一次 LLM 或 FFmpeg 成功當作品質通過。
上例的 `--no-publish` 只生成，沒有呼叫 Discord 或社群派送。

## 固定素材比較

`scripts/run_motion_comic_experiment.py` 是配對實驗入口，仍要求已存在的 `--story`。
這是為了把來源圖、文字、時長、尺寸與 FPS 固定，只比較漫畫和單圖剪輯呈現。
可以將正式策略自動保存的 `story.json` 交給它：

```powershell
python scripts/run_motion_comic_experiment.py --story "output/<生成目錄>/<comic目錄>/story.json" --output-dir output/comic_pair_01 --repeats 3 --warmup 1
```

輸出目錄必須新建，既有結果不覆寫。計時不包括共用圖片與 LLM 生成。
正式策略與配對實驗共用同一合成器；效率與觀感結論需各自的實際紀錄支持。

## 2026-10-11 驗證邊界

本機完整驗收為 696 passed、1 skipped、2 deselected，Ruff 通過。
另外量測既有 CLI 子程序驗收並合併完整測試資料，新策略五個模組的實際分支
覆蓋為 57／62（91.94%）；合併報告是 `coverage_with_children.json`。
新策略本機驗收使用符合真實 provider 契約的替身，以及真實 Pillow／FFmpeg；
涵蓋無故事檔、無題材、隨機路由、社群文案素材準備、非法故事、缺圖／壞圖與
共用狀態污染結尾的回歸。沒有以這些本機案例宣稱 GPU 品質通過。

另跑三輪真實 LLM／ComfyUI，自動保存故事與三張插畫、九秒 MP4。
最後一輪證據在 `output/motion_comic_automatic_final_20261011/validation_summary.json`，
逐格檢視紀錄在同目錄的 `perceptual_review.json`。
前兩輪分別有重複盤子與結尾蛋糕未消失的問題，保留原始結果；最後一輪
已呈現承諾、較小的剩餘蛋糕與空盤結尾，但第二格的預定缺口不夠明確。
這是基本敘事可見的篩選結果，仍待使用者審查。

主題、角色、Krea workflow、steps、seed、尺寸與 FPS 固定，但每輪重寫 LLM 故事，
並非只改單一圖片提示的嚴格配對；沒有宣稱全面品質提升或效率提升。
合成器抽出共用後，兩種版型共六頁與先前接受的頁面逐像素相同。
MySQL 角色選取、Discord、實際社群派送與 Docker 排程未在本輪驗證。
