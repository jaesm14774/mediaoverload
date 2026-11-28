# 文件索引

每份文件只回答一種問題。新增或調整文件前先判斷歸屬，不要把「現在是什麼」和「接下來要做什麼」混寫在同一篇。

| 欄位 | 回答 | 內容 |
|---|---|---|
| architecture | 現在是什麼 | 目前真實生效的實作、邊界與契約；只寫已落地的事實 |
| guide | 怎麼做 | 操作與調校流程 |
| reference | 查什麼 | 穩定的清單與驗收契約 |
| plans | 接下來做什麼 | 研究、方案與尚未落地的計畫；落地後把事實移進 architecture |

## architecture

- [comfyui_h3.md](comfyui_h3.md)：H3 工作流與輸入契約
- [minimax_h3_prompting.md](minimax_h3_prompting.md)：H3 提示契約
- [serial-video.md](serial-video.md)：連載短片的狀態、審核與 MySQL
- [news-fusion-story.md](news-fusion-story.md)：新聞融合故事策略
- [real-world-intruder.md](real-world-intruder.md)：誤闖真實世界策略
- [game_sprite.md](game_sprite.md)：Game Sprite 輸出契約
- [run-log.md](run-log.md)：Run Log 營運帳本
- [llm-providers.md](llm-providers.md)：文字 LLM 供應商、備援鏈與 token 帳

## guide

- [krea2_comfyui_best_practice.md](krea2_comfyui_best_practice.md)：Krea2 工作流操作
- [image_prompt_guide.md](image_prompt_guide.md)：圖片提示指南
- [story_card_editorial_guide.md](story_card_editorial_guide.md)：Story Card 編輯指南
- [reference_video_remix.md](reference_video_remix.md)：影片參考分析
- [comfyui_nvidia_vfx_postprocess.md](comfyui_nvidia_vfx_postprocess.md)：NVIDIA VFX 後處理設定

## reference

- [model_sources.md](model_sources.md)：模型來源與 ComfyUI 資產清單
- [comfyui_h3_acceptance.md](comfyui_h3_acceptance.md)：H3 驗收契約
- [nvidia_vfx_postprocess_acceptance.md](nvidia_vfx_postprocess_acceptance.md)：NVIDIA VFX 驗收契約

## plans

- [comfyui-model-swap-scheduling.md](comfyui-model-swap-scheduling.md)：換模型排程研究與方案
- [pixelle-video-study.md](pixelle-video-study.md)：Pixelle-Video 研究，素材導向流程與非 H3 輸出的音訊後製

`oauth/` 是 OAuth 申請用的靜態頁面（隱私權與條款），不屬於上述文件。
