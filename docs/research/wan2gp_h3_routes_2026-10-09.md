# H3 各策略改用 WanGP：實跑紀錄（2026-10-09）

MediaOverload 的 H3 影片生成已改接 **WanGP 官方 Python API**，五種 production workflow 均已登錄並完成 GPU 實跑。Krea 生圖仍使用 ComfyUI。Powered by WanGP。

## 已完成的路徑

| 現有策略／用途 | 新 workflow | 實際 conditioning |
|---|---|---|
| `native_h3_t2v_story` 文生影 | `wan2gp_h3_t2va` | 無圖，FL2VA checkpoint |
| `native_h3_story` 圖生影、一般圖片動畫 | `wan2gp_h3_i2va` | 首圖 `image_start`，`S` |
| `native_h3_fl2va_story` 首尾圖生影 | `wan2gp_h3_fl2va` | 首圖＋尾圖，`SE` |
| `native_h3_l2va_story` 結尾圖生影 | `wan2gp_h3_l2va` | 僅尾圖 `image_end`，`E` |
| `native_h3_ref2va`、`text2image2native_h3_ref2va` 參考生影 | `wan2gp_h3_ref2va` | **Ref2VA checkpoint**，圖片 `I`、影片 `V-U` 等旗標 |

Ref2VA 的圖片是外觀／角色／場景參考，不等同於固定首幀。圖片與影片可以混合，保留 `<Picture N>`／`<Video N>` 順序。現行契約上限為九張圖、三支影片；本次 GPU 實跑分別用了兩張圖、一支影片。多影片與混合輸入的設定契約已測試，尚未逐一做 GPU 生成。

Long-video 分段、動畫貼圖與 game-sprite 動作生成的 H3 呼叫也改為 `wan2gp.render_h3`。原本的 prompt、Krea 候選圖、人審、剪輯、QA 與發布流程仍在各自的層。舊 Comfy H3 production graph、工具註冊與 fallback 候選已移除；歷史比較圖只放在 `scripts/fixtures/`。

上游另外具備續接影片、指定位置插幀、控制影片／影片編輯、參考音訊等能力，本次未新增成產品策略。現行參考音訊限制維持原契約，輸出仍有 H3 原生音訊。依據：[固定版本 H3 handler](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/models/minimax_h3/minimax_h3_handler.py)、[設定旗標](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/docs/SETTINGS.md)。

## 六種短片的實測

硬體：RTX 4060 8GB、i5-14500、128GB RAM。WanGP 17.17，revision `6479db36bdc2619a904a852bba9c2d78e1a83f82`；profile 4、Sage2、Spectrum、16 steps、Q4 diffusion＋Q4 text encoder、seed `20261008`。

每支都是 **608×352、124 frames、24 FPS、5.1667 秒**。四種首尾圖策略使用相同的 785 字元基礎 prompt。Ref2VA 加上 Picture／Video 標籤說明，且使用不同 checkpoint，因此這張表是各路徑實際耗時，不是同輸入的模型性能排名。

| 路徑 | WanGP 生成秒數 | app 工具總耗時秒數 | conditioning | 媒體 QA |
|---|---:|---:|---|---|
| T2VA | 213.76 | 238.90 | 無圖 | 通過 |
| I2VA | 215.89 | 221.90 | 1 張首圖 | 通過 |
| FL2VA | 226.26 | 232.17 | 首圖＋尾圖 | 通過 |
| L2VA | 219.68 | 225.64 | 1 張尾圖 | 通過 |
| Ref2VA 圖片 | 259.67 | 266.56 | 2 張參考圖 | 通過 |
| Ref2VA 影片 | 466.63 | 473.93 | 1 支參考影片 | 通過 |

生成秒數包含模型載入、編碼、去噪、VAE 解碼及儲存，排除 worker 初始化與 app 交付。工具總耗時包含初始化／交付：本批只第一支 T2VA 承擔 24.11 秒初始化；後續共用 worker，每支結束透過公開 `session.close()` 釋放模型。每條路徑僅一次成功樣本，尚無多輪中位數或各模式的 Comfy 配對測試。

六支都用真實 FFmpeg／ffprobe 檢查：尺寸、幀數、FPS、時長、音訊、立體聲及音畫時長對齊通過。已檢視六支抽幀，可見 Kirby 接近、接住並舉起種子的動作；尚未完成盲評、完整播放聽音或品質優於 Comfy 的比較。

另以五種策略的真實生成 MP4，執行實際 planner 的 `native-h3-speed` 與 `native-h3-qa` 後段：2x 後為約 2.584 秒，24 FPS、立體聲、音畫對齊均通過，紀錄在 `native_suffix_qa.json`。該實跑發現並修正 Ref2VA 的 QA 仍使用原始 goal 時長，以及各 native 路徑的音訊檢查未統一的問題；QA 仍只記錄人審證據，不替人審做決定。

輸入 SHA256、effective settings、events、MP4、QA 與抽幀保存在 `output/benchmarks/wan2gp_h3_routes_20261009/`。可先看 `results.json`、`timings.json` 與各路徑的 `wan2gp_h3_summary.json`。

## 原本 15 秒 L2VA 的歷史重播

WanGP 長片重播成功：**總耗時 2212.87 秒（36 分 52.87 秒）**，相較原 Comfy render node **1.7062 倍加速，節省 41.39% 時間**。WanGP job 生成為 2185.59 秒，worker 初始化 24.61 秒；總耗時另外包含交付檔案與探測，排除最後的 worker 關閉。

來源 run `5fbf5fbe6369`，原始 Comfy native-render node 耗時 **3775.52 秒（62 分 55.52 秒）**。保留原始 10,076 字元 prompt、結尾圖、實際 seed `463948145`、512×640、362 frames、24 FPS、16 steps、Q4 diffusion＋Q2 text encoder。

從原始 MP4 的內嵌 `prompt` 執行 graph 再核對，而非僅依模板推測：prompt、尺寸、幀數、steps、seed、`res_multistep` solver、video shift 12 均相同；原始 audio shift 為 3。原 Comfy input 中保留的上傳結尾 PNG 與本次檔案 SHA256 完全相同。證據在 `historical_comfy_embedded_graph.json`、`matched_input_evidence.json`。兩邊均啟用 Spectrum，但實作／設定、offload、attention 與 runtime 不同。

舊 prompt 含有現行 scene-only contract 禁止的導演／新聞背景指令；本次以獨立歷史 benchmark 直接使用官方 API，保留原文，不修改正式驗證。另修正單一任務提交設定為 `multi_prompts_gen_type=FG`，防止 prompt 空白段落被預設 `PG` 拆成多支影片。正式提交亦使用 FG，測試已涵蓋段落保留。

結果在 `output/benchmarks/wan2gp_h3_15s_replay_20261009/replay_report.json`。這是與留存的 Comfy 執行紀錄比較，不是今天交替重跑的 A/B；backend、Torch、attention、offload 與 Spectrum 設定不同，倍率只適用於這個輸入與機器。15 秒 frame grid 實際為 362/24＝15.0833 秒。

原始 WanGP MP4 與交付 MP4 的尺寸／幀數皆為 512×640／362，沒有透過縮尺寸或減幀取得上述速度。兩邊的媒體 QA 均通過：24 FPS、立體聲 32kHz、音畫時長對齊。QA 與九格抽幀分別在 `historical_comfy_qa.json`、`wan2gp_qa.json`，兩張 contact sheet 可直接比較。此倍率涵蓋 native-render 階段，未包含 Krea、LLM、人工等待與發布；重播早段有本機測試／CPU QA 同時執行，亦非完全隔離的性能實驗。

## 使用與驗證邊界

本機 `D:/Wan2GP_benchmark` 已準備 FL2VA／Ref2VA 的 Q4、Q2 定義；`configs/wan2gp.yaml` 指向該隔離環境。正式 default 為 Q4。音訊 VAE 使用 WanGP 原始 weight-normalization 格式，不能拿 Comfy fused 格式直接替代。安裝、API 生命週期與交付規則見 [操作文件](../wan2gp_h3.md)。

```powershell
# 重跑六種 GPU 接口案例，從留存基礎影片抽首尾圖：
python scripts/validate_wan2gp_h3.py
# 真實 Krea 生成首尾／參考圖的整合 runner：
python scripts/run_h3_modes_e2e.py --mode i2va --smoke
# 重播上述歷史 15 秒 L2VA benchmark：
python scripts/replay_h3_on_wan2gp.py
```

本次未重新跑 LLM→Krea→人審→發布的完整產品流程，沒有發送社群內容。GPU 真實生成證據與本機測試證據分開列示。

正式 skill 接線另已實跑：保留首圖與多段落場景 prompt，透過實際 planner 的 `longvideo.render_native_h3`→`wan2gp.render_h3`→GPU→native QA，產出一支 608×352／124 幀／24 FPS 的立體聲影片。原文與 FG 提交方式、媒體 QA 均通過，證據為 `native_skill_smoke.json`。這個 1-step 案例僅驗證接線，不列入上述 16-step 耗時／品質比較。可重跑本批證據目錄下的 `validate_native_skill.py`、`validate_media_suffix.py`。

兩個新增 host 模組合併測試與 GPU trace 的覆蓋率（行與分支合計）為 84.91%（asset 模組 94%、render/client 模組 84%），純分支為 67/88＝76.14%；獨立 interpreter 的 worker 不在此 coverage 分母。尚未達到 90% 分支診斷目標；主要未涵蓋 occupied queue、worker 初始化失敗／逾時、取消、強制清理及缺失 MP4 的防禦分支。沒有為提高數字而佔用原 Comfy queue、故意中斷 GPU 或添加 mocks。精確分支數見 `coverage_combined.json`；正向 GPU 接口證據另外留存。

完整本機測試 `test_suite_latest.log`：611 passed、4 failed、1 skipped、1 deselected。四個失敗涉及原有 sprite strategy-context／reference-pack 欄位，以及既有 system prompt 的 `news-grounded` 字串斷言；本次前的設定快照已沒有該 strategy-context 區塊，system-prompt 來源檔亦是原有 dirty work。未修改這些產品語意來讓測試通過。之後新增的段落契約連同 WanGP 專項共 12 tests 通過，使用真實圖片、影片與 FFmpeg，沒有新增 mocks。

WanGP 授權與模型授權需分別看待；WanGP 本機使用條件見 [Community License 2.0](https://github.com/deepbeepmeep/Wan2GP/blob/6479db36bdc2619a904a852bba9c2d78e1a83f82/LICENSE.txt)，商業軟體／API／SaaS 整合有額外條件。
