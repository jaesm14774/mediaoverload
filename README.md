# MediaOverload

依角色設定、使用者目標與約束，完成媒體生成、編輯、技術 QA、人工審查與社群發布。圖片與 H3 影片共用 ComfyUI 生成服務；應用程式透過 HTTP 提交工作流。

## 安裝

從此 repo 的 checkout 執行，使用 Python 3.11 以上：

```powershell
python -m pip install -r requirements.txt
# 開發與測試工具
python -m pip install -e "./agentic[dev]"
```

Python 依賴只在 [agentic/pyproject.toml](agentic/pyproject.toml) 宣告；根目錄 requirements 安裝同一套件。FFmpeg / ffprobe 必須可從 PATH 執行。設定與工作流仍由 repo 的 `configs/` 提供，因此此套件從 checkout 使用。

生成前需設定模型服務、LLM 與必要憑證。角色群組與新聞選取使用 MySQL；Discord 審片與各社群平台需要各自憑證。機敏設定不納入版本控制。

| 設定 | 用途 |
|---|---|
| `COMFYUI_HOST`, `COMFYUI_PORT`, `COMFYUI_ROOT` | ComfyUI 連線與本機資產 |
| `AGENTIC_LLM_MODE` | 角色流程預設 `llm`；`template` 不提供新聞故事、字卡或漫畫寫作 |
| `AGENTIC_TEXT_MODEL_PROVIDER`, `AGENTIC_TEXT_MODEL` | 文字模型供應商與型號 |
| `AGENTIC_VISION_MODEL_PROVIDER`, `AGENTIC_VISION_MODEL` | 視覺模型供應商與型號 |
| `AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS` | 單次 LLM 請求逾時 |

## 使用入口

```powershell
# 查看角色介面所有選項
python run_media_interface.py --help

# 角色生成；仍會走 Discord 審片，禁止社群實際發布
python run_media_interface.py --character kirby --prompt "Kirby接住滑落的星星" --generation-type text2image2video --dry-run-publish

# 只產生執行計畫
agentic --goal "A hero catches a falling seed" --media-type image

# 不需 GPU 或 LLM 的本機圖片編輯
agentic --goal "Pan across the source image" --media-type image_sequence_edit --edit-input "D:/media/source.png" --edit-input-root "D:/media" --duration-seconds 2 --width 320 --height 320 --execute

# LLM 自動寫三格故事、生成插畫並合成漫畫短片；只生成、不發布
python run_media_interface.py --character kirby --generation-type motion_comic --prompt "只想吃一小口蛋糕，卻忍不住一直吃" --no-publish
```

`--dry-run-publish` 只禁止社群派送，生成與 Discord 審查仍會使用外部服務。未加 `--execute` 的 `agentic` 指令只輸出計畫。排程器入口為 `scheduler/scheduler.py`；不要為了測試重啟正在執行的生成工作。

## 設定與產物

| 路徑 | 責任 |
|---|---|
| `configs/characters/*.yaml` | 角色、群組、生成、發布設定 |
| `configs/routing.yaml` | 策略候選、權重、各階工作流與數量政策 |
| `configs/workflow/` | 模型工作流與資產 metadata；名稱以檔案 stem 為準 |
| `configs/storyboards/` | 目前故事與時間線契約 |
| `agentic/src/agentic/` | 應用協調、計畫、技能、模型與服務轉接 |
| `output/` | 圖片、影片、編輯、QA 與 review 產物 |
| `logs/` | run manifest、事件、LLM 請求、portfolio 記錄 |
| `.tmp-tests/` | 測試與建置暫存，不作為正式流程輸入 |

策略語意以 [strategies.py](agentic/src/agentic/runtime/strategies.py) 為準；已啟用候選以 routing YAML 為準。影片的模型型號與 conditioning 由工作流描述，新增同類模型通常只需新工作流與候選設定。

## 架構與維護

[架構與擴充指南](docs/architecture.md) 說明共用與策略專用責任、如何新增策略／模型／LLM，以及驗證邊界。

- [ComfyUI H3 工作流與輸入契約](docs/comfyui_h3.md)
- [H3 提示契約](docs/minimax_h3_prompting.md)
- [Krea2 工作流操作](docs/krea2_comfyui_best_practice.md)
- [圖片提示指南](docs/image_prompt_guide.md)
- [Story Card 編輯指南](docs/story_card_editorial_guide.md)
- [Game Sprite 輸出契約](docs/game_sprite.md)
- [生成策略候選與受控實驗](docs/strategy_experiments.md)
- [自動分格漫畫與隨機策略](docs/motion_comic.md)
- [影片參考分析](docs/reference_video_remix.md)

## 驗證

```powershell
python -m ruff check agentic/src agentic/tests scripts scheduler run_media_interface.py --config agentic/pyproject.toml
python -m pytest agentic/tests scripts/tests -m "not integration" -q
python -m pytest agentic/tests/integration --collect-only -q
```

本機測試包含真實 Pillow、FFmpeg、ffprobe 與本機 HTTP provider 契約，以及模型邊界的明確替身。計畫測試驗證真實 manifests、技能與工具可連接，但不代表 GPU 已生成。需要真實 DB、LLM、GPU、ComfyUI 或社群憑證的測試標為 `integration`，必須另行安排受控驗收。

目前 repo 未包含 CI workflow；上述命令是可重現的本機檢查，不宣稱有 hosted CI 或已完成正式環境驗收。
