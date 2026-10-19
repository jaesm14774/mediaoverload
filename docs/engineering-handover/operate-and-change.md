# 上手、觀察與安全變更

[回到工程交接入口](index.md)。本頁只提供能從 repo 驗證的本機入口、run 記錄解讀和程式變更線索。正式環境如何部署、復原或升級，需由服務 owner 提供適用環境的核准 runbook。

## 先做不會生成媒體的檢查

在 repo 根目錄查看手動角色介面：

```powershell
python run_media_interface.py --help
```

如果已安裝 Agentic CLI，可先列印計畫，不要加 `--execute`：

```powershell
agentic --goal "A hero catches a falling seed" --media-type image
```

CLI 的 `main()` 會先建立 goal 和 plan，印出 `=== PLAN ===`；程式只有看到 `--execute` 才進入 runner。這項檢查不證明所列 ComfyUI 資產在 GPU 服務端可用，也不驗證真正生成結果。安裝、FFmpeg／ffprobe 和必要環境設定請依[專案 README](../../README.md)；不要把 `media_overload.env.example` 的範例值當成有效憑證。

## 實際生成前確認哪些邊界

角色流程可能依路徑呼叫 ComfyUI、文字／視覺 LLM、FFmpeg／Pillow、Discord 或社群平台；新聞和角色群組功能可能使用 MySQL。實際請求是否走到這些服務，取決於策略、審查設定、角色設定和執行模式。repo 的 README 說明 Python 依賴、ComfyUI 設定和本機驗證命令；部署範例見 [`Dockerfile`](../../Dockerfile)、[`docker-compose.yml`](../../docker-compose.yml) 和 [`media_overload.env.example`](../../media_overload.env.example)。這些是版本控制內的定義與範例，不是當前部署狀態。

若要做一次受控的真實生成，先確認已獲授權的 ComfyUI／模型資產、LLM 供應商、輸出位置和必要審查渠道。角色 CLI 的 `--no-publish` 可略過最後的社群發布階段，但仍會執行生成，也可能進入中途 Discord 審查。`--dry-run-publish` 不會把整個請求變成無副作用模擬；它只把最終 publish dispatch 設為 dry run。不要在沒有指定環境和執行授權時，用正式憑證嘗試一條看似「dry run」的完整生成命令。

## 從一筆 run 開始排查

角色流程將媒體寫入 `output/<character>/`（角色設定或 `--output-dir` 可改位置；解析規則見 [`character_config.py`](../../agentic/src/agentic/app/character_config.py#L198)），並將每次 run 的追蹤資料放在 `logs/runs/<run_id>/`。直接 Agentic CLI 的預設媒體根目錄和 portfolio 摘要則在 `agentic/output/` 與 `agentic/logs/`；不要用角色目錄的慣例去猜直接 CLI 的產物位置。細節見[README 的產物表](../../README.md#設定與產物)和 [`RunRecorder`](../../agentic/src/agentic/runtime/observability.py#L36)。

| 你看到的情況 | 先看什麼 | 能回答什麼／下一步 |
| --- | --- | --- |
| 角色命令回傳 `failed` | `run_manifest.json` 的 `failure_node`、`failure_skill`、`failure_reason`；再找 `nodes/<node_id>_attempt_<n>.json` | 找出第一個失敗節點及其 logs／outputs，再依該 skill 或 tool 追到設定、provider 或工作流。不要先重跑整個生成。 |
| 程序提前結束或結果不完整 | `events.jsonl` 最後幾個事件、最後一筆 node／LLM 記錄；確認 manifest 是否由程序收尾程式標成 failed | `RunRecorder` 會嘗試為未正常結束的 run 建立失敗 manifest，reason 會提示查看最後未完成操作。先記下 run ID 與最後訊號；不要刪除不完整證據。 |
| 有媒體但沒有完成審查或發布 | manifest 的 `stage_status`、`publish.result`、`publication_state`、`publicly_visible`，以及相關 `review.assets.select` node | 分辨是 render、review、publish 哪一階段未完成。生成成功不等於核准；dry run、private、draft 或 container-only 結果也不等於公開。 |
| 必要 Discord 審查被阻擋 | node 輸出的 `review_mode`、`fallback_reason`；輸出目錄的 `review_sessions/` 記錄；依核准途徑確認 bot／channel／reviewer 存取 | 當必要 final review 未能啟動，程式會停止在審查關卡。不要以自動候選排序當成人工核准。中途審查的傳輸失敗行為則依該策略的 `pre_video_review_failure_policy`。 |
| workflow 報告缺少資產或 ComfyUI 節點失敗 | 對應 node 的 workflow 名稱、資產檢查輸出與 ComfyUI 診斷；再對照 `configs/workflow/` 的 JSON 和 metadata | 確認本機設定路徑與實際 ComfyUI 環境是否一致。repo 不能證明遠端 ComfyUI 的模型已安裝；不要自行清除模型或工作佇列。 |

`events.jsonl` 依時間記錄 run 與 node 事件；`nodes/` 保存 skill 輸出、metrics 和 logs；`llm/` 保存 LLM 呼叫資料；`run_manifest.json` 彙整最終狀態和路徑。LLM 記錄包含送出的 `messages` 和原始回應欄位，可能帶有 prompt 或引用內容，應依可能敏感的工作內容限制存取和分享；格式見 [`RunRecorder.record_llm_call()`](../../agentic/src/agentic/runtime/observability.py#L71)。repo 沒有寫明保留天數；在 owner 確認政策前，不要清理 run logs、review sessions 或輸出媒體。

若 Discord 顯示「重新生成」，[`run_character_session()`](../../agentic/src/agentic/app/character_session.py#L14) 會停止本輪並從頭建立另一個完整 run；若舊請求有固定 `--seed`，新 run 會清除該固定 seed。新 run 有新 ID，舊 run 證據不會被當成本輪結果。先從回傳 JSON 確認新 run ID 與狀態，避免把舊訊息或產物當成新結果。

## 修改前先追影響鏈

| 你要改的地方 | 可能改變的結果 | 修改前先核對 |
| --- | --- | --- |
| `configs/characters/*.yaml` | 角色身份、提示、策略候選／權重、輸出位置、審查與發布平台 | 角色 selection mode、`configs/routing.yaml` 的對應策略，以及最終產物和 review gate。 |
| `configs/routing.yaml` | 哪種情境可以選到哪個 strategy/workflow、候選數、審查數量和失敗分支 | strategy 到 plan builder 的對應、workflow 所需輸入／資產、拒絕與 service-failure 行為。 |
| `configs/workflow/*.json` 或 workflow metadata | ComfyUI 節點輸入、模型依賴、輸出路徑與媒體格式 | 對應的 asset manifest、ComfyUI 端的安裝資產，以及相關 workflow／provider contract 測試。 |
| planner、runtime skills 或 tools | 節點順序、資料契約、重試／失敗結果及對外呼叫 | `ExecutionPlan` 的依賴關係、`RunRecord` 輸出、consumer node 和該行為的 end-to-end／契約測試。 |
| Discord review 或 social adapter | 哪些媒體可被接受，以及 dispatch 是否會對外建立內容 | 審查是否必須、接受／編輯／拒絕狀態、`dry_run` 和平台回執語意；任何 live 驗收都需核准憑證和專用測試帳號。 |
| scheduler 設定或生命週期 | 自動生成頻率、使用的策略、服務重啟時機及可能重複的工作 | 目前執行實例、進行中工作、時間設定和正式 runbook。不要為了測試重啟正在處理生成工作的排程器。 |

repo README 提供 Ruff、非 integration pytest 和 integration test collection 命令，也說明目前 repo 未包含 CI workflow；依修改影響選最小本機範圍，再安排必要的外部驗收。此次交接文件更新沒有修改產品程式，未執行程式測試或 GPU／LLM／ComfyUI／社群平台驗收。驗證邊界見[README](../../README.md#驗證)。

## 排程與正式環境邊界

[`scheduler/scheduler.py`](../../scheduler/scheduler.py#L106) 讀取 `SCHEDULER_*` 設定，可依 daily 或 interval 觸發角色工作流；工作註冊入口在 [`register_jobs()`](../../scheduler/scheduler.py#L216)，而 [`docker-compose.yml`](../../docker-compose.yml) 定義 `media-scheduler` 容器範例。repo 也提供範例環境檔，但不能由此判斷目前 scheduler 是否啟用、執行在哪一台主機、使用哪一份 env，或是否有另一套部署方式。使用者也無法由 repo 得知當前排程是否正在執行。

| 待確認資訊 | 為何影響安全接手 | repo 內目前可確認的線索 | 確認前的做法 |
| --- | --- | --- | --- |
| 系統與排程 owner、值班／升級對象 | 決定誰能批准排程、平台和事故處置 | 本 repo 未列出負責團隊或聯絡方式 | 先從組織內的服務目錄或現任 owner 確認；不要以作者或 repo contributor 推定值班人。 |
| 正式部署環境、目前 scheduler 狀態、設定來源 | compose 與 env 範例不等於 live 狀態 | `Dockerfile`、`docker-compose.yml`、`media_overload.env.example` | 不啟動、停止或重啟正式 scheduler；先由 owner 指定觀察工具和部署 runbook。 |
| 正式生成、重跑、補償與回復流程 | 重跑可能再次消耗算力、建立新產物或觸發發布 | 程式記錄單次 run，但 repo 未提供核准事故 playbook | 保存 run ID、manifest 和關鍵 node evidence 後升級；未確認前不盲目重跑。 |
| 憑證取得、Discord reviewer 授權與平台測試帳號 | 憑證控制資料與外部發布能力 | `media_overload.env.example` 列出變數名稱及 `discord_review_allowed_user_ids` | 使用正式核准的密鑰管理／存取申請流程；具體路徑需 owner 提供。不要在 issue、文件或 log 貼秘密。 |
| 產物、LLM 日誌及 review sessions 的保存／刪除期限 | 可能涉及內容、引用資料、供應商回覆和稽核證據 | `RunRecorder` 會保存 node 與 LLM 呼叫內容；repo 未聲明保留政策 | 把資料當成可能含敏感內容，限制分享；owner 確認政策前不做批次刪除。 |
| SLA、告警和成功／逾時門檻 | 決定何時通知使用者或擴大事故 | repo 有 run-level log 與狀態，沒有可驗證的服務目標或正式告警文件 | 以明確失敗 node 和 run ID 交給 owner，別推定成功狀態等同服務健康。 |

正式環境操作手冊沒有在本 checkout 中被識別，因此本指南不編造部署、回滾、清除 queue、重啟服務或社群撤文命令。由 owner 提供適用環境的正式 runbook 後，應將其連結加回此頁並標示適用環境。
