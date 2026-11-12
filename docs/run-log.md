# Run Log

Run Log 是寫進 MySQL 的營運帳本，回答「跑了什麼、人有沒有接受、發到哪裡」。它和下列記錄分工不同：

| 記錄 | 用途 |
|---|---|
| `logs/runs/<run_id>/` | 單次 run 的除錯檔（事件、node、LLM） |
| 連載資料庫（`serial_video_mysql_database`） | 連載影片自己的審核與發布帳本 |
| **Run Log（本文件）** | 跨 run 查詢：策略、審核結論、發布平台與時間 |

> 程式內的 `RunRecord`（`runtime/contracts.py`）是單一 node 的記憶體記錄，與資料庫無關。

## 啟用

在 `media_overload.env` 設定 `run_log_mysql_database`、`run_log_mysql_user`、`run_log_mysql_password`（`mysql_host` / `mysql_port` 共用）。資料庫須先存在，資料表第一次使用時自動建立。

只有入口會啟用：`run_media_interface.py`（手動）和 `scheduler/scheduler.py`（排程）。直接呼叫 `run_character_workflow` 或 `run_character_session` 且不傳 `run_log` 時不記錄，連載流程內部為產生角色參考圖而呼叫的 run 也不會記錄。

- 未設定 `run_log_mysql_database`：不寫 Run Log，run 照常執行。
- 已設定但連不上：run 在開始生成前就失敗，避免跑完 GPU 才發現沒有記錄。排程是每個 job 開始時檢查，連不上只跳過該次 job。
- run 開始後寫入失敗：只寫入 run 的 log 檔並繼續，不會讓已發布的 run 變成錯誤而被重試重發；該列會停在 `running`。

## 資料表

`runs`：一次 run 一列。run 開始時先寫 `status='running'`，結束時更新，所以被中斷的 run 會停在 `running`。

| 欄位 | 說明 |
|---|---|
| `run_id` | 與 `logs/runs/<run_id>/` 相同 |
| `strategy` / `media_type` / `workflow_name` / `character_name` | 路由結果 |
| `prompt` | 實際使用的 prompt |
| `news_title` / `news_url` | 新聞驅動時的來源 |
| `status` | `running`、`success`、`failed`、`regeneration_requested`、`awaiting_user_action` |
| `review_outcome` | 審核結論，見下表；連載 run 為 `NULL`（結論在連載帳本） |
| `dry_run` | 發布是否為 dry run |
| `failure_reason` | 失敗原因 |
| `output_paths` | 這次 run 的最終成品位置（JSON 陣列），見下方「成品路徑」 |
| `started_at` / `finished_at` | UTC，`DATETIME(6)` |

`run_reviews`：每個 Discord 審核階段一列，依發生順序編號（`ordinal`）。

| 欄位 | 說明 |
|---|---|
| `plan` | `generation`（生成階段）或 `publish`（最終發布審核） |
| `stage` | `review_phase`，沒有則 `review_scope`（如 `opening_frame`、`final_media`），再沒有則 node id |
| `decision` | `approved`、`rejected`、`regenerate`、`timed_out`、`invalid_selection`、`unavailable`、`automatic` |
| `reviewer` / `review_session_id` | 審核人與 `output/review_sessions/` 的對應檔 |
| `selected_count` / `candidate_count` | 選了幾個、共幾個候選 |

`run_publications`：每個平台一列。

| 欄位 | 說明 |
|---|---|
| `platform` | `twitter`、`facebook`、`instagram_graph`、`threads`、`youtube`、`facebook_profile_handoff` |
| `outcome` | `published`、`failed`、`skipped`（原因在 `error`）、`awaiting_user_action` |
| `external_id` / `receipt_status` / `visibility` / `verified` | 平台回傳的收據 |
| `receipt_details` | 收據細節 JSON（如 Threads permalink） |
| `caption` / `media_paths` | 實際送出的文案與媒體 |
| `dispatched_at` | 該次平台派送結束的時間（UTC），不論成功、失敗或跳過 |

dry run 不會有 `run_publications` 列。派送本身丟出例外（沒有逐平台結果）時，也不會有列，原因在 `runs.failure_reason`。

## 成品路徑

`runs.output_paths` 與 `run_publications.media_paths` 都存**相對專案根目錄、以 `/` 分隔**的路徑，例如 `output/kingdedede/20261112_175800_.../final.png`。

Windows 跑的是 `C:\...\mediaoverload\output\...`，Docker 跑的是 `/app/output/...`。寫入時各自扣掉自己看到的專案根目錄（`CharacterWorkflowRequest.repo_root`：Windows 資料夾或容器內的 `/app`），所以兩邊存的是同一個值；要開檔就接上目前環境的專案根目錄。專案目錄之外的檔案（`output_dir` 設成別處的絕對路徑）保留絕對路徑。

成品從 run 的最終結果讀取，所以被審核拒絕的 run 也會記錄，方便回頭看。以例外中斷的 run 沒有最終結果可讀，一律是空陣列（即使檔案已在磁碟上，需到 `output/` 或 `logs/runs/<run_id>/` 找）。連載影片（`serial_video`）的成品在連載帳本，這裡是空陣列。

升級舊資料庫：`CREATE TABLE IF NOT EXISTS` 不會替既有資料表加欄位。已存在的 `runs` 需用管理帳號手動執行 `ALTER TABLE runs ADD COLUMN output_paths JSON NOT NULL DEFAULT (JSON_ARRAY()) AFTER failure_reason`（有預設值，可直接用在已有資料的表）。欄位有預設值，所以尚未重啟、仍跑舊程式的程序也能繼續寫入。

## 審核結論 `review_outcome`

只有真人決定算數；`automatic` 不算人工接受。

| 結論 | 條件 |
|---|---|
| `accepted` | 有 `approved`，且沒有其他未通過的階段 |
| `rejected` | 有 `rejected`，沒有 `approved` |
| `partial` | 前面階段 `approved`，後面階段 `rejected`、逾時、選取無效或無法審核（多階段才會發生） |
| `regenerated` | 任一階段按了重新生成（會另開新 run，不算 `rejected`） |
| `unreviewed` | 沒有任何人工結論（`--no-review`、逾時、Discord 不可用） |

只挑一部分候選（4 選 2）仍是 `accepted`，數量在 `selected_count` / `candidate_count`。逾時後若設定 `fallback_to_top`，媒體會繼續，但結論不是 `accepted`。

## 常用查詢

```sql
-- 各策略被接受的比例
SELECT strategy, review_outcome, COUNT(*) FROM runs GROUP BY strategy, review_outcome;

-- 最近發布到各平台的貼文
SELECT r.run_id, r.strategy, p.platform, p.outcome, p.external_id, p.dispatched_at
FROM run_publications p JOIN runs r USING (run_id)
WHERE p.outcome = 'published'
ORDER BY p.dispatched_at DESC LIMIT 20;

-- 停在 running 的 run（程序被中斷）
SELECT run_id, strategy, started_at FROM runs WHERE status = 'running' AND started_at < NOW() - INTERVAL 1 DAY;
```

## 已知限制

- Facebook Profile handoff 的貼文網址是之後人工回填的，Run Log 只記到 `awaiting_user_action`。
- 連載影片（`serial_video`）只在 `runs` 留一列；審核與發布在連載帳本。
- 同一個 run 內審核節點若被重試，只有帶 `review_decision` 的紀錄會被記錄。
