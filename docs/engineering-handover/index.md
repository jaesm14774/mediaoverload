# MediaOverload 工程交接指南

> **適用基線：** `a113252cebf41f3e9fa399337914a26c99e1cd4b`（2026-10-19）
>
> **文件核對日：** 2026-10-19（Asia/Taipei）
>
> **交接狀態：** 程式庫維護範圍已覆蓋，讀者驗收待進行。正式環境值班與部署交接仍待補充。

## 這套系統替誰完成什麼工作

一位內容製作者想把一段角色或主題構想，做成圖片、影片、漫畫或其他媒體，並在需要時交給人審查、整理成發布素材。MediaOverload 依角色設定、使用者要求和路由規則組出生成工作，呼叫已配置的媒體工作流，執行技術檢查，再把產物和審查結果留下來。這裡的「內容製作者」是依 CLI 與 Discord 審查介面推導的代表讀者；repo 沒有正式列出使用者角色、服務承諾或影響等級。

這個 repo 同時提供兩種入口：角色工作流把生成、審查與可選發布串成一個完整工作階段；Agentic CLI 可先把目標轉成執行計畫，或在明確加上 `--execute` 後執行計畫。排程器沿用角色工作流。兩者都使用 Agentic 的規劃與執行元件，但直接 CLI 不等於角色工作流，也不會自動包含角色流程的最終發布審查。

若請求失敗，內容製作者可能拿不到預期素材；若審查或發布狀態看錯，則可能誤判素材已核准或已公開。成功生成、審查通過和社群公開是不同結果，必須分別確認。

## 安全的第一步

先在 repo 根目錄查看角色入口，不會啟動生成或連線外部服務：

```powershell
python run_media_interface.py --help
```

要理解直接 Agentic CLI 的計畫格式，可使用 README 所列的 `agentic --goal "A hero catches a falling seed" --media-type image`；不加 `--execute` 時，CLI 只建立並列印計畫，不執行工作流。需要安裝和環境設定時，從[專案 README](../../README.md)開始。任何實際生成都可能呼叫 LLM、ComfyUI、Discord 或社群服務，並寫入媒體與 run 記錄；先確認使用的環境和授權。

## 依你要完成的工作閱讀

| 你現在要做的事 | 從這裡開始 |
| --- | --- |
| 了解一筆角色媒體請求如何變成產物、在哪裡停下來或進入發布 | [核心流程與系統責任](workflow-and-architecture.md) |
| 排查一次失敗、找到 manifest／node／LLM 證據 | [上手、觀察與安全變更](operate-and-change.md#從一筆-run-開始排查) |
| 建立本機環境或確認直接 CLI 與排程器的入口 | [上手、觀察與安全變更](operate-and-change.md#先做不會生成媒體的檢查)與[README](../../README.md) |
| 評估修改角色策略、工作流、審查或發布的影響 | [上手、觀察與安全變更](operate-and-change.md#修改前先追影響鏈) |
| 查 H3、漫畫、風格或指定媒體契約 | [repo 的專題文件](../)與[README 的文件索引](../../README.md#架構與維護) |

## 接手範圍索引

| 能力／維護流程 | 使用者或觸發 | 結果與主要確認方式 | 維護入口 | 覆蓋狀態 |
| --- | --- | --- | --- | --- |
| 手動角色生成 | 操作者執行 `run_media_interface.py` | 結果 JSON、`output/<character>/` 產物及 `logs/runs/<run_id>/` | [入口與核心流程](workflow-and-architecture.md#一筆請求怎麼走到結果) | 本指南說明 |
| 排程角色生成 | Scheduler 到達設定時間 | scheduler log、同一角色流程的 run manifest | [排程與部署邊界](operate-and-change.md#排程與正式環境邊界) | 程式入口已說明；實際環境待確認 |
| 直接 Agentic 計畫／執行 | `agentic --goal ...` | 不加 `--execute` 列印計畫；加上後依 plan 執行 | [雙入口差異](workflow-and-architecture.md#兩種入口各自負責什麼) | 本指南說明 |
| 人工審查與可選發布 | 工作流需要素材選擇或 publish review | 審查紀錄、`stage_status`、發布 dispatch 結果 | [審查如何保護結果](workflow-and-architecture.md#兩道審查關卡) | 本指南說明；live provider 未驗收 |
| 開發環境與本機檢查 | 新維護者準備閱讀或修改程式 | README 命令、repo 本機 lint／tests | [README](../../README.md) | 正式文件承接 |

## 範圍、證據與尚未確認的事

本指南解釋此 checkout 中可由程式碼、設定和已提交文件核對的角色流程、直接 Agentic CLI、排程入口、產物位置與程式庫維護線索。它不是正式環境 runbook，也沒有以本次文件撰寫證明某個 GPU、ComfyUI、LLM、資料庫、Discord 或社群帳號目前可用。

程式庫未指定服務 owner、值班人員、部署環境、正式操作／復原 runbook、告警與服務目標、憑證核准路徑，也沒有產物與 run 記錄的保留／刪除政策。這些項目會影響排程變更、正式發布、事故升級、存取申請與資料清理；確認前，只做低風險的本機查閱與計畫檢查，不啟動或重啟正式排程、不重送正式發布，也不刪除 output 或 logs。請由目前的服務 owner 指定權威 runbook、存取申請途徑和保存政策；repo 沒有提供可核實的團隊名稱或聯絡方式。

| 覆蓋面 | 本次文件如何處理 |
| --- | --- |
| 系統用途、讀者與影響 | 依角色生成與人工審查流程說明；讀者角色與業務影響明確標為推論，因 repo 未定義正式角色或 SLA。 |
| Domain 規則、路由與產物 | 以 workflow 類型、ComfyUI workflow、角色設定、審查狀態和輸出檔為例，見[核心流程](workflow-and-architecture.md)。 |
| 架構、外部邊界與排程 | 以責任圖呈現；實際部署和正在運作的環境待確認。 |
| 第一天上手、觀察與修改 | 提供不生成的第一步、run 證據路徑和變更影響鏈，見[操作頁](operate-and-change.md)。 |
| 安全、隱私與資料生命週期 | 連到已提交設定範例並指出日誌可能含 prompt／LLM 訊息；權限與保存期限待 owner 確認。 |
| 正式環境復原、服務責任與走讀 | 正式 runbook／owner 未在 repo 中指明；未安排未參與撰寫者走讀，故正式值班交接尚未驗收。 |

**維護責任：** 由 repo 維護者隨路由、審查契約或產物格式變更更新本指南；實際負責人尚未在 repo 註明。

**證據界線：** 程式碼支援此版本的行為說明；設定範例不證明線上服務狀態；測試只涵蓋其明確執行的契約。這次未執行程式測試或外部服務驗收。

## 延伸閱讀

- [核心流程與系統責任](workflow-and-architecture.md)：讀懂角色、策略、審查與發布如何接在一起。
- [上手、觀察與安全變更](operate-and-change.md)：本機起步、run 診斷、輸出位置、變更邊界和待確認的正式環境資訊。
- [專題文件入口](../../README.md#架構與維護)：H3、Story Card、Motion Comic、風格、模型與特定實驗契約。
