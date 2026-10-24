# MediaOverload 連載短片：操作、審核與 MySQL

連載模式以固定角色逐集製作短片。故事計畫、段落、H3 render、QA、人工接受、參考影格、成片與各平台發布結果都存進 MySQL 資料庫 mediaoverload_serial。角色參考圖與影片仍是輸出目錄中的檔案；MySQL 保存路徑、SHA-256 和流程紀錄。

## 目前可操作的流程

新集生成完成且最終影片技術 QA 通過後，MediaOverload 會把「第幾集、主題、發布文案、Instagram／Facebook／YouTube」送到 Discord。只有 Discord 接受後，流程才會 dispatch 到三個平台。每個平台的結果、錯誤與 provider receipt 都記到 episode_publications。拒絕、逾時、Discord 無法審核或發布失敗，都會留下可辨識的狀態；技術 QA 通過不代表影片已發布。

連載策略整合在既有 `run_media_interface.py` 入口中，不需要另外執行 PowerShell 腳本，也沒有新增瀏覽器頁面。角色 YAML 的 `generation.serial_video` 控制預設片長、連跑集數、風格、輸出目錄、Discord 審核逾時與固定身份參考圖；啟用的平台和發布設定沿用同一份角色 YAML 的 `social_media.platforms`。

~~~mermaid
flowchart LR
    Creator[創作者提供本集故事與主題] --> Plan[規劃段落與總片長]
    Plan --> MySQL[(mediaoverload_serial)]
    Plan --> H3[ComfyUI / H3 生成段落]
    H3 --> SegmentQA[段落技術 QA]
    SegmentQA --> SegmentReview[Discord 接受每段]
    SegmentReview -->|接受| Tail[抽取已接受影片尾幀]
    Tail --> Next[同集下一段 I2VA]
    Tail --> MySQL
    Next --> SegmentQA
    SegmentReview -->|全部接受| Assemble[組裝、後製、最終 QA]
    Assemble --> FinalReview[Discord 最終發布審核]
    FinalReview -->|接受| Dispatch[發布至 Instagram / Facebook / YouTube]
    Dispatch --> MySQL
    FinalReview -->|拒絕、逾時或無法審核| Pending[保留成片，不發布]
~~~

第 1 段使用前集已接受的集尾影格和固定角色參考圖做 Ref2VA，同集後續段使用已接受片段的尾幀做 I2VA。每段都要經過技術 QA 和 Discord 接受，才能作為下一段的參考或進入最終組裝。

最終發布審核卡會列出集數、主題和三個發布平台，也能先編輯文案再決定。發布前會重新補上「第 N 集｜主題：…」，因此編輯文案不會移除集數識別。一般 Accept 會發布並讓系列繼續；「完成系列並發布本集」會發布本集並把系列標為 completed；「廢棄系列」會標為 discarded，並拒絕發布本集。新集生成後不按終止選項，系列預設仍為 active。

## MySQL 資料表

執行時只使用 mediaoverload_serial，不使用 SQLite。資料表會由 SerialVideoStore 建立；應用程式使用 mediaoverload_serial_app 帳號，不需要 root 權限。media_overload.env 要包含以下設定，實際密碼只放在被 Git 忽略的環境檔：

~~~dotenv
serial_video_mysql_database=mediaoverload_serial
serial_video_mysql_user=mediaoverload_serial_app
serial_video_mysql_password=<your-generated-password>
~~~

正式資料庫共有八張表：

| 資料表 | 一列代表什麼 |
|---|---|
| series | 固定角色的系列與目前故事狀態 |
| episodes | 一集的一個版本，含主題、總片長、前集、狀態、最終影片路徑與 SHA-256 |
| episode_plans | 綁定該集的執行計畫及其 SHA-256 |
| segments | 有序故事段落、預計片長、目前與已接受的 render、尾幀狀態 |
| render_attempts | 每次 H3 生成、條件輸入、輸出檔案與 SHA-256 |
| reviews | 片段技術 QA 與人工決定，綁定受審影片 SHA-256 |
| reference_assets | 角色參考圖、續段尾幀、集尾影格與來源 render |
| episode_publications | 每次發布嘗試的每個平台列，含審核決定、文案、狀態、錯誤與 receipt |

不同集數共用相同資料表，透過 episode_id、segment_id、render_id 關聯；不會為每集建立新表。舊試作的 SQLite 檔案保留為歷史輸出，執行入口和後續寫入都改用 MySQL。既有試作資料已匯入 MySQL：series 1 列、episodes 4 列、episode_plans 3 列、segments 8 列、render_attempts 5 列、reviews 5 列、reference_assets 8 列；episode_publications 會從新的發布審核開始累積。

`series.lifecycle_status` 儲存 `active`、`completed` 或 `discarded`，另有 `lifecycle_note` 和 `lifecycle_updated_at` 記錄創作者決定。自動續集只查詢 active 系列，且要求該系列最新集為 ready。

從 repo 根目錄檢查資料表與每集發布狀態：

~~~powershell
@'
from dotenv import load_dotenv
import os
import pymysql
from pymysql.cursors import DictCursor

load_dotenv("media_overload.env")
connection = pymysql.connect(
    host=os.environ["mysql_host"],
    port=int(os.environ.get("mysql_port", "3306")),
    user=os.environ["serial_video_mysql_user"],
    password=os.environ["serial_video_mysql_password"],
    database=os.environ["serial_video_mysql_database"],
    charset="utf8mb4",
    cursorclass=DictCursor,
)
try:
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        print("Tables:", [next(iter(row.values())) for row in cursor.fetchall()])
        cursor.execute(
            """SELECT episode_number, version, episode_topic, target_duration_seconds,
                      status, final_video_path, final_video_sha256
               FROM episodes WHERE series_id=%s ORDER BY episode_number, version""",
            ("kirby-continuity-20261024",),
        )
        for row in cursor.fetchall():
            print(row)
        cursor.execute(
            """SELECT e.episode_number, p.attempt_number, p.platform, p.status,
                      p.review_decision, p.error, p.receipt_json
               FROM episode_publications p JOIN episodes e ON e.episode_id=p.episode_id
               WHERE e.series_id=%s ORDER BY e.episode_number, p.attempt_number, p.platform""",
            ("kirby-continuity-20261024",),
        )
        print("Publication rows:", cursor.fetchall())
finally:
    connection.close()
'@ | python -X utf8 -
~~~

## 片長與 H3 段落

--duration-seconds 是整集目標片長，預設 30 秒。連載 H3 以約 10 秒為一段；30 秒會規劃 3 段，60 秒會規劃 6 段。每多一段，就多一次 H3 生成、技術 QA、Discord 人工接受與尾幀銜接。第 2 集試作當時明確使用 --duration-seconds 10，所以它只有一段；要更長就提高這個總秒數，不需要更改資料表或程式常數。

例如 60 秒：

~~~powershell
python -m agentic.app.main --goal "Kirby guides a lost firefly home with the star flower." --episode-topic "星花引路，陪迷路火螢回家" --media-type long_video --duration-seconds 60 --style "watercolor storybook animation" --serial-mode --series-id kirby-continuity-20261024 --character Kirby --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

## 從既有入口自動產生連載

在角色 YAML 啟用 `generation.serial_video.enabled` 後，直接選擇既有入口的策略：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video
~~~

Kirby 預設每次跑一集、每集 30 秒。想一次連跑五集、每集 60 秒：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video --serial-episodes 5 --duration-seconds 60
~~~

也可以在 `configs/characters/kirby.yaml` 的 `generation.serial_video` 設定預設值：

~~~yaml
serial_video:
  enabled: true
  default_duration_seconds: 30
  episodes_per_run: 1
  style: watercolor storybook animation
  output_dir: output
  discord_review_timeout_seconds: 1800
  identity_references: []
~~~

這個角色設定會套用到每個片段審核和最終發布審核；設定值優先於命令列值。未設定時使用 1800 秒預設值。

每集送 Discord 審核和發布完成後，才會開始下一集。自動策略先查找最新的失敗集並從該集恢復：已接受的片段會重用，第一段就失敗時則從第一段重做；如果 Discord 審核逾時或失敗，這次連跑立即停止，資料庫把該集標為 blocked。下次啟動會先重試這集，不會先抽新系列或下一集。沒有待重試集時，才以 50/50 選擇新系列或從 active 系列最新 ready 集續集。completed 和 discarded 系列不會被當成新續集候選；completed 系列若有尚未完成的發布失敗，允許重試該集。

使用 `--prompt` 和 `--episode-topic` 可指定這次連跑的第一集；後續集數仍由自動故事規劃產生。CLI 的 `--duration-seconds` 會覆蓋 YAML 預設，片長每增加約 10 秒就會多一次 H3 render、技術 QA、Discord 片段審核和尾幀銜接。第一次建立系列前，請在 YAML 的 `identity_references` 設定固定身份參考圖；已有角色身份圖登記在 MySQL 時可以留空。

## 產生新集

第一次建立系列要傳固定角色參考圖；後續集數使用相同系列 ID，不必重傳。每次新集都要提供清楚的 --episode-topic，主題會出現在 Discord 發布卡與實際發布文案：

~~~powershell
$env:PYTHONPATH = "agentic/src"
python -m agentic.app.main --goal "Kirby finds a fallen star seed in a quiet garden, plants it, and protects it as a tiny star flower blooms." --episode-topic "星種發芽，守護花園裡的星花" --media-type long_video --duration-seconds 30 --style "watercolor storybook animation" --serial-mode --series-id kirby-continuity-20261024 --character Kirby --serial-character-reference "D:\ComfyUI_windows_portable\ComfyUI\input\krea2_kirby_identity.png" --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

角色設定中啟用的社群平台預設為 Instagram、Facebook、YouTube。若要改目標，可重複傳 --platform，但指定的平台仍必須在角色 YAML 設為啟用。

## 把已完成的第 2 集送進最終發布審核

已完成的影片不用重跑生成。以下命令會確認 MySQL 中第 2 集是 ready、故事前提相符、影片 SHA 路徑有效，再把它送到 Discord 最終發布審核。接受後才會發布到三個啟用平台：

~~~powershell
python -m agentic.app.main --goal "At night, Kirby follows the star flower's glow and guides a lost firefly across a moonlit stepping-stone path to its home in an old hollow tree." --media-type long_video --serial-mode --series-id kirby-continuity-20261024 --character Kirby --serial-publish-episode-number 2 --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

Discord 審核卡的文案會以「第 2 集｜主題：星花引路，陪迷路火螢回家」開頭，並列出發布平台。按接受後才呼叫發布工具。發布結果會逐平台寫入 episode_publications；工具回報成功和 provider receipt 是發布證據，但仍不等同於確認所有平台公開可見。

## 審核逾時後恢復

逾時時，系統不會自動接受素材或發布。先看實際 MP4，再用確切受審檔案記錄片段決定；之後以原始故事描述和 --serial-resume-episode-number 恢復。保存的已接受片段與計畫會重用，只有尚未完成的工作才會繼續執行；續跑時 Discord review timeout 以目前 character config 為準。

~~~powershell
python -m agentic.app.main --goal "At night, Kirby follows the star flower's glow and guides a lost firefly across a moonlit stepping-stone path to its home in an old hollow tree." --media-type long_video --duration-seconds 30 --style "watercolor storybook animation" --serial-mode --series-id kirby-continuity-20261024 --character Kirby --serial-resume-episode-number 2 --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

## 程式位置與驗證邊界

- CLI、MySQL 連線、單集最終發布入口：agentic/src/agentic/app/main.py
- MySQL schema、SHA 驗證、片段狀態和逐平台發布紀錄：agentic/src/agentic/runtime/serial_video.py
- 長片段規劃、最終 QA 與 Discord 發布 gate：agentic/src/agentic/runtime/planning/long_video.py
- 審核卡文案、按下接受後的三平台 dispatch：agentic/src/agentic/skills/agent_social.py

本機 MySQL 契約測試能驗證資料表、片長分段、審核狀態和 receipt 儲存。實際 H3 畫面品質需要逐段看影片；真實 Discord 按鈕、Instagram／Facebook／YouTube 的外部服務結果，需以該次 run 的 Discord session、發布 manifest 和每平台 provider receipt 驗證。
