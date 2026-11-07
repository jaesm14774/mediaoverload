# MediaOverload 連載短片：操作、審核與 MySQL

連載模式以固定角色逐集製作短片。故事計畫、段落、H3 render、QA、人工接受、參考影格、成片與各平台發布結果都存進 MySQL 資料庫 mediaoverload_serial。角色參考圖與影片仍是輸出目錄中的檔案；MySQL 保存路徑、SHA-256 和流程紀錄。

## Story-first 連載流程

新系列沿用命令列選定的角色作為主角。預設 `simple` 複雜度適合動畫連載短劇：整個故事有完整主線與結局，每集處理一件容易看懂的小事，並延續角色、線索、道具和場景狀態；分集數依故事自然篇幅規劃，不設三集上限。想寫多層因果、較大事件與角色弧線時，可選 `cinematic`。`episodes_per_run` 只限制單次執行製作幾集，不限制故事總集數。

從 repo 根目錄啟動完整流程：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video
~~~

也可以選擇複雜度並提供故事方向：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video --story-complexity cinematic --prompt "Kirby helps a lost star find its way home."
~~~

故事、角色卡與參考圖、首集大綱都會送到 Discord 審核。按 Accept 後，同一個執行流程自動接續下一階段，最後進入既有影片生成、片段審核和發布流程。拒絕、逾時或 Discord 無法審核時，流程會停在該關卡，不會在本機要求手動核准，也不會略過審核。

每個系列審核——故事、角色卡與參考圖、各集大綱、每個影片片段和最終審核——都有「廢棄系列」按鈕，不必等到最後一段才能放棄。片段審核只按 Reject 時，該集仍是 blocked，下次啟動會恢復它；按「廢棄系列」則把系列標為 discarded，之後不再被恢復或選為續集。尚未開始製作的新系列在故事、角色或首集大綱階段廢棄時，草稿記為拒絕並停止，不會建立製作中的系列。

所有影片都由同一個 FFmpeg 剪輯階段分析成片中的畫面變化，找出相對安靜與動作較多的時間區間，再分別調整速度並同步處理音訊。連載影片也在各段審核接受、組裝完成後才做這次分析，不需要在故事計畫或角色 YAML 指定速度；慢速區間會補插影格，避免只拉長播放時間造成卡頓。成片長度會依內容節奏變化。

`--story-complexity` 可選 `simple` 或 `cinematic`；未指定時讀取 `generation.serial_video.story_complexity`，預設為 `simple`。`--prompt` 提供故事方向；若未指定，隨機新聞只提供一個小型虛構靈感，不會成為故事主線的事件清單。故事、角色和分集大綱的核准內容及 SHA-256 會保存到 MySQL。

角色參考圖會依核准角色卡自動生成並在 Discord 供創作者挑選；若 YAML 已設定 `identity_references`，則將它們列為候選圖。沒有可用 Discord 審核或圖片生成失敗時，流程會回報阻塞原因，不會用未核准的角色參考圖開始影片生成。

### 單階段操作（維護用）

以下命令只執行單一 onboarding 階段；一般連載請使用上方的 `run_media_interface.py` 完整流程，讓人類審核留在 Discord 並在接受後繼續生成。預設會從現有 `news_ch.news` 新聞庫隨機挑選一則可追溯來源的文章當靈感；若使用自訂故事且明確不採新聞靈感，傳 `--serial-news-inspiration none`。

在 repo 根目錄建立故事草稿：

~~~powershell
$env:PYTHONPATH = "agentic/src"
python -m agentic.app.main `
  --serial-story-draft `
  --goal "一名燈塔守望者幫迷路的雲朵找回海洋。" `
  --output-dir "agentic/output/serial-series"
~~~

命令會回報 `series_id`、`story_plan_id` 和 Markdown 檔案路徑。先打開並編輯該檔案，確認完整故事、因果、基調與結局。若使用自訂故事且明確不採新聞靈感：

~~~powershell
python -m agentic.app.main `
  --serial-story-draft `
  --serial-news-inspiration none `
  --goal "一名燈塔守望者幫迷路的雲朵找回海洋。" `
  --output-dir "agentic/output/serial-series"
~~~

使用回報的 `story_plan_id` 與同一份 Markdown 檔核准；系統會把審核當下的文字、審核人和 SHA 存入 MySQL：

~~~powershell
python -m agentic.app.main `
  --serial-story-review-plan-id <story_plan_id> `
  --serial-story-review-decision accepted `
  --serial-story-reviewed-file "<story_markdown_path>"
~~~

拒絕草稿時不需要傳檔案，可加上原因：

~~~powershell
python -m agentic.app.main `
  --serial-story-review-plan-id <story_plan_id> `
  --serial-story-review-decision rejected `
  --serial-story-review-notes "主角的目標和結尾還不夠清楚。"
~~~

故事核准後，用剛才回報的 `series_id` 生角色卡。可以加角色偏好；不提供時會請 LLM 依核准故事創作主角：

~~~powershell
python -m agentic.app.main `
  --serial-character-draft `
  --series-id <series_id> `
  --serial-character-preferences "主角是一名勇敢但有點迷糊的海豹郵差。" `
  --output-dir "agentic/output/serial-series"
~~~

先讀取角色卡中的「角色參考圖提示」，用既有圖片流程生成候選圖，再由人挑選畫面。這會實際呼叫目前設定的 LLM 和 ComfyUI：

~~~powershell
python -m agentic.app.main `
  --goal "<貼上角色卡中的角色參考圖提示>" `
  --media-type image `
  --execute `
  --output-dir "agentic/output/serial-series"
~~~

編輯 Markdown 角色卡後，核准時同時指定最終角色名稱和挑選的參考圖。最多可選三張，核准後每集固定使用這些圖的 SHA-256：

~~~powershell
python -m agentic.app.main `
  --series-id <series_id> `
  --serial-character-card-review-id <character_card_id> `
  --serial-character-card-review-decision accepted `
  --serial-character-card-name "潮汐" `
  --serial-character-card-reviewed-file "<character_card_path>" `
  --serial-character-reference "<selected_identity_image.png>"
~~~

每集先依完整系列路線和上一集人審核准的狀態，產生故事節點、場景與鏡頭文字清單。提供集名／主題，`--goal` 可選填本集方向。簡單模式也會列出角色位置、身體與情緒狀態、角色知情、關係、線索、道具持有人／位置／狀況、場景變化和下集目標，讓下一集有明確起點：

~~~powershell
python -m agentic.app.main `
  --serial-episode-plan-draft `
  --series-id <series_id> `
  --serial-episode-number 1 `
  --episode-topic "燈塔失去訊號" `
  --goal "潮汐發現鹽晶體擋住燈罩，必須找出原因。" `
  --output-dir "agentic/output/serial-series"
~~~

審閱或修改分鏡 Markdown，再核准該集：

~~~powershell
python -m agentic.app.main `
  --serial-episode-plan-review-id <outline_id> `
  --serial-episode-plan-review-decision accepted `
  --serial-episode-plan-reviewed-file "<outline_path>"
~~~

故事、角色和該集分鏡都核准後，才開始生成。`--character-config` 指定現有的生成與發布設定檔；資料庫核准的角色名稱、身份卡和參考圖仍是此系列實際使用的角色資料。核准分鏡會自動作為本集影片故事內容，不必再複製到 `--goal`：

~~~powershell
python -m agentic.app.main `
  --media-type long_video `
  --serial-mode `
  --series-id <series_id> `
  --character-config "configs/characters/kirby.yaml" `
  --duration-seconds 30 `
  --execute
~~~

`--character-config` 應指向你要使用的影片、Discord 審核與平台發布設定檔。`generation.serial_video.series_selection_weights` 控制自動策略在新系列與續接系列間的比例，預設各為 `0.5`。若沒有符合條件的舊系列，系統一律開始新系列故事草稿，不受權重影響；草稿會停在人工審核，不會直接產生影片。故事核准後再依序建立並審核主角身分卡與參考圖、分集劇情／場景／鏡頭分鏡，全部核准後才進入影片生成。故事、角色或分鏡改版時，舊角色參考與受影響集數會標記 stale，必須核准新版角色與分鏡後再生成。

故事、角色卡、分集分鏡與每集實際 lineage 會分別記錄在 `series_story_plans`、`series_character_cards`、`series_episode_outlines` 和 `episode_onboarding_lineage`。新建過程不會建立可生成影片的 production `series`；第一集開始時才建立。

## 目前可操作的流程

新集生成完成且最終影片技術 QA 通過後，既有 Discord 最終審核會附上完整成片、公開文案，以及上一集核准狀態和本集提出的結尾狀態。創作者直接判斷整集故事、節奏、角色與道具銜接；不使用 LLM 故事品質評分或額外語意 QA。Accept 會將本集結尾狀態連同核准者、審核 session 和成片 SHA 寫入系列狀態，並核准公開文案進入發布流程。集數、主題與工作流資訊不會加進公開文案。各平台的結果、錯誤與 provider receipt 都記到 episode_publications。拒絕、逾時或 Discord 無法審核時，系列狀態不前進且不發布；技術 QA 通過只代表影片技術檢查完成。

更新前的系列若只有舊版技術 QA 寫入的 ending 摘要，不會被當作人審核准狀態自動續集；需從新系列開始。系統不會把舊摘要自動升級成連續狀態。

連載策略整合在既有 `run_media_interface.py` 入口中，不需要另外執行 PowerShell 腳本，也沒有新增瀏覽器頁面。角色 YAML 的 `generation.serial_video` 控制預設片長、連跑集數、風格、輸出目錄、故事複雜度、Discord 審核逾時與固定身份參考圖；啟用的平台和發布設定沿用同一份角色 YAML 的 `social_media.platforms`。

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

最終審核將成片與故事連續狀態放在內部審核內容，公開文案仍只含核准的 caption 和 hashtags。一般 Accept 會核准狀態並發布，系列保持 active；「完成系列並發布本集」會發布本集並把系列標為 completed；「廢棄系列」會標為 discarded，並拒絕發布本集。「完成系列並發布本集」只在最終審核提供；「廢棄系列」在每個審核都提供。新集生成後不按終止選項，系列預設仍為 active。影片技術檢查和整集創作判斷分開：前者只處理成片技術狀態，後者完全由創作者決定。

## MySQL 資料表

執行時只使用 mediaoverload_serial，不使用 SQLite。資料表會由 SerialVideoStore 建立；應用程式使用 mediaoverload_serial_app 帳號，不需要 root 權限。media_overload.env 要包含以下設定，實際密碼只放在被 Git 忽略的環境檔：

~~~dotenv
serial_video_mysql_database=mediaoverload_serial
serial_video_mysql_user=mediaoverload_serial_app
serial_video_mysql_password=<your-generated-password>
~~~

正式資料庫的影片執行帳本有八張表，另有一張故事草稿與審核版本表：

| 資料表 | 一列代表什麼 |
|---|---|
| series | 固定角色的系列與最近一次人審核准的連續狀態 |
| series_story_plans | 完整系列故事草稿、新聞靈感快照、人工決定與 SHA-256；角色核准前獨立於 production series |
| episodes | 一集的一個版本，含主題、總片長、前集、故事狀態提案與起始狀態、最終影片路徑與 SHA-256 |
| episode_plans | 綁定該集的執行計畫及其 SHA-256 |
| segments | 有序故事段落、預計片長、目前與已接受的 render、尾幀狀態 |
| render_attempts | 每次 H3 生成、條件輸入、輸出檔案與 SHA-256 |
| reviews | 片段技術 QA 與人工決定，綁定受審影片 SHA-256 |
| reference_assets | 角色參考圖、續段尾幀、集尾影格與來源 render |
| episode_publications | 每次發布嘗試的每個平台列，含審核決定、文案、狀態、錯誤與 receipt |

不同集數共用相同資料表，透過 episode_id、segment_id、render_id 關聯；不會為每集建立新表。舊試作的 SQLite 檔案保留為歷史輸出，執行入口和後續寫入都改用 MySQL。既有試作資料已匯入 MySQL：series 1 列、episodes 4 列、episode_plans 3 列、segments 8 列、render_attempts 5 列、reviews 5 列、reference_assets 8 列；episode_publications 會從新的發布審核開始累積。

`series.lifecycle_status` 儲存 `active`、`completed` 或 `discarded`，另有 `lifecycle_note` 和 `lifecycle_updated_at` 記錄創作者決定。自動續集只查詢 active 系列，且要求該系列最新集為 ready 並帶有目前格式的人審核准狀態；最新集為 blocked、或被中斷而停在 rendering／planned 的系列，則先走上述恢復路徑。更新前、只有舊版 ending 摘要的 active 系列兩條路都進不去，請在 Discord 廢棄。若最新集在計畫存檔前就失敗或被中斷（沒有可恢復的計畫與段落），自動選擇開始時會把該 active 系列標為 discarded；仍有寫入的新集不受影響。

### 清理已結束系列的本機媒體

可手動預覽並清理已標記為 `completed` 或 `discarded` 的系列媒體。從 repo 根目錄執行：

~~~powershell
python scripts/cleanup_serial_video_media.py
~~~

預設只列出影片、片段、影格、參考圖片和音訊檔，不會刪除。確認清單後加上 `--delete` 才會刪除；故事、角色卡、分鏡等文字檔和 MySQL 紀錄會保留。預設媒體根目錄是 serial CLI 的 `agentic/output/`，自訂輸出目錄可用 `--output-dir` 指定，但必須位於 repo 內。

`discarded` 系列可直接清理。`completed` 系列若仍有可恢復的 blocked 集數，或 ready 集數沒有每個已記錄平台的成功發布 receipt，腳本會保留整個系列的媒體，讓發布重試仍可使用原檔。`active` 系列永遠不會列入清理；若結束系列媒體也被其他系列的資料庫紀錄引用，也會保留該檔。

掛載點檢查支援 Windows 和 Linux；其他作業系統會安全停止清理，避免把未辨識的掛載樹當成系列檔案刪除。

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

在角色 YAML 啟用 `generation.serial_video.enabled` 後，可明確指定連載策略：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video
~~~

不指定 `--generation-type` 時，入口會依角色 YAML 的 `generation.generation_type_weights` 隨機選策略。Kirby 的 `serial_video` 權重為 40；若抽中連載，會直接交給同一個連載 runner，執行故事、角色與首集大綱的 Discord 審核，再接續影片流程。審核遭拒、逾時或生成失敗時，入口會回報 blocked／failed，不會假裝整個流程成功。

預設 `simple` 是連載短劇模式：全系列有完整結局，每集是一件小事、少量角色和簡單結果；分集路線沒有三集上限，且會把人審核准的狀態帶到下一集。要提高故事層次時可加 `--story-complexity cinematic`；也能在角色 YAML 的 `generation.serial_video.story_complexity` 設定預設值。`episodes_per_run` 控制一次生成的集數，和故事規劃集數分開。

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
  story_complexity: simple
  style: watercolor storybook animation
  output_dir: output
  discord_review_timeout_seconds: 1800
  identity_references: []
~~~

這個角色設定會套用到每個片段審核和最終發布審核；設定值優先於命令列值。未設定時使用 1800 秒預設值。

每集送 Discord 審核、核准狀態並完成發布流程後，才會開始下一集。自動策略先查找目前格式的最新未完成集並從該集恢復：已接受的片段會重用，第一段就失敗時則從第一段重做。若片段審核逾時或失敗，這次連跑立即停止，資料庫把該集標為 blocked。若執行被強制中斷（程式被關閉、當機、斷電），該集會停在 rendering 或 planned；超過 2 小時沒有任何寫入就視為被中斷，和 blocked 一樣被恢復，不會變成永遠選不到的孤兒系列。2 小時內仍有寫入的 rendering／planned 集視為另一個執行中的 run，不會被重複生成。下次啟動會先恢復符合目前格式的待處理集，不會先抽新系列或下一集。沒有待重試集時，符合條件的舊系列會依 `generation.serial_video.series_selection_weights` 隨機選擇新系列或續集；若續集尚無下一集大綱，系統會先依完整故事路線和人審核准狀態產生大綱，等待 Discord 接受後才產片。若沒有符合條件的舊系列，系統會建立新系列並在同一次執行中等待 Discord 核准故事、角色與首集分鏡，再接續產片。權重預設 `new: 0.5`、`continue: 0.5`，可在角色 YAML 中調整。completed 和 discarded 系列不會被當成新續集候選；completed 系列若有尚未完成的發布失敗，允許重試該集。

使用 `--prompt` 可提供新系列故事草稿的想法；選到既有系列時，它會作為本集方向。後續集數仍由自動故事規劃產生。CLI 的 `--duration-seconds` 會覆蓋 YAML 預設，片長每增加約 10 秒就會多一次 H3 render、技術 QA、Discord 片段審核和尾幀銜接。新系列的角色參考圖在故事核准後產生並由創作者挑選核准，無須預先設定固定身份圖。

每張首輪候選圖依角色卡的固定外觀錨點生成：單一角色、正面略帶四分之三角度、依角色形態完整入鏡、素色背景，沿用角色設定檔的美術風格。Discord 審核時對照輪廓、臉部或頭部特徵、配色、服裝或身體表面和標誌細節，只接受完全符合的一張；若候選圖都不符合就拒絕，避免把錯誤設計鎖成後續各集的身份參考。接受後，選定圖片的 SHA-256 會和角色卡一起保存。

## 產生新集

第一次建立系列要傳固定角色參考圖；後續集數使用相同系列 ID，不必重傳。手動建立新集時用 `--episode-topic` 提供故事規劃主題，或使用 `--serial-auto-next` 讓連載點子生成器產生主題。主題只供故事規劃與連載紀錄使用，不會自動加進公開文案：

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

Discord 審核卡顯示本集的發布文案與標籤。按接受後才呼叫發布工具。發布結果會逐平台寫入 episode_publications；工具回報成功和 provider receipt 是發布證據，但仍不等同於確認所有平台公開可見。

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
