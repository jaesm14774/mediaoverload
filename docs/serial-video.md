# MediaOverload 連載短片：操作、審核與 MySQL

連載模式以固定角色逐集製作短片。故事計畫、段落、H3 render、QA、人工接受、參考影格、成片與各平台發布結果都存進 MySQL 資料庫 mediaoverload_serial。角色參考圖與影片仍是輸出目錄中的檔案；MySQL 保存路徑、SHA-256 和流程紀錄。

## Story-first 連載流程

新系列沿用命令列選定的角色作為主角。連載是一部會一直演下去的連續劇，不是有結局的單篇故事：

- **長期懸念**：整部劇有一個單集內解不開的問題或心願，要到第 7 集（最後一集）才會被收掉。
- **每集 30 到 40 秒**：每集是一個看得懂的完整小故事，有自己的小目標、麻煩和小結果，同時把長期懸念推進一小步。
- **鉤子**：每集結尾留一個尚未回答的小問題或意外畫面，讓人想看下一集；單集內不得完成主角的最終目標。
- **每集是上一集的後果**：故事只寫開頭幾集的場景和動作，最後一集由故事規劃收尾。每一集都要用「因此」或「但是」接上上一集交出的目標，並留下一個看得見的改變（關係、知情內容、持有物或地點）。下一集的提案收到的是已播出集數的摘要與交給下一集的目標、系列還欠觀眾的線索，而不是要它換題材，所以續集是接下去演，不是重演。詳見下方「連續性：線索帳本與常駐角色」。
- **新聞只是入口**：只借一種情緒或小難題，不使用新聞中的國家、地名、機構或真實人物。

預設 `simple` 複雜度保持單集簡單；想要較大的事件規模與角色成長時選 `cinematic`，但單集仍是 40 秒內的完整小故事。系列最多 7 集（每集至多 40 秒，全系列不超過五分鐘）；`episodes_per_run` 只限制單次執行製作幾集。第 7 集由故事規劃收掉主線，創作者接受後系列自動標為 completed；中途想提早結束，仍可在最終審核按「完成系列並發布本集」。

從 repo 根目錄啟動完整流程：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video
~~~

也可以選擇複雜度並提供故事方向：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video --story-complexity cinematic --prompt "Kirby helps a lost star find its way home."
~~~

故事、角色卡與參考圖、常駐角色身份圖（故事有常駐角色時）、首集大綱都會送到 Discord 審核。按 Accept 後，同一個執行流程自動接續下一階段，最後進入既有影片生成、片段審核和發布流程。拒絕、逾時或 Discord 無法審核時，流程會停在該關卡，不會在本機要求手動核准，也不會略過審核。

每個系列審核——故事、角色卡與參考圖、常駐角色身份圖、各集大綱、每個影片片段和最終審核——都有「廢棄系列」按鈕，不必等到最後一段才能放棄。片段審核只按 Reject 時，該集仍是 blocked，下次啟動會恢復它；按「廢棄系列」則把系列標為 discarded，之後不再被恢復或選為續集。尚未開始製作的新系列在故事、角色或首集大綱階段廢棄時，草稿記為拒絕並停止，不會建立製作中的系列。

所有影片都由同一個 FFmpeg 剪輯階段分析成片中的畫面變化，找出相對安靜與動作較多的時間區間，再分別調整速度並同步處理音訊。連載影片也在各段審核接受、組裝完成後才做這次分析，不需要在故事計畫或角色 YAML 指定速度；慢速區間會補插影格，避免只拉長播放時間造成卡頓。成片長度會依內容節奏變化。

`--story-complexity` 可選 `simple` 或 `cinematic`；未指定時讀取 `generation.serial_video.story_complexity`，預設為 `simple`。`--prompt` 提供故事方向；若未指定，隨機新聞只提供一個小型虛構靈感，不會成為故事主線的事件清單。故事、角色和分集大綱的核准內容及 SHA-256 會保存到 MySQL。

角色參考圖會依核准角色卡自動生成並在 Discord 供創作者挑選；若 YAML 已設定 `identity_references`，則將它們列為候選圖。沒有可用 Discord 審核或圖片生成失敗時，流程會回報阻塞原因，不會用未核准的角色參考圖開始影片生成。

## 連續性：線索帳本與常駐角色

續集要像電視劇，靠的是下面三個機制。三者都是規劃與審核階段的文字與資料，不會附加到 H3 或圖片生成提示；也沒有任何 LLM 自評或自動淘汰，最後仍由創作者在 Discord 決定。

**線索帳本**：每集大綱必須有一段「線索帳本」，只能寫以下條列，程式逐行解析，格式不符會被退回（草稿自動重寫最多三次，三次都不合會回報 failed 並附原因；創作者在 Discord 核准時也會再驗一次）。寬鬆處理的只有：分隔符號可用「：」、「無」或「沒有」的空項目會略過、主角名字寫進「出場」會被忽略：

~~~text
- 新增 主線｜觀眾想知道答案的長期問題（第一集必寫，之後系列永遠要有一條開放中的主線）
- 新增｜本集埋下的新線索
- 推進 T1｜這條開放中的線索本集往前走了什麼
- 回收 T2｜本集揭曉的線索（每集最多一條；主線不能由單集回收，只有創作者按「完成系列」才結束）
- 出場｜第一個節拍就在畫面上的常駐角色，用「、」分隔
- 下集目標｜這集結尾留給下一集的未完成目標（必須剛好一行）
~~~

驗證規則：第 N 集（N>1）的大綱要等第 N-1 集在最終審核被接受、帳本記錄之後才能通過；有開放中的線索時，本集至少要推進或回收其中一條；任何開放線索超過三集沒被碰到，下一集必須碰它；同時開放的線索最多六條；引用不存在或已回收的編號、或「出場」寫了名單以外的角色，都會被退回並列出可用的編號與名單。線索編號由程式依順序產生（T1、T2…），不由模型決定；本集新增的線索在本集被接受前還沒有編號，所以第一集的大綱只能「新增」，不能「推進」或「回收」，之後的集數也不能在新增它的同一集引用它。帳本被退回時，只會請模型重寫帳本段落，已寫好的故事保持原樣；上一集尚未被接受這類重寫也無法解決的原因，會直接停止。送進影片規劃的只有故事與節拍，結尾狀態卡和線索帳本留給創作者審核，不會進入影片提示詞。

**帳本何時生效**：只有在創作者於最終審核按 Accept 時，才把該集的帳本變動記入 `series_episode_ledger`。沒被接受的集、被重做的集，都不會留下線索；系列的帳本永遠是目前有效各集變動的總和。下一集的提案會收到帳本（含到期警告）、逐集摘要（「第 N 集《主題》：摘要｜交給下一集：目標」，不截斷）和常駐角色名單。

**常駐角色**：每個系列最多兩位常駐角色。新系列在主角角色卡核准後，由故事提出常駐角色名單（可以是零位），每位依系列畫風生成候選圖，由創作者在 Discord 各挑一張，成為固定身份參考（`series_companions`）。主角的角色卡或系列故事重新核准時，既有常駐角色標為過期。只有大綱「出場」列出的角色，其身份參考才會加入該集第 1 段的 Ref2VA 參考圖組。

**第 1 段參考圖組**：固定順序為前集集尾影格、主角身份參考（依路徑排序）、常駐角色身份參考（依路徑排序），總數上限 4 張。主角身份參考加常駐角色超過上限時，登記常駐角色會被拒絕。後續段落仍從上一段已接受的尾幀做 I2VA，所以常駐角色的身份參考只在第 1 段進入影片生成。

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

`--character-config` 應指向你要使用的影片、Discord 審核與平台發布設定檔。`generation.serial_video.series_selection_weights` 控制自動策略在新系列與續接系列間的比例，預設各為 `0.5`。若沒有符合條件的舊系列，系統一律開始新系列故事草稿，不受權重影響；草稿會停在人工審核，不會直接產生影片。故事核准後再依序建立並審核主角身分卡與參考圖、常駐角色身份圖（拒絕任何一位會讓整個新系列在該關卡停止，不會跳過）、分集劇情／場景／鏡頭分鏡，全部核准後才進入影片生成。故事、角色或分鏡改版時，舊角色參考與受影響集數會標記 stale，必須核准新版角色與分鏡後再生成。

故事、角色卡、分集分鏡與每集實際 lineage 會分別記錄在 `series_story_plans`、`series_character_cards`、`series_episode_outlines` 和 `episode_onboarding_lineage`。新建過程不會建立可生成影片的 production `series`；第一集開始時才建立。

## 目前可操作的流程

新集生成完成且最終影片技術 QA 通過後，既有 Discord 最終審核會附上完整成片、公開文案，以及上一集核准狀態和本集提出的結尾狀態。創作者直接判斷整集故事、節奏、角色與道具銜接；不使用 LLM 故事品質評分或額外語意 QA。Accept 會將本集結尾狀態連同核准者、審核 session 和成片 SHA 寫入系列狀態，並核准公開文案進入發布流程。集數、主題與工作流資訊不會加進公開文案。各平台的結果、錯誤與 provider receipt 都記到 episode_publications。拒絕、逾時或 Discord 無法審核時，系列狀態不前進且不發布；技術 QA 通過只代表影片技術檢查完成。

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

第 1 段使用前集已接受的集尾影格、固定角色參考圖和出場常駐角色的身份參考圖（共最多 4 張）做 Ref2VA，同集後續段使用已接受片段的尾幀做 I2VA。每段都要經過技術 QA 和 Discord 接受，才能作為下一段的參考或進入最終組裝。

最終審核將成片與故事連續狀態放在內部審核內容，公開文案仍只含核准的 caption 和 hashtags。一般 Accept 會核准狀態並發布，系列保持 active；「完成系列並發布本集」會發布本集並把系列標為 completed；「廢棄系列」會標為 discarded，並拒絕發布本集。「完成系列並發布本集」只在最終審核提供；「廢棄系列」在每個審核都提供。新集生成後不按終止選項，系列預設仍為 active。影片技術檢查和整集創作判斷分開：前者只處理成片技術狀態，後者完全由創作者決定。

## MySQL 資料表

執行時只使用 mediaoverload_serial。資料表會由 SerialVideoStore 建立；應用程式使用 mediaoverload_serial_app 帳號，不需要 root 權限。media_overload.env 要包含以下設定，實際密碼只放在被 Git 忽略的環境檔：

~~~dotenv
serial_video_mysql_database=mediaoverload_serial
serial_video_mysql_user=mediaoverload_serial_app
serial_video_mysql_password=<your-generated-password>
~~~

主要資料表如下：

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
| series_episode_ledger | 創作者在最終審核接受的一集所帶來的線索帳本變動、摘要與交給下一集的目標；重做該集會取代舊列 |
| series_companions | 系列的常駐角色（最多兩位）、外觀描述與創作者挑選的身份參考圖 SHA-256 |

不同集數共用相同資料表，透過 episode_id、segment_id、render_id 關聯；不會為每集建立新表。

`series.lifecycle_status` 儲存 `active`、`completed` 或 `discarded`，另有 `lifecycle_note` 和 `lifecycle_updated_at` 記錄創作者決定。自動續集只查詢 active 系列，且要求該系列最新集為 ready 並帶有目前格式的人審核准狀態；最新集為 blocked、或被中斷而停在 rendering／planned 的系列，則先走上述恢復路徑。若最新集在計畫存檔前就失敗或被中斷（沒有可恢復的計畫與段落），自動選擇開始時會把該 active 系列標為 discarded；仍有寫入的新集不受影響。

### 清理已結束系列的本機媒體

可手動預覽並清理已標記為 `completed` 或 `discarded` 的系列媒體。從 repo 根目錄執行：

~~~powershell
python scripts/cleanup_media.py
~~~

預設只列出影片、片段、影格、參考圖片和音訊檔，不會刪除。確認清單後加上 `--delete` 才會刪除；故事、角色卡、分鏡等文字檔和 MySQL 紀錄會保留。預設媒體根目錄是 serial CLI 的 `agentic/output/`，自訂輸出目錄可用 `--output-dir` 指定，但必須位於 repo 內。

`discarded` 系列可直接清理。`completed` 系列若仍有可恢復的 blocked 集數，或 ready 集數沒有每個已記錄平台的成功發布 receipt，腳本會保留整個系列的媒體，讓發布重試仍可使用原檔。`active` 系列永遠不會列入清理；若結束系列媒體也被其他系列的資料庫紀錄引用，也會保留該檔。

同一支腳本也會清理其他策略的已發布輸出：根目錄（預設 `output/` 與 `agentic/output/`，可重複使用 `--output-dir`）下，若某個角色資料夾的 `publish_manifest.json` 顯示 `publication_state` 為 `published`，該資料夾內建立與修改時間都不晚於發布時間的媒體檔（含 `publish_ready/`，也包含發布前就產生但從未發布的舊輪次）都會列入清單；發布之後才產生的新一輪檔案、文字檔與 manifest 本身會保留。`awaiting_user_action`、`partially_published` 等未完全公開的狀態一律保留，`serial_series/` 與 `serial_publication_retries/` 仍由上述資料庫規則決定。

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

--duration-seconds 是整集目標片長，預設 30 秒，上限 40 秒（超過會在啟動時被拒絕）。連載是短短劇，約為真正電視劇的十分之一到二十分之一：每個片段固定 10 秒、只演一拍，所以 30 秒是 3 段、40 秒是 4 段。系列最多 7 集，全系列不超過五分鐘，最後一集要把主線收掉。片段越少，接力次數越少，角色跑位累積得越少。每多一段，就多一次 H3 生成、技術 QA、Discord 人工接受與尾幀銜接。節拍數等於片段數：本集大綱每拍約 40 字、只寫鏡頭拍得到的動作，分段編劇逐段只演自己的那一拍，最後一段演完小結果後接上鉤子（最後一集改為收尾）。每段的動作寫成三個分秒節拍（0.00-3.00 s、3.00-6.50 s、6.50-10.00 s，約 60 到 90 字），主要追逐或大動作在約 4 秒前開始，需要被看見的變化寫成具體畫面；鏡頭由編劇依節拍規劃（追逐用貼地橫向跟拍、情緒轉折用臉部特寫、揭露可用最多三個預先規劃的切換，不用推近或變焦），沒有鏡頭計畫時才退回單一固定鏡頭。量測與限制見 `docs/minimax_h3_prompting.md`。

## 系列畫風鎖定

每個系列第一次需要畫布與插畫風格時，由系列 ID 作種子，從角色 `generation.creative_profile.style.model` 指定模型的 `configs/styles/<model>.yaml` 的 `enabled` 風格抽一個（畫布取自 `generation.creative_profile.canvas`），並把結果存成該系列資料夾的 `look.json`。之後所有角色參考圖和每一集都讀這份存檔，不再重抽；之後在挑選頁增刪風格或改畫布權重，都不影響已有系列。串連載命令列的 `--style`、`--width`、`--height` 在連載路徑上會被系列畫風覆蓋。直接執行 `python -m agentic.app.main` 時，未傳 `--duration-seconds` 的預設同為 30 秒，時長必須是 10、20、30 或 40 秒（整數個 10 秒片段，最長 40 秒）。

模型沒有對應的 `configs/styles/<model>.yaml`、`enabled` 列出不存在的風格 ID、或缺少畫布時，啟動連載的設定檢查就會報錯，不會等到故事與角色卡審核之後。

同集內的跑位仍未解決。2026-11-13 在本機 ComfyUI 用 Klein 在每段之間重新對齊尾幀並實測 I2VA：整張重畫成水彩能穩住單段內的畫風，但把夜晚改成白天，接縫會跳；改成只修角色，接縫不跳，卻會把背對鏡頭、抱著信的 Kirby 改成正面並弄丟信；降低 denoise 到 0.4 則修不動。再試過把系列角色參考圖當第二張參考圖一起餵給 Klein：在夜晚賽璐珞、Kirby 較小的尾幀上，5 個 seed 都保住接縫與信，接著的 I2VA 也演出劇本動作，單段內畫風不翻轉；但在水彩場景、Kirby 佔滿畫面的尾幀上，5 個 seed 全部把 Kirby 畫成白底黑星花紋，影片也照著延續這個錯誤。結果取決於尾幀內容，失敗時沒有可靠的偵測方式（不用 LLM 品質關卡），所以流程仍然沒有加入這個步驟。

角色參考圖走專用路徑：六張候選使用同一個提示（含系列風格），不經 LLM 構圖變體，只差在雜訊，角色置中、完整入鏡、約佔畫面一半高並四周留白。置中是為了避免參考圖出現縮在角落或不同風格的版本，進而被 Ref2VA 帶進影片；只佔一半高，是因為圓球身體在 1:1、4:5 畫布上佔三分之二高就會幾乎塞滿畫面，影片首段會承襲這個大小。

參考圖不是通用的官方設定圖。角色卡要依故事設計這個系列專屬的造型（來自故事的世界與處境，跨集固定），並把主角性格演在預設神情上：眼神有焦點、看向某個對象，眉眼、嘴角、重心都在說牠想要什麼。提示詞不再要求「中性表情」或「不加任何配件」，那兩條會讓指定卡比的每個系列都畫成同一隻置中、超大、呆望鏡頭的標準卡比。圖片生成器只讀角色卡「角色參考圖提示：」之後的單一段落，所以外觀錨點、專屬造型與預設神情都要寫在這一段；取景、視角、背景與大小由系統決定，卡片不寫。預設神情會隨參考圖進入每集首段，各集再依劇情改變表情與動作。已鎖定的舊系列仍沿用舊參考圖，需另行決定是否重做。

例如 30 秒：

~~~powershell
python -m agentic.app.main --goal "Kirby guides a lost firefly home with the star flower." --episode-topic "星花引路，陪迷路火螢回家" --media-type long_video --duration-seconds 30 --serial-mode --series-id kirby-continuity-20261024 --character Kirby --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

## 從既有入口自動產生連載

在角色 YAML 啟用 `generation.serial_video.enabled` 後，可明確指定連載策略：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video
~~~

不指定 `--generation-type` 時，入口會依角色 YAML 的 `generation.generation_type_weights` 隨機選策略。Kirby 的 `serial_video` 權重為 3；若抽中連載，會直接交給同一個連載 runner，執行故事、角色與首集大綱的 Discord 審核，再接續影片流程。審核遭拒、逾時或生成失敗時，入口會回報 blocked／failed，不會假裝整個流程成功。

預設 `simple` 是連載短劇模式：全系列最多 7 集，第 7 集收掉主線，創作者也可提早按「完成系列並發布本集」收尾；每集是一件小事、最多兩位常駐角色和簡單結果，單集最多一個小反轉；分集路線以 7 集為上限，且會把人審核准的狀態和線索帳本帶到下一集。要提高故事層次時可加 `--story-complexity cinematic`；也能在角色 YAML 的 `generation.serial_video.story_complexity` 設定預設值。`episodes_per_run` 控制一次生成的集數，和故事規劃集數分開。

Kirby 預設每次跑一集、每集 30 秒。想一次連跑三集、每集 40 秒（上限）：

~~~powershell
python run_media_interface.py --character kirby --generation-type serial_video --serial-episodes 3 --duration-seconds 40
~~~

也可以在 `configs/characters/kirby.yaml` 的 `generation.serial_video` 設定預設值：

~~~yaml
serial_video:
  enabled: true
  default_duration_seconds: 30
  episodes_per_run: 1
  story_complexity: simple
  output_dir: output
  discord_review_timeout_seconds: 1800
  identity_references: []
~~~

這個角色設定會套用到每個片段審核和最終發布審核；設定值優先於命令列值。未設定時使用 1800 秒預設值。

每集送 Discord 審核、核准狀態並完成發布流程後，才會開始下一集。自動策略先查找目前格式的最新未完成集並從該集恢復：已接受的片段會重用，第一段就失敗時則從第一段重做。若片段審核逾時或失敗，這次連跑立即停止，資料庫把該集標為 blocked。若執行被強制中斷（程式被關閉、當機、斷電），該集會停在 rendering 或 planned；超過 2 小時沒有任何寫入就視為被中斷，和 blocked 一樣被恢復，不會變成永遠選不到的孤兒系列。2 小時內仍有寫入的 rendering／planned 集視為另一個執行中的 run，不會被重複生成。下次啟動會先恢復符合目前格式的待處理集，不會先抽新系列或下一集。沒有待重試集時，符合條件的舊系列會依 `generation.serial_video.series_selection_weights` 隨機選擇新系列或續集；若續集尚無下一集大綱，系統會先依完整故事路線和人審核准狀態產生大綱，等待 Discord 接受後才產片。若沒有符合條件的舊系列，系統會建立新系列並在同一次執行中等待 Discord 核准故事、角色與首集分鏡，再接續產片。權重預設 `new: 0.5`、`continue: 0.5`，可在角色 YAML 中調整。completed 和 discarded 系列不會被當成新續集候選；completed 系列若有尚未完成的發布失敗，允許重試該集。

使用 `--prompt` 可提供新系列故事草稿的想法；選到既有系列時，它會作為本集方向。後續集數仍由自動故事規劃產生。CLI 的 `--duration-seconds` 會覆蓋 YAML 預設，片長每增加 10 秒就會多一次 H3 render、技術 QA、Discord 片段審核和尾幀銜接。新系列的角色參考圖在故事核准後產生並由創作者挑選核准，無須預先設定固定身份圖。

每張首輪候選圖依角色卡「角色參考圖提示」段落（外觀錨點、專屬造型、預設神情）生成：單一角色、正面略帶四分之三角度、完整入鏡、置中、四周留白、素色背景，使用該系列鎖定的畫風。Discord 審核時對照輪廓、臉部或頭部特徵、配色、服裝或身體表面和標誌細節，只接受完全符合的一張；若候選圖都不符合就拒絕，避免把錯誤設計鎖成後續各集的身份參考。接受後，選定圖片的 SHA-256 會和角色卡一起保存。

## 產生新集

第一次建立系列要傳固定角色參考圖；後續集數使用相同系列 ID，不必重傳。手動建立新集時用 `--episode-topic` 提供故事規劃主題，或使用 `--serial-auto-next` 讓連載點子生成器產生主題。主題只供故事規劃與連載紀錄使用，不會自動加進公開文案：

~~~powershell
$env:PYTHONPATH = "agentic/src"
python -m agentic.app.main --goal "Kirby finds a fallen star seed in a quiet garden, plants it, and protects it as a tiny star flower blooms." --episode-topic "星種發芽，守護花園裡的星花" --media-type long_video --duration-seconds 30 --serial-mode --series-id kirby-continuity-20261024 --character Kirby --serial-character-reference "D:\ComfyUI_windows_portable\ComfyUI\input\krea2_kirby_identity.png" --output-dir "agentic/output/kirby_continuity_20261024" --execute
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
python -m agentic.app.main --goal "At night, Kirby follows the star flower's glow and guides a lost firefly across a moonlit stepping-stone path to its home in an old hollow tree." --media-type long_video --duration-seconds 30 --serial-mode --series-id kirby-continuity-20261024 --character Kirby --serial-resume-episode-number 2 --output-dir "agentic/output/kirby_continuity_20261024" --execute
~~~

## 程式位置與驗證邊界

- CLI、MySQL 連線、單集最終發布入口：agentic/src/agentic/app/main.py
- MySQL schema、SHA 驗證、片段狀態和逐平台發布紀錄：agentic/src/agentic/runtime/serial_video.py
- 線索帳本的解析、驗證與渲染：agentic/src/agentic/runtime/serial_ledger.py
- 分集大綱提示與逐集摘要：agentic/src/agentic/runtime/serial_episode_plan.py
- 常駐角色提案與身份圖提示：agentic/src/agentic/runtime/serial_character.py
- 長片段規劃、最終 QA 與 Discord 發布 gate：agentic/src/agentic/runtime/planning/long_video.py
- 審核卡文案、按下接受後的三平台 dispatch：agentic/src/agentic/skills/agent_social.py

本機 MySQL 契約測試能驗證資料表、片長分段、審核狀態和 receipt 儲存。實際 H3 畫面品質需要逐段看影片；真實 Discord 按鈕、Instagram／Facebook／YouTube 的外部服務結果，需以該次 run 的 Discord session、發布 manifest 和每平台 provider receipt 驗證。
