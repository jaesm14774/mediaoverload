# MediaOverload

## CI 與本機 E2E 的分工

CI（Continuous Integration，持續整合）是每次 push 或 pull request 自動執行的基本品質防線。Hosted CI 執行可重現的 unit/contract tests 與 Python compile；需要真實 DB、LLM、ComfyUI 或 GPU 的測試會標記為 `integration`，不會在 hosted PR runner 上假裝通過。Hosted CI 會驗證 integration suite 可被發現，正式整合則由受保護的 Formal Integration workflow 執行。

需要真實 DB、LLM、GPU 或 ComfyUI 的流程，會在 Windows self-hosted runner 上透過同一個 `python run_media_interface.py` 做 Formal Integration smoke test；該 workflow 只在 `main`、schedule 或手動觸發，並使用 `mediaoverload-integration` Environment 的 `MEDIAOVERLOAD_ENV` secret。這樣 Hosted CI 仍能快速回饋，Formal Integration 則負責驗證 production-like runtime；兩者都不能被另一者取代。

Formal Integration runner 必須具備 `self-hosted`, `windows`, `x64`, `mediaoverload-integration` labels、可用的 ComfyUI/GPU、FFmpeg 與資料庫連線。它會使用 `--stage-probe` 與 `--dry-run-publish`，產生並上傳 JSON/log evidence，但不執行 live publish。

以 **Agentic 媒體執行階層**（`agentic/`）為核心的專案：依「目標＋約束」組出計畫，透過技能／工具鏈呼叫 ComfyUI、媒體處理與發佈。角色向的一鍵流程由 **`run_media_interface.py`** 讀取角色 YAML，並依 **`configs/routing.yaml`** 做 **生成策略（generation strategy）** 路由。

---

## 環境與安裝

- **Python**：建議 3.11+（`agentic/pyproject.toml` 要求 `>=3.11`）。
- **根目錄依賴**：`pip install -r requirements.txt`
- **Agentic 套件**（含 `agentic` 指令）：

```bash
cd agentic
pip install -e .
```

- **ComfyUI**：實際出圖／影片時需可連線的 ComfyUI；預設連線由環境變數控制（見下表）。

### 常用環境變數

| 變數 | 說明 | 預設 |
|------|------|------|
| `COMFYUI_HOST` | ComfyUI 主機位址 | `host.docker.internal`（程式內預設） |
| `COMFYUI_PORT` | ComfyUI 埠號 | `8188` |
| `AGENTIC_LLM_MODE` | LLM 後端模式（角色流程的路由與自動 prompt 會用到） | `llm` |
| `AGENTIC_RUN_LOGGER_NAME` | 由 `run_character_workflow` 設定，供執行期記錄用 | （執行時注入） |

---

## 專案結構（精簡）

| 路徑 | 用途 |
|------|------|
| `run_media_interface.py` | 角色媒體介面 CLI：載入角色設定並呼叫 `run_character_workflow` |
| `agentic/` | Agentic runtime：planner、runner、skills、tools |
| `configs/routing.yaml` | 全域 **策略候選**、**各階 workflow 候選**、**數量政策**、路由提示 |
| `configs/characters/*.yaml` | 角色設定（社群、輸出目錄、可選 `additional_params.strategies`） |
| `configs/workflow/*.json` | ComfyUI workflow 定義（檔名 stem 作為 workflow 名稱） |

---

## 生成策略總覽

策略名稱對應 **`character_workflow.CONFIG_MEDIA_TYPE_MAP`**，決定送進 planner 的 `media_type`：

| 策略 `generation_type` | `media_type`（內部） | 說明 |
|------------------------|----------------------|------|
| `text2img` | `image` | 靜態圖（可多张候選） |
| `text2video` | `text2video` | 短影片／單鏡頭動態（預設路由檔中可註解關閉） |
| `text2image2video` | `text2img2video` | 先圖後短片（可含 upscale 階） |
| `text2longvideo` | `long_video` | 文字→關鍵圖／reference→多段 I2V→轉場→合成；不是純 T2V，也非固定 15 秒 |
| `native_h3_story` | `native_h3_story` | Native H3 I2VA：單一 approved opening image 進連續故事 |
| `native_h3_t2v_story` | `native_h3_t2v_story` | Native H3 T2VA：純文字直接生影片 |
| `native_h3_fl2va_story` | `native_h3_fl2va_story` | Native H3 FL2VA：opening + landing 兩端 conditioning |
| `native_h3_l2va_story` | `native_h3_l2va_story` | Native H3 L2VA：只使用 landing frame conditioning |
| `native_h3_ref2va` | `native_h3_ref2va` | Native H3 Ref2VA：有效 manifest 直接使用；空 manifest 自動六張候選圖→Discord 選擇 |
| `text2image2image` | `text2img2img` | 先圖後 img2img refine |
| `sticker_pack` | `sticker_pack` | 多表情貼圖批次 |
| `image2image` | `image` | 單純 img2img 路線（路由需納入候選時才會被選到） |

**LLM 路由**（未指定 `--generation-type` 時）會讀取 `configs/routing.yaml` 的 `routing.strategy_candidates`、`workflow_stage_candidates`、`count_policies`、`routing_hints`，並回傳結構化結果（鍵：`generation_type`、`workflow_plan`、`count_plan`、`reason`）。  
**手動指定策略**時：CLI 傳 `--generation-type <策略名>` 或程式傳 `preferred_generation_type=...`，會略過 LLM，並對各階 workflow 取該策略在設定檔中的**第一個候選**，數量則取各 `count_key` 的 **policy 最小值**。

### 各階 workflow 鍵（`workflow_plan`）

與 `LLMPromptEngine` 的 `WORKFLOW_STAGE_KEYS` 一致，所有策略的計畫物件都包含這五個鍵（未使用則空字串）：

| 鍵 | 用途 |
|----|------|
| `image_workflow_name` | 文生圖或首幀／靜態階 |
| `video_workflow_name` | 圖生影片、片段動態 |
| `refine_workflow_name` | img2img 精修 |
| `transition_workflow_name` | 長片轉場／過幀 |
| `upscale_workflow_name` | 放大收尾 |

各 **策略** 實際會用到的階，請對照下方「策略與 YAML／程式範例」。

### 數量計畫（`count_plan`）

路由結果與約束裡會帶入以下整數（範圍由 `configs/routing.yaml` 的 `count_policies.<策略>` 定義）：

| 鍵 | 意義 |
|----|------|
| `image_count` | 圖像候選或首幀相關數量 |
| `video_count` | 影片支數 |
| `segment_count` | 長片分段數（**僅長片策略語意最重**） |
| `review_selection_limit` | 審核／挑圖上限 |
| `sticker_expression_count` | 貼圖表情數量 |
| `images_per_prompt` | 每個 prompt／表情批次出圖數 |

**長片總時長**（`text2longvideo`）：`duration_seconds = max(10, segment_count × segment_duration)`，其中 `segment_count` 來自路由的 `count_plan`，`segment_duration` 預設 `5` 秒，可透過角色 YAML 覆寫（見 `text2longvideo` 範例）。

---

## 策略一覽：使用方式與範例

以下 **CLI** 假設在專案根目錄執行，並使用內建 `configs/characters/kirby.yaml`（可依需求複製改名）。

### Group 角色選擇

`configs/characters/kirby.yaml` 的 `character.group_name: Kirby` 已啟用群組選角。使用目前的角色入口即可觸發：

```powershell
python run_media_interface.py --character kirby --generation-type text2img --no-review --no-publish
```

每次 `run_character_workflow` 執行時，runtime 會從 `anime.anime_roles` 讀取同一 `group_name` 下 `status=1` 且 `weight>0` 的角色，依 `weight` 做一次加權隨機選擇。角色描述與 keywords 會一併注入 prompt/storyboard；`run_manifest.json` 和 `events.jsonl` 會記錄 `character_selection`、候選數、權重與選中角色。

排程不需要新增參數，維持：

```dotenv
SCHEDULER_CHARACTER=kirby
```

若 MySQL 未設定、查不到可用角色，該 run 會在選角階段直接失敗並記錄 `character.group.selection_failed`，不會自動改用 Kirby。直接呼叫 `build_goal_payload_from_character_config` 只負責建立已選角色的 payload；要執行 group 選角請使用 CLI、scheduler 或 `run_character_workflow`。

### 1. `text2img` — 靜態圖

**會用到的 `workflow_plan` 鍵**：主要是 `image_workflow_name`。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "霓虹夜市中的角色立繪，正視，高完成度" --generation-type text2img --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**（僅建構目標 payload、不執行 runner）：

```python
from pathlib import Path
from agentic.app.character_requests import CharacterGenerationOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    CharacterWorkflowRequest(
        repo_root=repo_root,
        config_path=repo_root / "configs" / "characters" / "kirby.yaml",
        generation=CharacterGenerationOptions(
            prompt="賽博龍與拉麵攤，海報構圖",
            preferred_generation_type="text2img",
        ),
    )
)
# payload["media_type"] == "image"
# payload["constraints"]["image_workflow_name"] 等為路由結果
```

**角色 YAML 可選覆寫**（`additional_params.strategies.text2img`）：

```yaml
additional_params:
  strategies:
    text2img:
      workflow_name: krea2_turbo
      workflow_path: configs/workflow/krea2_turbo.json
```

---

### 2. `text2video` — 短影片／單鏡頭

**會用到的鍵**：`video_workflow_name` 為必要的 T2V stage；`image_workflow_name` 若被選出，只是 ideation／cover review artifact，不會接成影片的首幀 conditioning。若要真正的首幀 I2V，請用 `text2image2video`。

預設 `configs/routing.yaml` 中 `text2video` 可能在 `strategy_candidates` 被註解；若要啟用，請取消註解並確認 workflow 候選存在。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "角色在雨中緩步，單鏡頭 loop 感" --generation-type text2video --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**：

```python
from pathlib import Path
from agentic.app.character_requests import CharacterGenerationOptions, CharacterRuntimeOptions, CharacterReviewOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import run_character_workflow

repo = Path(__file__).resolve().parent
result = run_character_workflow(
    CharacterWorkflowRequest(
        repo_root=repo,
        config_path=repo / "configs" / "characters" / "kirby.yaml",
        generation=CharacterGenerationOptions(
            prompt="短動態展示，鏡頭固定",
            preferred_generation_type="text2video",
        ),
        review=CharacterReviewOptions(publish_after_generate=False),
        runtime=CharacterRuntimeOptions(comfy_host="127.0.0.1", comfy_port=8188),
    )
)
print(result["status"], result.get("routing_summary", {}))
```

**`generation.workflows` 可選**（合併進候選）：

```yaml
generation:
  workflows:
    text2img: krea2_turbo
    text2video: minimax_h3_lowvram_t2v
```

---

### 3. `text2image2video` — 先圖後短片（可 upscale）

**會用到的鍵**：`image_workflow_name`、`video_workflow_name`、`upscale_workflow_name`。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "先鎖定角色再做成短片，電影感光線" --generation-type text2image2video --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**：

```python
from pathlib import Path
from agentic.app.character_requests import CharacterGenerationOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    CharacterWorkflowRequest(
        repo_root=repo_root,
        config_path=repo_root / "configs" / "characters" / "kirby.yaml",
        generation=CharacterGenerationOptions(
            prompt="key visual 轉五秒動態",
            preferred_generation_type="text2image2video",
        ),
    )
)
```

**YAML 階層式覆寫**：

```yaml
additional_params:
  strategies:
    text2image2video:
      first_stage:
        workflow_name: krea2_turbo
        t2i_workflow_path: configs/workflow/krea2_turbo.json
        upscale_workflow_name: Tile Upscaler SDXL
      video:
        workflow_name: minimax_h3_lowvram_i2v
        i2v_workflow_path: configs/workflow/minimax_h3_lowvram_i2v.json
```

---

### 4. `text2longvideo` — 長片／多段

**會用到的鍵**：`image_workflow_name`、`video_workflow_name`、`transition_workflow_name`。

這不是單純的 text-to-long-video。planner 會把 story 拆成多個 segment，依每段抽到的 recipe 產生 anchor/reference 圖，接著跑多段 I2V、tail/continuation 或 transition，最後 concat；總時長由 `segment_count × segment_duration` 決定，不固定為 Native H3 的 15 秒。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "三段式小故事：相遇、追逐、收尾" --generation-type text2longvideo --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**（含長片設定與 TTS）：

```python
from pathlib import Path
from agentic.app.character_requests import CharacterGenerationOptions, CharacterRuntimeOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import run_character_workflow

repo = Path(__file__).resolve().parent
result = run_character_workflow(
    CharacterWorkflowRequest(
        repo_root=repo,
        config_path=repo / "configs" / "characters" / "kirby.yaml",
        generation=CharacterGenerationOptions(
            prompt="旁白式多場景故事",
            preferred_generation_type="text2longvideo",
        ),
        runtime=CharacterRuntimeOptions(comfy_host="127.0.0.1", comfy_port=8188),
    )
)
```

**YAML**（片段時長、段數語意、是否 TTS）：

```yaml
additional_params:
  strategies:
    text2longvideo:
      first_stage:
        workflow_name: krea2_turbo
      video_generation:
        workflow_name: minimax_h3_lowvram_i2v
      frame_transition:
        workflow_name: krea2_turbo_img2img
      longvideo_config:
        segment_count: 4
        segment_duration: 5
        use_tts: true
```

---

### 5. `text2image2image` — 先圖後 refine

**會用到的鍵**：`image_workflow_name`、`refine_workflow_name`。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "保留構圖，加強細節與線稿乾淨度" --generation-type text2image2image --comfy-host 127.0.0.1 --comfy-port 8188
```

**YAML**：

```yaml
additional_params:
  strategies:
    text2image2image:
      first_stage:
        workflow_name: krea2_turbo
      second_stage:
        workflow_name: krea2_turbo_img2img
        workflow_path: configs/workflow/krea2_turbo_img2img.json
```

---

### 6. `sticker_pack` — 貼圖包

**會用到的鍵**：`image_workflow_name`、`video_workflow_name`（動態貼圖時）。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "8 種聊天室表情：開心、翻白眼、大哭..." --generation-type sticker_pack --comfy-host 127.0.0.1 --comfy-port 8188
```

**YAML**：

```yaml
additional_params:
  strategies:
    sticker_pack:
      static_config:
        workflow_name: krea2_turbo
      animated_config:
        workflow_name: minimax_h3_lowvram_i2v
        i2v_workflow_path: configs/workflow/minimax_h3_lowvram_i2v.json
```

---

### 7. `image2image` — 純 refine 路線

需在 **`configs/routing.yaml`** 的 `routing.strategy_candidates` 加入 `image2image` 才會進入 LLM 候選；程式手動指定 `preferred_generation_type="image2image"` 則可直接走覆寫路由邏輯。

**Python 範例**：

```python
from pathlib import Path
from agentic.app.character_requests import CharacterGenerationOptions, CharacterWorkflowRequest
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    CharacterWorkflowRequest(
        repo_root=repo_root,
        config_path=repo_root / "configs" / "characters" / "kirby.yaml",
        generation=CharacterGenerationOptions(
            prompt="在既有構圖上重做光影",
            preferred_generation_type="image2image",
        ),
    )
)
```

**YAML**：

```yaml
additional_params:
  strategies:
    image2image:
      workflow_name: krea2_turbo_img2img
      workflow_path: configs/workflow/krea2_turbo_img2img.json
```

---

## 完整 CLI 執行清單（含 Discord review / publish gate）

以下是目前 repo 可由 `run_media_interface.py` 呼叫的完整策略清單。每一列都會走同一條角色流程：載入 YAML → 路由 → 建立 workflow plan → 執行 ComfyUI → 進入 publish/review stage。`--dry-run-publish` 會保留 publish stage 與 Discord 審核，但不把結果送到社群平台；因此適合逐條 E2E 驗證。不要把 `--no-publish` 加到這組測試，否則不會驗證 Discord publish gate。

逐一呼叫各策略（每個命令都要在 Discord 完成審核；若成果不合格，請在 Discord 選 reject/block）：

```powershell
# 1. 靜態圖：krea2_turbo
python run_media_interface.py --character kirby --prompt '高完成度角色主視覺，霓虹夜市，清楚輪廓' --generation-type text2img

# 2. 短影片／單鏡頭：minimax_h3_lowvram_t2v
python run_media_interface.py --character kirby --prompt '單鏡頭短動畫，角色在雨中奔跑，鏡頭跟拍' --generation-type text2video

# 3. 先圖後短片：krea2_turbo -> minimax_h3_lowvram_i2v -> Tile Upscaler SDXL
python run_media_interface.py --character kirby --prompt '先鎖定角色外觀，再把 key visual 做成電影感短片' --generation-type text2image2video

# 4. 長片／多段故事：每段依 recipe 產生 anchor/reference -> I2V -> tail/transition -> concat
python run_media_interface.py --character kirby --prompt '三段式故事：相遇、追逐、收尾；每段要有狀態轉移' --generation-type text2longvideo

# 5. Native H3 I2VA：generated opening image -> minimax_h3_lowvram_15s_fl2va_i2v
python run_media_interface.py --character kirby --prompt 'Native H3 單一連續故事，開場動作要立即發生' --generation-type native_h3_story

# 6. Native H3 T2VA：prompt only -> minimax_h3_lowvram_t2v
python run_media_interface.py --character kirby --prompt 'Native H3 純文字生影片，角色從第一幀就開始行動' --generation-type native_h3_t2v_story

# 7. Native H3 FL2VA：generated opening + generated landing -> minimax_h3_lowvram_15s_fl2va_i2v
python run_media_interface.py --character kirby --prompt 'Native H3 首尾幀故事，起點與結局都要由人工審核' --generation-type native_h3_fl2va_story

# 8. Native H3 L2VA：generated landing only -> minimax_h3_lowvram_15s_fl2va_i2v
python run_media_interface.py --character kirby --prompt 'Native H3 以最後一幀作為故事落點，前段保持因果動作' --generation-type native_h3_l2va_story

# 9. Native H3 Ref2VA：valid manifest 直用；空 manifest -> 六張候選圖 -> Discord 選擇 -> minimax_h3_ref2va
python run_media_interface.py --character kirby --prompt 'Native H3 參考圖與參考影片共同控制身份和運鏡' --generation-type native_h3_ref2va

# 10. 先圖後 img2img refine：krea2_turbo -> krea2_turbo_img2img
python run_media_interface.py --character kirby --prompt '保留構圖與角色身份，只修正光影和細節' --generation-type text2image2image

# 11. 貼圖包：krea2_turbo -> minimax_h3_lowvram_i2v（動態貼圖階段）
python run_media_interface.py --character kirby --prompt '聊天貼圖表情包：開心、生氣、驚訝、無奈' --generation-type sticker_pack
```

### 自動路由與加權隨機

角色流程先選角色，再選策略，最後才生成內容：scheduler/runtime 先依 YAML 的 `character.group_name` 從 DB 做 group weighted selection，再讀取 `generation.generation_type_weights` 做 strategy weighted selection，並把新聞或指定 prompt 交給已選策略的 LLM story/brief stage；LLM 不決定角色或 scheduler 要走哪一種 strategy。scheduler 會把候選 route 放進持久化 shuffle bag（狀態預設在 `agentic/state/routing_selection/<config>.json`），所以短窗口不會因隨機抽樣連續撞到同一路由，但整體仍遵守 YAML 權重。Kirby 目前的 strategy 權重包含 `text2img: 1`、`text2image2video: 1`、`text2longvideo: 2`、`native_h3_story: 1`、`native_h3_t2v_story: 1`、`native_h3_fl2va_story: 1`、`native_h3_l2va_story: 1`、`native_h3_ref2va: 1`、`text2image2native_h3_ref2va: 1`、`sticker_pack: 2`。

固定 route 的 image/refine/transition workflow 會先經過 `AssetRegistry` 的 required-asset readiness，再依 `configs/routing.yaml` 的 `workflow_selection_weights` 選擇；空的 required-asset manifest 只代表未驗證，不會被當成 ready。當前這台 ComfyUI 只有 Krea 的 image assets 被明確驗證，因此 Krea 集中是資產可用性結果，不是機率失效；先補齊並登錄其他 workflow assets，權重才會在它們之間生效。explicit generation-type override 只固定策略家族，若 scheduler 有 RNG，stage workflow 仍會依權重抽樣。若要指定 routing bag 檔案，可設定 `SCHEDULER_ROUTING_HISTORY_PATH`。

`text2image2video` 的正常 review path 仍保留 6 張 raw keyframe 候選，但不再先跑未被 review 消費的 upscale。Discord 明確 Reject 仍會停止；若 review 發生 timeout、連線錯誤或沒有決策，`pre_video_review.failure_policy: fallback_to_top` 會選 deterministic top-ranked frame 繼續 I2V，並在 run manifest 留下 fallback evidence。

直接呼叫 `run_character_workflow` 時也會執行 group 選角；傳入 `rng` 可讓測試或受控實測重現同一組權重抽樣。`build_goal_payload_from_character_config` 是 payload builder，不是 group 選角入口。

```powershell
# LLM 自動路由（不指定 --generation-type）
python run_media_interface.py --character kirby --prompt '幫我做一個有明確衝突與結局的 Kirby 媒體作品'

# 加權隨機：每次重新抽樣；抽樣後仍會完整走同一個 E2E + Discord gate
$randomType = python -c "import random,sys; from pathlib import Path; sys.path.insert(0,'agentic/src'); from agentic.app.character_workflow import choose_media_type,load_character_config; print(choose_media_type(load_character_config(Path('configs/characters/kirby.yaml')), rng=random.Random())[0])"
Write-Host "weighted random generation_type=$randomType"
# news-driven 會抓取一則尚未使用的新聞；不要用舊 prompt 或舊 media path 代替
python run_media_interface.py --character kirby --generation-type $randomType --news-driven
```

`image2image` 目前不是 `configs/routing.yaml` 的預設候選，只有在角色 YAML 的 `additional_params.strategies.image2image` 提供 workflow override 後才適合執行；不要把它和可直接 E2E 的 `text2image2image` 混用。它的 CLI 形式如下：

```powershell
python run_media_interface.py --character kirby --prompt '在既有構圖上重做光影' --generation-type image2image
```

### MiniMax H3 五種 canonical mode 的獨立 ComfyUI E2E

這支 runner 直接測試 H3 模式與 strict technical QA，不取代上面的角色流程或 Discord review。可一次跑全部，也可以明確一條一條跑：

```powershell
# 全部五種：t2va / i2va / fl2va / l2va / ref2va
python scripts/run_h3_modes_e2e.py `
  --comfy-root 'D:\ComfyUI_windows_portable' `
  --output-root 'D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_h3_p2_e2e'

# 逐條重跑指定 mode（可重複 --mode）
python scripts/run_h3_modes_e2e.py --mode t2va
python scripts/run_h3_modes_e2e.py --mode i2va
python scripts/run_h3_modes_e2e.py --mode fl2va
python scripts/run_h3_modes_e2e.py --mode l2va
python scripts/run_h3_modes_e2e.py --mode ref2va

# 只對已產生的影片重跑 strict QA
python scripts/verify_h3_e2e_outputs.py `
  --output-root 'D:\ComfyUI_windows_portable\ComfyUI\output\mediaoverload_h3_p2_e2e'
```

H3 runner 的 canonical workflow 對應：`t2va → minimax_h3_lowvram_t2v`、`i2va → minimax_h3_lowvram_i2v`、`fl2va/l2va → minimax_h3_lowvram_15s_fl2va_i2v`、`ref2va → minimax_h3_ref2va`。完整 conditioning、解析度、音訊與 QA contract 請見 [`docs/minimax_h3_p2_e2e.md`](docs/minimax_h3_p2_e2e.md)。

### 測試順序與 publish 邊界

建議順序是先跑 `text2img` 確認基礎 ComfyUI 與 Discord，再跑五種 native H3，接著跑一般影片／長片／貼圖，最後跑 Ref2VA 與隨機。每條命令都應保留 `run_id`、`log_path`、`review_session_path`、輸出媒體與 `publish_review_summary.json`。

本機測試不會自動公開發佈：`--dry-run-publish` 只驗證 publish/review graph；即使 Discord 通過，也不會送出社群平台。若要做不公開的平台 adapter smoke test，使用 `--publish-mode safe_poc`（YouTube private、Facebook draft、Instagram container-only）；不要在未經人工確認前使用 `--publish-mode live`。

`--news-driven` 會把 `title + keyword` 寫入 `agentic/state/news_selection/<character>.json`，下一次隨機執行會排除已用新聞；沒有可用的新新聞時會直接 fail，不會用 generic prompt 或過去媒體充數。OpenRouter publish caption 預設會輪替已驗證 vision pool、每個模型最多 retry 2 次；可用 `AGENTIC_PUBLISH_CAPTION_MAX_RETRIES`、`AGENTIC_PUBLISH_CAPTION_MAX_MODELS_PER_CALL`（`0`/未設定代表整個 pool）、`AGENTIC_PUBLISH_CAPTION_TIMEOUT_SECONDS` 調整。所有 prompt/story LLM request 也可用 `AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS` 設定單次 request timeout（預設 30 秒）。

---

## `run_character_workflow` 參數說明

| 參數 | 型別 | 說明 |
|------|------|------|
| `repo_root` | `Path` | 專案根目錄（需能解析到 `configs/routing.yaml`） |
| `config_path` | `str \| Path` | 角色 YAML 路徑 |
| `prompt` | `str` | 使用者提示；空字串時可能觸發新聞／LLM 自動場景（視設定） |
| `temperature` | `float` | 來源溫度，會寫入 `constraints.source_temperature` |
| `preferred_generation_type` | `str \| None` | **強制策略**；對應 CLI `--generation-type` |
| `dry_run_publish` | `bool` | 發佈階段不實際送出 |
| `publish_after_generate` | `bool` | 生成成功後是否再跑 `publish_review` |
| `output_dir` | `str \| None` | 覆寫輸出目錄 |
| `enable_review_loop` | `bool` | 是否啟用支援的複查／重試分支 |
| `review_notes` | `str` | 複查備註 |
| `comfy_host` | `str \| None` | 覆寫 ComfyUI host |
| `comfy_port` | `int \| None` | 覆寫 ComfyUI port |
| `comfy_root` | `str \| None` | ComfyUI 安裝根目錄（資產檢查用） |
| `auto_download_assets` | `bool` | 是否允許自動準備 workflow 資產 |
| `rng` | `random.Random \| None` | 僅影響未強制策略時的加權抽樣（若未來啟用） |

**回傳 dict** 主要鍵：`status`、`run_id`、`plan`、`generation`、`routing`、`routing_summary`、`publish`、`stage_status`、`artifacts`、`memory` 等。即使 review、caption 或 publish 失敗，`artifacts.media_paths` 仍會保留已產生的影片／圖片；image-only route 也不會把 YouTube 當成可用影片平台。

---

## 直接操作 Comfy workflow：`AgenticNodeManager` 內建策略

`agentic.tools.comfy_backend.AgenticNodeManager` 內建兩類自動對節點套參數的 **builtin strategies**（不需手寫 `node_id` 時）：

| 名稱 | 行為 |
|------|------|
| `text` | 依序嘗試 `PrimitiveString` / `CLIPTextEncode`，並依 `is_negative` 區分正負向提示 |
| `sampler` | 依序對 `RandomNoise`、`KSamplerAdvanced`、`KSampler`、`MMAudioSampler` 寫入種子 |

**範例**（載入 workflow JSON，產生 updates 再交給 `AgenticMediaGenerator`）：

```python
import json
from pathlib import Path
from agentic.tools.comfy_backend import AgenticNodeManager, AgenticMediaGenerator

workflow_path = Path("configs/workflow/krea2_turbo.json")
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

updates = AgenticNodeManager.generate_updates(
    workflow,
    updates_config=None,
    description="1girl, anime, upper body, soft light",
    seed=123456789,
    use_noise_seed=False,
    is_negative=False,
)

extra = AgenticNodeManager.generate_updates(
    workflow,
    updates_config=[
        {
            "node_type": "CLIPTextEncode",
            "node_index": 0,
            "inputs": {"text": "low quality"},
            "filter": {"is_negative": True},
        }
    ],
    description=None,
    seed=987654321,
)

generator = AgenticMediaGenerator(host="127.0.0.1", port=8188)
saved = generator.generate(
    str(workflow_path),
    updates + extra,
    output_dir="./output",
    file_prefix="demo",
)
print(saved)
```

`generate_updates` 相關參數：

| 參數 | 說明 |
|------|------|
| `workflow` | ComfyUI API 格式的 workflow `dict` |
| `updates_config` | 自訂節點更新列表；可含 `node_id` + `direct_update`，或 `node_type` + `node_index` + `inputs` + `filter` |
| `description` | 若未在 `updates_config` 手動指定文字節點，則套用內建 **text** 策略寫入正向提示 |
| `seed` | 若設定，套用 **sampler** 策略 |
| `use_noise_seed` | `True` 時優先寫 `noise_seed` 類節點 |
| `exclude_sampler_indices` | 略過指定 sampler 索引 |
| `is_negative` | 傳入 `additional_params` 供內建文字策略篩選負向節點 |

---

## 另見

- 底層 runtime 設計與 **`agentic` CLI**（`--goal`、`--media-type` 等）：請讀 [`agentic/README.md`](agentic/README.md)。
