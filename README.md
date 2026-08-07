# MediaOverload

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
| `text2longvideo` | `long_video` | 多段故事／分鏡／轉場 |
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

### 1. `text2img` — 靜態圖

**會用到的 `workflow_plan` 鍵**：主要是 `image_workflow_name`。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "霓虹夜市中的角色立繪，正視，高完成度" --generation-type text2img --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**（僅建構目標 payload、不執行 runner）：

```python
from pathlib import Path
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    repo_root,
    repo_root / "configs" / "characters" / "kirby.yaml",
    prompt="賽博龍與拉麵攤，海報構圖",
    preferred_generation_type="text2img",
)
# payload["media_type"] == "image"
# payload["constraints"]["image_workflow_name"] 等為路由結果
```

**角色 YAML 可選覆寫**（`additional_params.strategies.text2img`）：

```yaml
additional_params:
  strategies:
    text2img:
      workflow_name: anima_anime        # 或
      workflow_path: configs/workflow/anima_anime.json
```

---

### 2. `text2video` — 短影片／單鏡頭

**會用到的鍵**：`image_workflow_name`、`video_workflow_name`（首幀＋ I2V）。

預設 `configs/routing.yaml` 中 `text2video` 可能在 `strategy_candidates` 被註解；若要啟用，請取消註解並確認 workflow 候選存在。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "角色在雨中緩步，單鏡頭 loop 感" --generation-type text2video --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**：

```python
from pathlib import Path
from agentic.app.character_workflow import run_character_workflow

repo = Path(__file__).resolve().parent
result = run_character_workflow(
    repo,
    repo / "configs" / "characters" / "kirby.yaml",
    prompt="短動態展示，鏡頭固定",
    preferred_generation_type="text2video",
    comfy_host="127.0.0.1",
    comfy_port=8188,
    publish_after_generate=False,
)
print(result["status"], result.get("routing_summary", {}))
```

**`generation.workflows` 可選**（合併進候選）：

```yaml
generation:
  workflows:
    text2img: z_image_plus_nova_model
    text2video: wan2.2_gguf_i2v
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
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    repo_root,
    repo_root / "configs" / "characters" / "kirby.yaml",
    prompt="key visual 轉五秒動態",
    preferred_generation_type="text2image2video",
)
```

**YAML 階層式覆寫**：

```yaml
additional_params:
  strategies:
    text2image2video:
      first_stage:
        workflow_name: z_image_plus_nova_model
        t2i_workflow_path: configs/workflow/z_image_plus_nova_model.json
        upscale_workflow_name: Tile Upscaler SDXL
      video:
        workflow_name: wan2.2_gguf_i2v
        i2v_workflow_path: configs/workflow/wan2.2_gguf_i2v.json
```

---

### 4. `text2longvideo` — 長片／多段

**會用到的鍵**：`image_workflow_name`、`video_workflow_name`、`transition_workflow_name`。

**CLI 範例**：

```bash
python run_media_interface.py --character kirby --prompt "三段式小故事：相遇、追逐、收尾" --generation-type text2longvideo --comfy-host 127.0.0.1 --comfy-port 8188
```

**Python 範例**（含長片設定與 TTS）：

```python
from pathlib import Path
from agentic.app.character_workflow import run_character_workflow

repo = Path(__file__).resolve().parent
result = run_character_workflow(
    repo,
    repo / "configs" / "characters" / "kirby.yaml",
    prompt="旁白式多場景故事",
    preferred_generation_type="text2longvideo",
    comfy_host="127.0.0.1",
    comfy_port=8188,
)
```

**YAML**（片段時長、段數語意、是否 TTS）：

```yaml
additional_params:
  strategies:
    text2longvideo:
      first_stage:
        workflow_name: z_image_plus_nova_model
      video_generation:
        workflow_name: wan2.2_gguf_i2v
      frame_transition:
        workflow_name: image_to_image
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
        workflow_name: nova_model_plus_z_image_anime
      second_stage:
        workflow_name: z_image_i2i_anime
        workflow_path: configs/workflow/z_image_i2i_anime.json
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
        workflow_name: anima_anime
      animated_config:
        workflow_name: wan2.2_gguf_i2v
        i2v_workflow_path: configs/workflow/wan2.2_gguf_i2v.json
```

---

### 7. `image2image` — 純 refine 路線

需在 **`configs/routing.yaml`** 的 `routing.strategy_candidates` 加入 `image2image` 才會進入 LLM 候選；程式手動指定 `preferred_generation_type="image2image"` 則可直接走覆寫路由邏輯。

**Python 範例**：

```python
from pathlib import Path
from agentic.app.character_workflow import build_goal_payload_from_character_config

repo_root = Path(__file__).resolve().parent
payload = build_goal_payload_from_character_config(
    repo_root,
    repo_root / "configs" / "characters" / "kirby.yaml",
    prompt="在既有構圖上重做光影",
    preferred_generation_type="image2image",
)
```

**YAML**：

```yaml
additional_params:
  strategies:
    image2image:
      workflow_name: z_image_i2i_anime
      workflow_path: configs/workflow/z_image_i2i_anime.json
```

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

**回傳 dict** 主要鍵：`status`、`run_id`、`plan`、`generation`、`routing`、`routing_summary`、`publish`、`memory` 等。

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

workflow_path = Path("configs/workflow/anima_anime.json")
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
