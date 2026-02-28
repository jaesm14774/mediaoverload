# Quick Draw — 快速圖片生成範例

基於 MediaOverload 框架的彈性圖片生成工具，使用 **FlexibleGenerator** 提供簡單直覺的 API。

## 目錄

- [特點](#特點)
- [快速開始](#快速開始)
- [可用的 System Prompts](#可用的-system-prompts)
- [配置說明](#配置說明)
- [進階用法](#進階用法)
- [返回值結構](#返回值結構)
- [與完整版的差異](#與完整版的差異)
- [常見問題](#常見問題)
- [工具和輔助類](#工具和輔助類)

---

## 特點

- **簡化流程** — 跳過耗時的圖文匹配分析和文章生成
- **快速執行** — 專注於圖片生成，適合人工審核
- **彈性設計** — 使用 system_prompt + keywords 架構
- **完全相容** — 使用現有的 MediaOverload 基礎設施

---

## 快速開始

### 方式 1：Jupyter Notebook（推薦）

```bash
# 在專案根目錄執行
jupyter notebook examples/flexible_generation_examples.ipynb
```

Notebook 包含完整的範例和說明、互動式環境、自動顯示生成的圖片，以及多種風格和場景示範。

### 方式 2：在代碼中使用

#### 基本圖片生成

```python
import os
from examples.quick_draw.helpers import FlexibleGenerator

# Windows 環境需要設定 ComfyUI 連接
os.environ['COMFYUI_HOST'] = '127.0.0.1'

# 初始化生成器
generator = FlexibleGenerator()

# 生成圖片
result = generator.generate_images(
    keywords=["cat", "cherry blossoms", "spring"],
    system_prompt="stable_diffusion_prompt",
    character="kirby",
    style="soft lighting, peaceful atmosphere",
    num_images=4
)

print(f"生成了 {len(result['media_files'])} 張圖片")
```

#### 不同風格範例

```python
# 1. 黑色幽默風格
result = generator.generate_images(
    keywords="sleeping peacefully",
    system_prompt="black_humor_system_prompt",
    character="kirby",
    secondary_character="waddle dee",
    style="minimalist style",
    num_images=2
)

# 2. 雙角色互動
result = generator.generate_images(
    keywords=["friendship", "playing together"],
    system_prompt="two_character_interaction_generate_system_prompt",
    character="kirby",
    secondary_character="waddle dee",
    style="warm and cozy",
    num_images=2
)

# 3. 電影級別風格
result = generator.generate_images(
    keywords=["epic adventure", "heroic pose"],
    system_prompt="cinematic_stable_diffusion_prompt",
    character="kirby",
    style="cinematic composition",
    num_images=2
)

# 4. 佛性 / 靈性風格
result = generator.generate_images(
    keywords=["meditation", "enlightenment"],
    system_prompt="buddhist_combined_image_system_prompt",
    character="kirby",
    style="spiritual atmosphere",
    num_images=2
)
```

#### 影片生成

```python
result = generator.generate_videos(
    keywords=["flying", "stars", "night sky"],
    system_prompt="stable_diffusion_prompt",
    character="kirby",
    style="smooth motion",
    num_videos=2
)
```

#### 批次生成

```python
prompts = [
    {
        "keywords": ["morning", "sunrise"],
        "style": "bright and cheerful"
    },
    {
        "keywords": ["night", "stars"],
        "style": "peaceful and mystical"
    }
]

results = generator.batch_generate(
    prompts=prompts,
    media_type="image",
    base_config={
        "character": "kirby",
        "system_prompt": "stable_diffusion_prompt",
        "num_images": 2
    }
)
```

---

## 可用的 System Prompts

FlexibleGenerator 的核心是 **system_prompt + keywords** 架構。

| System Prompt | 說明 | 適用場景 |
|---------------|------|---------|
| `stable_diffusion_prompt` | 標準 Stable Diffusion 風格 | 通用圖片生成 |
| `black_humor_system_prompt` | 黑色幽默 | 諷刺、反差效果 |
| `buddhist_combined_image_system_prompt` | 佛性 / 靈性風格 | 禪意、靈性主題 |
| `cinematic_stable_diffusion_prompt` | 電影級別 | 戲劇性、史詩感 |
| `two_character_interaction_generate_system_prompt` | 雙角色互動 | 角色對話和互動 |
| `warm_scene_description_system_prompt` | 溫馨場景 | 溫暖、治癒系 |
| `unbelievable_world_system_prompt` | 不可思議的世界 | 超現實、奇幻 |

---

## 配置說明

### 環境需求

1. 確保 `media_overload.env` 已正確配置
2. MySQL 資料庫中有角色資料
3. ComfyUI 已啟動並可訪問

### ComfyUI 連接設定

#### Windows / 本機環境（如 Jupyter Notebook）

在導入模組前設定環境變數：

```python
import os
os.environ['COMFYUI_HOST'] = '127.0.0.1'
os.environ['COMFYUI_PORT'] = '8188'
```

#### Docker 環境

無需額外設定，預設使用 `host.docker.internal:8188`。

#### 自定義連接

```python
from lib.comfyui.websockets_api import ComfyUICommunicator

communicator = ComfyUICommunicator(host='192.168.1.100', port=8188)
```

**配置優先順序**：

1. 明確傳入的參數（`host`、`port`）
2. 環境變數（`COMFYUI_HOST`、`COMFYUI_PORT`）
3. 預設值（`host.docker.internal`、`8188`）

### 預設路徑

| 項目 | 路徑 |
|------|------|
| 工作流 | `configs/workflow/` |
| 輸出 | `output_media/` |
| 環境變數 | `media_overload.env` |

### 自定義路徑

```python
generator = FlexibleGenerator(
    workflow_folder='your/workflow/path',
    output_folder='your/output/path',
    env_path='your/env/path'
)
```

---

## 進階用法

### 使用 ConfigBuilder（底層 API）

如果需要更細緻的控制，可以直接使用 ConfigBuilder：

```python
from examples.quick_draw.helpers import ConfigBuilder
from examples.simple_content_service import SimpleContentGenerationService

config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_workflow('configs/workflow/nova-anime-xl.json') \
    .with_output_dir('output_media') \
    .with_prompt('peaceful sleeping') \
    .with_style('minimalist') \
    .with_image_system_prompt('stable_diffusion_prompt') \
    .with_images_per_description(2) \
    .build()

# 使用簡化服務
service = SimpleContentGenerationService()
result = service.generate_content(config)
```

### Image to Image 生成

```python
config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_workflow('configs/workflow/nova-anime-xl.json') \
    .with_input_image('path/to/input.png') \
    .with_denoise(0.7) \
    .with_prompt('transform into watercolor style') \
    .with_image_system_prompt('stable_diffusion_prompt') \
    .build()

service = SimpleContentGenerationService()
result = service.generate_content(config)
```

---

## 返回值結構

所有生成方法返回相同的結構：

```python
{
    'descriptions': List[str],      # 生成的描述列表
    'media_files': List[str],       # 生成的圖片 / 影片路徑列表
    'filter_results': [],           # 空列表（已跳過分析）
    'article_content': ''           # 空字串（已跳過文章生成）
}
```

---

## 與完整版的差異

| 功能 | Quick Draw（範例版） | 完整版 |
|------|---------------------|--------|
| 描述生成 | 有 | 有 |
| 圖片 / 視頻生成 | 有 | 有 |
| 圖文匹配分析 | 跳過 | 有 |
| 文章內容生成 | 跳過 | 有 |
| Hashtag 生成 | 跳過 | 有 |
| 執行速度 | **快速** | 較慢 |
| 適用場景 | 快速測試、人工審核 | 自動化發布 |

### 為什麼跳過分析和文章生成？

1. **節省時間** — 圖文匹配分析和 hashtag 生成需要大量時間
2. **人工審核** — 範例用途，通常需要人工檢查結果
3. **快速迭代** — 專注於圖片生成，加快測試速度
4. **靈活性** — 生成後可以手動決定後續處理

---

## 常見問題

### 如何切換不同的工作流？

在生成時指定 `workflow` 參數：

```python
result = generator.generate_images(
    keywords=["adventure"],
    character='Kirby',
    workflow='flux-krea-dev',  # 使用不同的工作流（不含 .json）
    system_prompt='stable_diffusion_prompt'
)
```

### 如何調整圖片數量？

使用 `num_images` 參數：

```python
result = generator.generate_images(
    keywords=["peaceful"],
    character='Kirby',
    num_images=10,  # 生成 10 張圖片
    system_prompt='stable_diffusion_prompt'
)
```

### 如何自定義 system_prompt？

System prompts 定義在 `configs/prompt/image_system_guide.py` 中。可以：

1. 使用現有的 system_prompt
2. 在該文件中添加新的 system_prompt
3. 在生成時指定新的 system_prompt 名稱

### 如何使用完整版的功能？

使用 `ContentGenerationService`：

```python
from lib.services.implementations.content_generation_service import ContentGenerationService

service = ContentGenerationService()
result = service.generate_content(config)  # 包含完整分析和文章生成
```

### Windows 環境中無法連接 ComfyUI？

在導入模組前設定環境變數：

```python
import os
os.environ['COMFYUI_HOST'] = '127.0.0.1'

# 然後再導入
from examples.quick_draw.helpers import FlexibleGenerator
```

---

## 工具和輔助類

### FlexibleGenerator

核心生成器類，提供簡單的 API：

- `generate_images()` — 生成圖片
- `generate_videos()` — 生成影片
- `batch_generate()` — 批次生成
- `generate_from_config()` — 使用自定義配置生成

### ConfigBuilder

配置建構器，提供 Builder Pattern API：

- `with_character()` — 設定角色
- `with_prompt()` — 設定提示詞
- `with_keywords()` — 設定關鍵字
- `with_style()` — 設定風格
- `with_workflow()` — 設定工作流
- `with_image_system_prompt()` — 設定系統提示詞
- `with_input_image()` — 設定輸入圖片（image2image）
- `with_denoise()` — 設定降噪強度

詳細 API 請參考 [ConfigBuilder 原始碼](helpers/config_builder.py)。

---

## 授權

MIT License
