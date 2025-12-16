# MediaOverload 使用範例

本目錄包含 MediaOverload 專案的各種使用範例，幫助您快速上手和理解系統功能。

## 📁 目錄結構

```
examples/
├── README.md                          # 本文件
├── all_strategies_examples.ipynb      # 所有策略完整範例 ⭐⭐ 推薦
├── simple_content_service.py          # 簡化的內容生成服務
├── social_media_example.py            # 社群媒體發布範例
└── quick_draw/                        # Quick Draw 模組
    ├── README.md                      # Quick Draw 詳細說明
    └── helpers/                       # 輔助工具
        ├── __init__.py
        ├── config_builder.py          # 配置建構器
        ├── workflow_loader.py         # 工作流載入器
        └── flexible_generator.py      # 彈性生成器 ⭐
```

## 🚀 快速開始

### ⭐⭐ 方式 0: 所有策略完整範例（最推薦）

**包含所有策略的完整範例！** 每個策略都有自定義 Prompt 和從資料庫獲取 News 兩種模式，並支援批量生成：

```bash
# 在 Jupyter Notebook 中使用
jupyter notebook examples/all_strategies_examples.ipynb
```

**特色功能**：
- ✅ 8 種策略完整範例（Text2Image, Image2Image, Text2Image2Image, Text2Video, Text2Image2Video, Text2LongVideo, Text2LongVideoFirstFrame, StickerPack）
- ✅ 每個策略支援自定義 Prompt 和從資料庫獲取 News
- ✅ 支援批量生成（可指定數量，如 30 張、50 張等）
- ✅ 自動執行生成，無需手動干預
- ✅ **🆕 長影片直接模式**：不保存中間圖片，只輸出最終完整影片（含 TTS）

**批量生成範例**：

```python
# 批量生成 30 張圖片
results = batch_generate_by_count(
    strategy_type='text2image',
    num_total=30,
    use_news=True,
    character="kirby",
    num_images=4
)

# 🆕 批量生成長影片（直接模式）
results = batch_generate_by_count(
    strategy_type='text2longvideo',
    num_total=2,
    use_news=True,
    character="kirby",
    skip_candidate_stage=True,  # 不保存中間圖片
    segment_count=3,
    use_tts=True
)
```

詳細用法請參考 [all_strategies_examples.ipynb](all_strategies_examples.ipynb)

### 方式 1: 使用 ConfigBuilder（進階用法）

如果需要更細緻的控制，可以直接使用 ConfigBuilder：

```python
from examples.quick_draw.helpers import ConfigBuilder
from examples.simple_content_service import SimpleContentGenerationService

# 建立配置
config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_workflow('configs/workflow/nova-anime-xl.json') \
    .with_output_dir('output_media') \
    .with_prompt('your keywords here') \
    .with_image_system_prompt('stable_diffusion_prompt') \
    .with_images_per_description(4) \
    .build()

# 使用簡化服務（跳過分析和文章生成）
service = SimpleContentGenerationService()
result = service.generate_content(config)
```

## 📚 範例說明

### All Strategies Examples ⭐⭐

**檔案**: `all_strategies_examples.ipynb`

包含所有 8 種生成策略的完整範例：

**支援的策略**：
1. Text2ImageStrategy - 文生圖
2. Image2ImageStrategy - 圖生圖
3. Text2Image2ImageStrategy - 文生圖 → 圖生圖
4. Text2VideoStrategy - 文生影片
5. Text2Image2VideoStrategy - 文生圖 → 圖生影片
6. Text2LongVideoStrategy - 文生長影片（尾幀驅動）
7. Text2LongVideoFirstFrameStrategy - 文生長影片（首幀驅動）
8. StickerPackStrategy - 貼圖包生成

**每個策略都包含**：
- ✅ 自定義 Prompt 範例
- ✅ 從資料庫獲取 News 範例
- ✅ 批量生成範例（可指定數量）

**優點**：
- 最全面的範例集合
- 互動式 Jupyter Notebook
- 支援批量生成（30 張、50 張等）
- 包含輔助函數簡化使用

**使用時機**：
- 需要了解所有策略的使用方法
- 喜歡互動式開發環境
- 需要查看即時結果

### Simple Content Service

**檔案**: `simple_content_service.py`

簡化版的內容生成服務，專為範例設計：

- ✅ **包含**: 描述生成、圖片/視頻生成
- ❌ **跳過**: 圖文匹配分析、文章內容生成、Hashtag 生成

**優點**:
- 執行速度快
- 適合快速測試
- 適合需要人工審核的情況

**使用時機**:
- 開發和測試階段
- 需要細緻控制配置
- 與 ConfigBuilder 搭配使用

### ConfigBuilder

**檔案**: `quick_draw/helpers/config_builder.py`

配置建構器，提供 Builder Pattern API：

```python
config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_prompt('keywords here') \
    .with_style('minimalist') \
    .with_workflow('configs/workflow/nova-anime-xl.json') \
    .with_image_system_prompt('stable_diffusion_prompt') \
    .with_images_per_description(4) \
    .build()
```

**主要方法**:
- `with_character()` - 設定角色
| 圖片/視頻生成 | ✅ | ✅ |
| 圖文匹配分析 | ❌ | ✅ |
| 文章內容生成 | ❌ | ✅ |
| Hashtag 生成 | ❌ | ✅ |
| 執行速度 | **快** | 慢 |
| 適用場景 | 測試、範例 | 生產環境 |

### 如何使用完整版？

如需完整功能，使用標準的 ContentGenerationService：

```python
from lib.services.implementations.content_generation_service import ContentGenerationService
from lib.media_auto.strategies.base_strategy import GenerationConfig

service = ContentGenerationService()
config = GenerationConfig(...)
result = service.generate_content(config)  # 包含所有功能
```

## ⚙️ 配置

### 環境需求

1. **環境變數**: 確保 `media_overload.env` 已配置
2. **資料庫**: MySQL 資料庫包含角色資料
3. **ComfyUI**: 已啟動並可訪問

### ComfyUI 連接設定

本專案支援不同環境的 ComfyUI 連接配置：

#### 🪟 Windows/本機環境（如 Jupyter Notebook）

在導入模組前設定環境變數：

```python
import os
os.environ['COMFYUI_HOST'] = '127.0.0.1'
os.environ['COMFYUI_PORT'] = '8188'

# 然後正常導入和使用
from examples.quick_draw.helpers import FlexibleGenerator
generator = FlexibleGenerator()
```

#### 🐳 Docker 環境

無需額外設定，預設使用 `host.docker.internal:8188`

#### 🔧 自定義連接

也可以在需要時手動指定：

```python
from lib.comfyui.websockets_api import ComfyUICommunicator

# 連接到自定義地址
communicator = ComfyUICommunicator(host='192.168.1.100', port=8188)
```

**配置優先順序**：
1. 明確傳入的參數（`host`, `port`）
2. 環境變數（`COMFYUI_HOST`, `COMFYUI_PORT`）
3. 預設值（`host.docker.internal`, `8188`）

### 預設路徑

範例使用以下預設路徑：

- 工作流: `configs/workflow/`
- 輸出: `output_media/`
- 環境變數: `media_overload.env`

### 自定義路徑

可以在初始化時自定義路徑：

```python
generator = FlexibleGenerator(
    workflow_folder='your/workflow/path',
    output_folder='your/output/path',
    env_path='your/env/path'
)
```

## 📖 進階用法

### 批次生成

```python
from examples.quick_draw.helpers import FlexibleGenerator

generator = FlexibleGenerator()

# 批次生成不同主題
prompts = [
    {"keywords": ["morning", "sunrise"], "style": "bright"},
    {"keywords": ["afternoon", "tea"], "style": "warm"},
    {"keywords": ["night", "stars"], "style": "mystical"}
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

### Image to Image 生成

```python
from examples.quick_draw.helpers import ConfigBuilder
from examples.simple_content_service import SimpleContentGenerationService

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

### 使用不同的系統提示詞

```python
# 黑色幽默風格
result = generator.generate_images(
    keywords="sleeping peacefully",
    system_prompt="black_humor_system_prompt",
    character="kirby",
    num_images=2
)

# 電影級別風格
result = generator.generate_images(
    keywords=["epic battle", "heroic"],
    system_prompt="cinematic_stable_diffusion_prompt",
    character="kirby",
    num_images=2
)

# 佛性風格
result = generator.generate_images(
    keywords=["meditation", "zen"],
    system_prompt="buddhist_combined_image_system_prompt",
    character="kirby",
    num_images=2
)
```

可用的系統提示詞：
- `stable_diffusion_prompt` - 標準
- `two_character_interaction_generate_system_prompt` - 雙角色互動
- `buddhist_combined_image_system_prompt` - 佛性風格
- `black_humor_system_prompt` - 黑色幽默
- `cinematic_stable_diffusion_prompt` - 電影級別
- `warm_scene_description_system_prompt` - 溫馨場景
- `unbelievable_world_system_prompt` - 不可思議的世界

## 🐛 常見問題

### Q: 範例執行失敗怎麼辦？

A: 檢查以下項目：
1. 環境變數是否正確載入
2. 資料庫連接是否正常
3. ComfyUI 是否已啟動
4. 工作流文件是否存在
5. (Windows 環境) 是否設定了 `COMFYUI_HOST='127.0.0.1'`

### Q: 如何查看生成的圖片？

A: 圖片預設保存在 `output_media/` 目錄，可以通過返回值的 `media_files` 欄位查看路徑：

```python
result = generator.generate_images(...)
for img_path in result['media_files']:
    print(img_path)
```

### Q: 如何修改工作流？

A: 將工作流 JSON 文件放在 `configs/workflow/` 目錄，然後在生成時指定：

```python
result = generator.generate_images(
    keywords=["adventure"],
    character="Kirby",
    workflow='your_workflow_name',  # 不含 .json
    system_prompt='stable_diffusion_prompt'
)
```

### Q: Windows 環境中無法連接 ComfyUI？

A: 確保在導入模組前設定環境變數：

```python
import os
os.environ['COMFYUI_HOST'] = '127.0.0.1'

# 然後再導入
from examples.quick_draw.helpers import FlexibleGenerator
```

### Q: 我需要完整的分析功能怎麼辦？

A: 使用完整版的 ContentGenerationService：

```python
from lib.services.implementations.content_generation_service import ContentGenerationService

service = ContentGenerationService()
result = service.generate_content(config)  # 包含分析和文章生成
```

### Q: 如何自定義 system_prompt？

A: System prompts 定義在 `configs/prompt/image_system_guide.py` 中。您可以：

1. 使用現有的 system_prompt
2. 在該文件中添加新的 system_prompt
3. 在生成時指定新的 system_prompt 名稱

## 📝 相關文檔

- [所有策略完整範例](all_strategies_examples.ipynb) ⭐⭐
- [Quick Draw 詳細說明](quick_draw/README.md)
- [專案主 README](../README.md)

## 🤝 貢獻

歡迎提交新的範例！請確保：

1. 使用簡化服務以保持範例快速
2. 添加清晰的註釋和文檔
3. 包含使用說明

## 📄 授權

MIT License
