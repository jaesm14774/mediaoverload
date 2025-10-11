# MediaOverload 使用範例

本目錄包含 MediaOverload 專案的各種使用範例，幫助您快速上手和理解系統功能。

## 📁 目錄結構

```
examples/
├── README.md                          # 本文件
├── simple_content_service.py          # 簡化的內容生成服務
├── quick_draw_example.py              # 快速範例執行腳本
├── quick_draw_examples.ipynb          # Jupyter Notebook 範例 ⭐
└── quick_draw/                        # Quick Draw 模組化範例
    ├── README.md                      # Quick Draw 詳細說明
    ├── use_cases/                     # 使用案例模組
    │   ├── __init__.py
    │   ├── base_use_case.py          # 基類
    │   ├── single_character.py       # 單角色生成
    │   ├── character_interaction.py  # 雙角色互動
    │   ├── news_based.py             # 基於新聞
    │   ├── buddhist_style.py         # 佛性風格
    │   ├── black_humor.py            # 黑色幽默
    │   └── cinematic.py              # 電影級別
    └── helpers/                       # 輔助工具
        ├── __init__.py
        ├── config_builder.py         # 配置建構器
        └── workflow_loader.py        # 工作流載入器
```

## 🚀 快速開始

### 方式 1: Jupyter Notebook（推薦）

如果您有 Jupyter 環境，可以直接使用 notebook：

```bash
jupyter notebook examples/quick_draw_examples.ipynb
```

**優點**：
- 📊 互動式環境，可以直接看到結果
- 🖼️ 自動顯示生成的圖片
- 📝 包含完整的說明和註解
- ⚡ 可以逐步執行，方便學習

### 方式 2: 運行範例腳本

最簡單的方式是運行範例腳本：

```bash
python examples/quick_draw_example.py
```

這會啟動一個互動式選單，讓您選擇要運行的範例。

### 在程式碼中使用

#### 1. 單角色圖片生成

```python
from examples.quick_draw.use_cases import SingleCharacterUseCase

use_case = SingleCharacterUseCase()
result = use_case.execute(
    character='Kirby',
    topic='peaceful sleeping',
    images_per_description=2
)
```

#### 2. 雙角色互動

```python
from examples.quick_draw.use_cases import CharacterInteractionUseCase

use_case = CharacterInteractionUseCase()
result = use_case.execute(
    main_character='Kirby',
    secondary_character='Waddle Dee',
    topic='friendship'
)
```

#### 3. 使用簡化服務

```python
from examples.simple_content_service import SimpleContentGenerationService
from examples.quick_draw.helpers import ConfigBuilder

# 建立配置
config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_workflow('configs/workflow/nova-anime-xl.json') \
    .with_output_dir('output_media') \
    .with_prompt('your prompt here') \
    .build()

# 使用簡化服務（跳過分析和文章生成）
service = SimpleContentGenerationService()
result = service.generate_content(config)
```

## 📚 範例說明

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
- 快速原型製作
- 需要人工檢查生成結果

### Quick Draw 模組

**目錄**: `quick_draw/`

完整的模組化範例系統，提供 6 種使用案例：

1. **SingleCharacterUseCase** - 單角色圖片生成
   - 為單一角色生成基於特定主題的圖片
   
2. **CharacterInteractionUseCase** - 雙角色互動
   - 生成兩個角色互動的場景
   
3. **NewsBasedUseCase** - 基於新聞關鍵字
   - 根據最新新聞生成相關圖片
   
4. **BuddhistStyleUseCase** - 佛性/靈性風格
   - 融合宗教/靈性元素的圖片
   
5. **BlackHumorUseCase** - 黑色幽默
   - 具有諷刺意味的黑色幽默圖片
   
6. **CinematicUseCase** - 電影級別
   - 電影感的寬螢幕比例圖片

詳細說明請參考 [quick_draw/README.md](quick_draw/README.md)

## 🆚 範例版 vs 完整版

| 功能 | 範例版 (Quick Draw) | 完整版 |
|------|-------------------|--------|
| 描述生成 | ✅ | ✅ |
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
2. **資料庫**: MySQL 資料庫包含角色和新聞資料
3. **ComfyUI**: 已啟動並可訪問

### 預設路徑

範例使用以下預設路徑：

- 工作流: `configs/workflow/`
- 輸出: `output_media/`
- 環境變數: `media_overload.env`

### 自定義路徑

可以在初始化時自定義路徑：

```python
use_case = SingleCharacterUseCase(
    workflow_folder='your/workflow/path',
    output_folder='your/output/path',
    env_path='your/env/path'
)
```

## 📖 進階用法

### 批次生成

```python
from examples.quick_draw.use_cases import CharacterInteractionUseCase

use_case = CharacterInteractionUseCase()
results = use_case.execute_batch(
    main_character='Kirby',
    batch_size=10  # 生成 10 組
)
```

### 自定義圖片尺寸

```python
from examples.quick_draw.use_cases import CinematicUseCase

use_case = CinematicUseCase()
result = use_case.execute(
    main_character='Kirby',
    custom_size=(2048, 1024)  # 自定義 2:1 比例
)
```

### 使用不同的系統提示詞

```python
from examples.quick_draw.helpers import ConfigBuilder

config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_image_system_prompt('black_humor_system_prompt') \
    .build()
```

可用的系統提示詞：
- `stable_diffusion_prompt` - 標準
- `two_character_interaction_generate_system_prompt` - 雙角色互動
- `buddhist_combined_image_system_prompt` - 佛性風格
- `black_humor_system_prompt` - 黑色幽默
- `cinematic_stable_diffusion_prompt` - 電影級別

## 🐛 常見問題

### Q: 範例執行失敗怎麼辦？

A: 檢查以下項目：
1. 環境變數是否正確載入
2. 資料庫連接是否正常
3. ComfyUI 是否已啟動
4. 工作流文件是否存在

### Q: 如何查看生成的圖片？

A: 圖片預設保存在 `output_media/` 目錄，可以通過返回值的 `media_files` 欄位查看路徑：

```python
result = use_case.execute(...)
for img_path in result['media_files']:
    print(img_path)
```

### Q: 如何修改工作流？

A: 將工作流 JSON 文件放在 `configs/workflow/` 目錄，然後在執行時指定：

```python
result = use_case.execute(
    workflow_name='your_workflow_name'  # 不含 .json
)
```

### Q: 我需要完整的分析功能怎麼辦？

A: 使用完整版的 ContentGenerationService：

```python
from lib.services.implementations.content_generation_service import ContentGenerationService

service = ContentGenerationService()
result = service.generate_content(config)  # 包含分析和文章生成
```

## 📝 相關文檔

- [Quick Draw 詳細說明](quick_draw/README.md)
- [專案主 README](../README.md)

## 🤝 貢獻

歡迎提交新的範例！請確保：

1. 使用簡化服務以保持範例快速
2. 添加清晰的註釋和文檔
3. 包含使用說明

## 📄 授權

MIT License

