# Quick Draw - 快速圖片生成範例

基於 mediaoverload 框架的模組化圖片生成工具，專為快速範例和測試設計。

## ✨ 特點

- **🎯 簡化流程** - 跳過耗時的圖文匹配分析和文章生成
- **⚡ 快速執行** - 專注於圖片生成，適合人工審核
- **🔄 模組化設計** - 每個使用案例獨立、易於維護
- **📦 完全相容** - 使用現有的 mediaoverload 基礎設施

## 🚀 快速開始

### 方式 1: Jupyter Notebook（推薦）

```bash
# 在專案根目錄執行
jupyter notebook examples/quick_draw_examples.ipynb
```

Notebook 包含：
- ✅ 完整的 6 種使用案例
- ✅ 互動式環境
- ✅ 自動顯示生成的圖片
- ✅ 詳細的說明和註解

### 方式 2: Python 腳本

```bash
python examples/quick_draw_example.py
```

### 方式 3: 在代碼中使用

#### 1. 單角色圖片生成

```python
from examples.quick_draw.use_cases import SingleCharacterUseCase

# 創建使用案例
use_case = SingleCharacterUseCase()

# 執行生成
result = use_case.execute(
    character='Kirby',
    topic='peaceful sleeping',
    style='minimalist style',
    images_per_description=2
)

print(f"生成了 {len(result['media_files'])} 張圖片")
print(f"圖片路徑: {result['media_files']}")
```

### 2. 雙角色互動

```python
from examples.quick_draw.use_cases import CharacterInteractionUseCase

use_case = CharacterInteractionUseCase()
result = use_case.execute(
    main_character='Kirby',
    secondary_character='Waddle Dee',
    topic='friendship',
    images_per_description=2
)
```

### 3. 基於新聞關鍵字

```python
from examples.quick_draw.use_cases import NewsBasedUseCase

use_case = NewsBasedUseCase()
result = use_case.execute(
    character='Kirby',
    news_count=3,  # 使用 3 條新聞
    images_per_description=2
)

print(f"處理了 {result['total_news']} 條新聞")
```

### 4. 佛性/靈性風格

```python
from examples.quick_draw.use_cases import BuddhistStyleUseCase

use_case = BuddhistStyleUseCase()
result = use_case.execute(
    character='Kirby',
    spiritual_theme='meditation',
    use_news=True,
    images_per_description=2
)
```

### 5. 黑色幽默

```python
from examples.quick_draw.use_cases import BlackHumorUseCase

use_case = BlackHumorUseCase()
result = use_case.execute(
    main_character='Kirby',
    secondary_character='Waddle Dee',
    images_per_description=2
)
```

### 6. 電影級別

```python
from examples.quick_draw.use_cases import CinematicUseCase

use_case = CinematicUseCase()
result = use_case.execute(
    main_character='Kirby',
    aspect_ratio='cinematic',  # 16:9 (1280x720)
    use_news=True,
    images_per_description=2
)
```

## 📋 可用的使用案例

| 使用案例 | 說明 | 適用場景 |
|---------|------|---------|
| `SingleCharacterUseCase` | 單角色圖片生成 | 為指定角色生成主題圖片 |
| `CharacterInteractionUseCase` | 雙角色互動 | 生成兩個角色互動場景 |
| `NewsBasedUseCase` | 基於新聞關鍵字 | 根據最新新聞生成相關圖片 |
| `BuddhistStyleUseCase` | 佛性/靈性風格 | 融合宗教/靈性元素 |
| `BlackHumorUseCase` | 黑色幽默 | 諷刺意味的黑色幽默圖片 |
| `CinematicUseCase` | 電影級別 | 電影感的寬螢幕圖片 |

## 🔧 配置說明

### 環境需求

1. 確保 `media_overload.env` 已正確配置
2. MySQL 資料庫中有角色和新聞資料
3. ComfyUI 已啟動並可訪問

### 預設路徑

- **工作流**: `configs/workflow/`
- **輸出**: `output_media/`
- **環境變數**: `media_overload.env`

### 自定義路徑

```python
use_case = SingleCharacterUseCase(
    workflow_folder='your/workflow/path',
    output_folder='your/output/path',
    env_path='your/env/path'
)
```

## ⚙️ 進階用法

### 使用 ConfigBuilder

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

### 批次生成

```python
# 批次生成多組互動圖片
use_case = CharacterInteractionUseCase()
results = use_case.execute_batch(
    main_character='Kirby',
    batch_size=5,
    images_per_description=2
)
```

## 📊 返回值結構

所有使用案例返回相同的結構：

```python
{
    'descriptions': List[str],      # 生成的描述列表
    'media_files': List[str],       # 生成的圖片路徑列表
    'filter_results': [],           # 空列表（已跳過分析）
    'article_content': ''           # 空字串（已跳過文章生成）
}
```

## 🆚 與完整版的差異

| 功能 | Quick Draw (範例版) | 完整版 |
|------|-------------------|--------|
| 描述生成 | ✅ | ✅ |
| 圖片/視頻生成 | ✅ | ✅ |
| 圖文匹配分析 | ❌ 跳過 | ✅ |
| 文章內容生成 | ❌ 跳過 | ✅ |
| Hashtag 生成 | ❌ 跳過 | ✅ |
| 執行速度 | **快速** | 較慢 |
| 適用場景 | 快速測試、人工審核 | 自動化發布 |

## 💡 為什麼跳過分析和文章生成？

1. **節省時間** - 圖文匹配分析和 hashtag 生成需要大量時間
2. **人工審核** - 範例用途，通常需要人工檢查結果
3. **快速迭代** - 專注於圖片生成，加快測試速度
4. **靈活性** - 生成後可以手動決定後續處理

## 📖 相關文檔

- [架構說明](../../tmp/quick_draw/ARCHITECTURE.md)
- [遷移指南](../../tmp/quick_draw/MIGRATION_GUIDE.md)
- [專案總結](../../tmp/quick_draw/PROJECT_SUMMARY.md)

## 🐛 常見問題

### Q: 如何切換不同的工作流？

A: 在 `execute()` 中指定 `workflow_name` 參數：

```python
result = use_case.execute(
    character='Kirby',
    workflow_name='flux_krea_dev',  # 使用不同的工作流
    topic='adventure'
)
```

### Q: 如何調整圖片數量？

A: 使用 `images_per_description` 參數：

```python
result = use_case.execute(
    character='Kirby',
    images_per_description=5  # 每個描述生成 5 張圖片
)
```

### Q: 如何使用完整版的功能？

A: 使用 `lib/services/implementations/content_generation_service.py` 中的 `ContentGenerationService`：

```python
from lib.services.implementations.content_generation_service import ContentGenerationService

service = ContentGenerationService()
result = service.generate_content(config)  # 包含完整分析和文章生成
```

## 📝 授權

MIT License

