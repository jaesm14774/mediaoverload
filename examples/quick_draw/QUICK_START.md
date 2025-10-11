# Quick Draw 快速入門指南

5 分鐘快速開始使用 Quick Draw 範例系統！

## 🚀 最快開始

### 步驟 1: 確保環境準備就緒

```bash
# 確保環境變數已配置
ls media_overload.env

# 確保 ComfyUI 已啟動（預設在 8188 端口）
# curl http://localhost:8188
```

### 步驟 2: 運行範例

```bash
# 方式 1: 運行互動式範例（推薦）
python examples/quick_draw_example.py

# 方式 2: 直接在代碼中使用
python -c "
from examples.quick_draw.use_cases import SingleCharacterUseCase
use_case = SingleCharacterUseCase()
result = use_case.execute(character='Kirby', topic='sleeping', images_per_description=2)
print(f'生成了 {len(result[\"media_files\"])} 張圖片')
"
```

## 📝 最簡單的代碼範例

### 1. 單角色生成

```python
from examples.quick_draw.use_cases import SingleCharacterUseCase

# 創建並執行
use_case = SingleCharacterUseCase()
result = use_case.execute(
    character='Kirby',
    topic='peaceful sleeping',
    images_per_description=2
)

# 查看結果
print(f"生成了 {len(result['media_files'])} 張圖片")
for img in result['media_files']:
    print(f"- {img}")
```

### 2. 雙角色互動

```python
from examples.quick_draw.use_cases import CharacterInteractionUseCase

use_case = CharacterInteractionUseCase()
result = use_case.execute(
    main_character='Kirby',
    secondary_character='Waddle Dee',
    topic='friendship'
)
```

### 3. 快速執行（一行代碼）

```python
from examples.quick_draw.use_cases import SingleCharacterUseCase

result = SingleCharacterUseCase.quick_execute(
    character='Kirby',
    topic='adventure'
)
```

## 🎨 6 種使用案例一覽

| 案例 | 導入 | 用途 |
|------|------|------|
| 單角色 | `SingleCharacterUseCase` | 基礎圖片生成 |
| 雙角色 | `CharacterInteractionUseCase` | 角色互動場景 |
| 新聞 | `NewsBasedUseCase` | 結合時事 |
| 佛性 | `BuddhistStyleUseCase` | 靈性風格 |
| 黑色幽默 | `BlackHumorUseCase` | 諷刺風格 |
| 電影級 | `CinematicUseCase` | 寬螢幕圖片 |

## ⚙️ 常用參數

```python
use_case.execute(
    character='Kirby',              # 角色名稱
    topic='your topic',             # 主題
    style='minimalist style',       # 風格
    workflow_name='nova-anime-xl',  # 工作流
    images_per_description=2,       # 每個描述生成幾張圖
    group_name='Kirby'              # 角色群組（可選）
)
```

## 📂 輸出位置

預設輸出到: `output_media/`

可以自定義：

```python
use_case = SingleCharacterUseCase(
    output_folder='my/custom/path'
)
```

## 🔍 查看結果

```python
result = use_case.execute(...)

# 查看描述
print("描述:", result['descriptions'])

# 查看生成的圖片路徑
print("圖片:")
for img in result['media_files']:
    print(f"  - {img}")
```

## ❓ 常見問題

### Q: 為什麼沒有生成圖片？

A: 檢查：
1. ComfyUI 是否在運行
2. 環境變數是否正確
3. 資料庫連接是否正常
4. 查看終端輸出的錯誤信息

### Q: 如何加快生成速度？

A: 減少 `images_per_description` 數量：

```python
result = use_case.execute(
    character='Kirby',
    images_per_description=1  # 只生成 1 張
)
```

### Q: 如何使用不同的工作流？

A: 指定 `workflow_name`：

```python
result = use_case.execute(
    character='Kirby',
    workflow_name='flux_krea_dev'  # 使用不同的工作流
)
```

## 🚀 進階技巧

### 批次生成

```python
use_case = CharacterInteractionUseCase()
results = use_case.execute_batch(
    main_character='Kirby',
    batch_size=5  # 生成 5 組
)
```

### 自定義尺寸（電影級別）

```python
use_case = CinematicUseCase()
result = use_case.execute(
    main_character='Kirby',
    custom_size=(2048, 1024)  # 自定義尺寸
)
```

### 使用 ConfigBuilder

```python
from examples.quick_draw.helpers import ConfigBuilder

config = ConfigBuilder() \
    .with_character('Kirby') \
    .with_prompt('peaceful sleeping') \
    .with_images_per_description(2) \
    .build()
```

## 📖 更多資訊

- [完整 README](README.md) - 詳細說明
- [範例總覽](../README.md) - 所有範例
- [主專案 README](../../README.md) - 專案文檔

---

💡 **提示**: 範例版本跳過了耗時的圖文匹配分析和文章生成，專注於快速生成圖片。如需完整功能，請使用 `ContentGenerationService`。

