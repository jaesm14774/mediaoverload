# Jupyter Notebook 使用指南

本指南說明如何使用 `quick_draw_examples.ipynb` 來快速體驗 MediaOverload 的圖片生成功能。

## 📚 Notebook 內容

Notebook 包含以下內容：

### 1. 環境設定
- 自動配置 Python 路徑
- 導入所有必要的模組

### 2. 六種使用案例
1. **單角色圖片生成** - 為單一角色生成主題圖片
2. **雙角色互動** - 生成兩個角色的互動場景
3. **基於新聞關鍵字** - 結合最新新聞生成圖片
4. **佛性/靈性風格** - 融合宗教元素的圖片
5. **黑色幽默** - 諷刺意味的黑色幽默圖片
6. **電影級別** - 寬螢幕電影感圖片

### 3. 圖片顯示
- 使用 IPython.display 自動顯示生成的圖片
- 方便即時查看結果

### 4. 總結說明
- 使用案例對比
- 相關文檔連結

## 🚀 如何使用

### 前置需求

1. ✅ 安裝 Jupyter Notebook 或 JupyterLab：
   ```bash
   pip install jupyter
   # 或
   pip install jupyterlab
   ```

2. ✅ 環境變數已配置（`media_overload.env`）
3. ✅ 資料庫已連接
4. ✅ ComfyUI 已啟動

### 啟動 Notebook

```bash
# 在專案根目錄執行
jupyter notebook examples/quick_draw_examples.ipynb

# 或使用 JupyterLab
jupyter lab examples/quick_draw_examples.ipynb
```

### 執行步驟

1. **依序執行所有 cells**
   - 按 `Shift + Enter` 執行當前 cell 並移動到下一個
   - 或點擊上方的 "Run" 按鈕

2. **環境設定** (前 2 個 cells)
   - 第一個 cell：配置 Python 路徑
   - 第二個 cell：導入所有使用案例

3. **選擇使用案例**
   - 執行您感興趣的使用案例
   - 每個使用案例都是獨立的，可以單獨執行

4. **查看結果**
   - 執行「顯示生成的圖片」cell
   - 圖片會自動顯示在 notebook 中

## 💡 使用技巧

### 1. 調整參數

可以直接修改 cell 中的參數：

```python
result = single_char.execute(
    character='Kirby',        # 改變角色
    topic='sleeping',         # 改變主題
    images_per_description=2  # 改變生成數量
)
```

### 2. 重複執行

- 可以多次執行同一個 cell
- 每次執行都會生成新的圖片

### 3. 查看輸出

notebook 會顯示：
- ✅ 生成進度信息
- ✅ 描述內容
- ✅ 圖片路徑
- ✅ 圖片預覽

### 4. 保存結果

生成的圖片會保存在 `output_media/` 目錄，可以隨時查看。

## 🎨 顯示圖片

Notebook 中包含自動顯示圖片的 cell：

```python
from IPython.display import Image, display
from pathlib import Path

# 顯示最近生成的圖片
if result['media_files']:
    print("📸 生成的圖片:")
    for img_path in result['media_files'][:3]:
        if Path(img_path).exists():
            print(f"\n{Path(img_path).name}")
            display(Image(filename=img_path, width=600))
```

## 🔧 常見問題

### Q: 圖片無法顯示？

A: 檢查：
1. 圖片路徑是否正確
2. 圖片是否已生成
3. 嘗試重新執行生成 cell

### Q: 執行速度慢？

A: 這是正常的：
- 描述生成需要調用 AI 模型
- 圖片生成需要 ComfyUI 處理
- 第一次執行可能較慢（模型載入）

### Q: 如何加快測試？

A: 減少生成數量：
```python
result = use_case.execute(
    character='Kirby',
    images_per_description=1  # 只生成 1 張
)
```

### Q: 如何使用不同的工作流？

A: 指定 `workflow_name` 參數：
```python
result = use_case.execute(
    character='Kirby',
    workflow_name='flux_krea_dev'  # 使用不同工作流
)
```

## 📊 與 Python 腳本的對比

| 特點 | Jupyter Notebook | Python 腳本 |
|------|-----------------|------------|
| 互動性 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 圖片顯示 | ✅ 自動顯示 | ❌ 需要手動開啟 |
| 逐步執行 | ✅ | ❌ |
| 學習友好 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 適合場景 | 學習、測試 | 批次處理 |

## 🎓 學習路徑

建議的學習順序：

1. **第一步**: 執行「單角色圖片生成」
   - 最簡單的使用案例
   - 了解基本流程

2. **第二步**: 嘗試「雙角色互動」
   - 了解角色配對機制
   - 學習參數調整

3. **第三步**: 探索其他使用案例
   - 新聞、佛性、黑色幽默、電影級別
   - 了解不同風格的差異

4. **第四步**: 查看代碼實現
   - 打開 `examples/quick_draw/use_cases/` 查看源碼
   - 了解如何自定義使用案例

## 📖 相關資源

- [examples/README.md](README.md) - 範例總覽
- [quick_draw/README.md](quick_draw/README.md) - Quick Draw 詳細說明
- [quick_draw/QUICK_START.md](quick_draw/QUICK_START.md) - 5 分鐘快速入門
- [主 README.md](../README.md) - 專案主文檔

## 💬 反饋與建議

如果您在使用 notebook 時遇到任何問題或有改進建議，歡迎：
- 提交 Issue
- 發起 Pull Request
- 聯繫維護團隊

---

**提示**: Jupyter Notebook 是學習和測試的最佳方式！享受互動式的開發體驗吧！ 🎉

