# WebSocket 連接問題修復說明

## 問題描述

之前的實現存在以下問題：

1. **雙重 WebSocket 連接管理**：
   - 策略層（`generate_strategies.py`）創建並管理 WebSocket 連接
   - 通訊層（`websockets_api.py`）的 `process_workflow` 方法也會重新創建連接
   - 導致連接混亂和資源浪費

2. **過早關閉連接**：
   - 在生成多張圖片時，每次都會關閉並重新建立 WebSocket 連接
   - 導致後續的圖片生成失敗或超時

3. **缺乏調試信息**：
   - `wait_for_completion` 方法缺少進度和狀態信息
   - 當出現問題時，用戶只能等待 900 秒超時，無法知道發生了什麼

## 解決方案

### 1. WebSocket 連接管理改進

#### `websockets_api.py` 的改進：

- **新增 `auto_close` 參數**：
  ```python
  def process_workflow(self, workflow, updates, output_path, file_name=None, auto_close=True)
  ```
  - 允許控制是否自動關閉 WebSocket 連接
  - 預設為 `True` 以保持向後兼容性

- **智能連接檢查**：
  ```python
  if not self.ws or not self.ws.connected:
      print("建立新的 WebSocket 連線")
      self.connect_websocket()
  ```
  - 只在 WebSocket 未連接時才建立新連線
  - 避免重複連接

- **增強的等待機制**：
  - 添加詳細的進度顯示
  - 支援多種消息類型（executing, progress, status, execution_error）
  - 更好的錯誤處理和超時檢測
  - 顯示當前處理的節點和進度百分比

#### `generate_strategies.py` 的改進：

**Text2ImageStrategy（圖片生成）**：
```python
def generate_media(self):
    self.communicator = ComfyUICommunicator()
    
    try:
        # 僅建立一次 WebSocket 連接
        self.communicator.connect_websocket()
        
        for idx, description in enumerate(self.descriptions):
            for i in range(images_per_description):
                # 處理工作流，但不關閉連接
                self.communicator.process_workflow(
                    workflow=workflow,
                    updates=updates,
                    output_path=f"{self.config.output_dir}",
                    file_name=f"{self.config.character}_d{idx}_{i}",
                    auto_close=False  # 保持連接
                )
    finally:
        # 所有圖片生成完成後才關閉連接
        if self.communicator and self.communicator.ws and self.communicator.ws.connected:
            self.communicator.ws.close()
```

**Text2VideoStrategy（視頻生成）**：
- 採用與圖片生成相同的改進策略
- 在所有視頻生成完成後才關閉 WebSocket 連接

### 2. 使用者體驗改進

- **進度顯示**：
  ```
  [1/4] 為描述 1/1，生成第 1/4 張圖片
  開始等待工作流 abc-123 完成...
    → 正在處理節點: 5
    → 進度: 45.5% (10/22)
  ✓ 工作流 abc-123 執行完成（耗時 12.34 秒）
  ```

- **錯誤追蹤**：
  - 顯示最後處理的節點
  - 顯示詳細的錯誤信息
  - 添加超時警告

- **總耗時統計**：
  ```
  ✅ 生成圖片總耗時: 45.67 秒
  ```

## 效能提升

### 修復前：
- 每張圖片都需要建立和關閉 WebSocket 連接
- 連接開銷：~1-2 秒/次
- 生成 4 張圖片總開銷：~4-8 秒

### 修復後：
- 整個批次只需建立一次 WebSocket 連接
- 連接開銷：~1-2 秒/批次
- 生成 4 張圖片總開銷：~1-2 秒

**預期效能提升**：對於生成多張圖片的場景，可節省 50-75% 的連接時間。

## 向後兼容性

- `auto_close` 參數預設為 `True`，保持原有行為
- 現有的單次調用程式碼無需修改
- 只有批次處理場景需要使用新的連接管理策略

## 使用建議

### 批次生成（推薦）：
```python
# 策略層會自動管理連接
service = SimpleContentGenerationService(...)
result = service.generate_content(config)  # 自動處理多張圖片
```

### 單次生成（也支援）：
```python
communicator = ComfyUICommunicator()
communicator.process_workflow(...)  # auto_close=True（預設）
```

### 手動管理連接（高級用法）：
```python
communicator = ComfyUICommunicator()
communicator.connect_websocket()

try:
    for i in range(10):
        communicator.process_workflow(..., auto_close=False)
finally:
    if communicator.ws and communicator.ws.connected:
        communicator.ws.close()
```

## 測試建議

1. **測試批次生成**：
   ```bash
   python examples/quick_draw_example.py
   # 選擇任一範例，觀察進度顯示
   ```

2. **測試錯誤處理**：
   - 故意使用無效的工作流配置
   - 觀察錯誤信息是否清晰

3. **測試長時間運行**：
   - 生成大量圖片或視頻
   - 確認沒有連接洩漏或超時問題

## 故障排除

### 如果仍然遇到超時問題：

1. **檢查 ComfyUI 是否正在運行**：
   ```bash
   curl http://host.docker.internal:8188/system_stats
   ```

2. **增加超時時間**（如果 ComfyUI 處理時間確實很長）：
   ```python
   communicator = ComfyUICommunicator(timeout=1800)  # 30 分鐘
   ```

3. **檢查 ComfyUI 日誌**：
   - 查看 ComfyUI 控制台是否有錯誤信息
   - 確認工作流配置是否正確

4. **啟用詳細日誌**：
   - 修改後的 `wait_for_completion` 會顯示詳細進度
   - 觀察是否卡在特定節點

## 更新日期

2025-10-12

## 相關檔案

- `lib/comfyui/websockets_api.py`
- `lib/media_auto/strategies/generate_strategies.py`
- `examples/simple_content_service.py`

