# 架構優化遷移指南

## 概述

本次架構優化主要實現了以下目標：
1. **職責分離** - 將原本龐大的 `ContentProcessor` 拆分為多個專門的服務
2. **配置外部化** - 角色配置可以從 YAML 檔案載入，不再需要硬編碼
3. **依賴注入** - 服務之間的依賴透過工廠模式管理
4. **資料庫存取層** - 所有資料庫操作都封裝在 Repository 中

## 新架構組件

### 服務層 (Services)
- **PromptService** - 負責生成提示詞
- **ContentGenerationService** - 負責內容生成（描述、圖片、文章）
- **ReviewService** - 負責 Discord 審核流程
- **PublishingService** - 負責圖片處理和社群媒體發布
- **NotificationService** - 負責成功/失敗通知
- **OrchestrationService** - 協調各服務完成工作流程（新的 ContentProcessor）

### 資料存取層 (Repositories)
- **CharacterRepository** - 處理角色相關的資料庫操作
- **NewsRepository** - 處理新聞相關的資料庫操作

### 配置管理
- **ConfigLoader** - 負責載入和解析 YAML 配置檔案
- **ConfigurableCharacter** - 支援從配置檔案載入的角色基類

## 使用方式

### 方式一：使用配置檔案（推薦）

1. 創建角色配置檔案 `configs/characters/your_character.yaml`：

```yaml
character:
  name: your_character
  group_name: YourGroup
  
generation:
  output_dir: /app/output_image
  workflow_path: /app/configs/workflow/your-workflow.json
  similarity_threshold: 0.7
  generation_type: text2img
  
  prompt_method_weights:
    default: 0.3
    news: 0.7
    
  image_system_prompt_weights:
    default: 0.6
    unbelievable_world_system_prompt: 0.4

social_media:
  default_hashtags:
    - tag1
    - tag2
  platforms:
    instagram:
      config_folder_path: /app/configs/social_media/ig
      enabled: true

additional_params:
  images_per_description: 3
  is_negative: false
```

2. 執行程式：

```bash
python run_media_interface_new.py --config configs/characters/your_character.yaml --prompt "your prompt"
```

### 方式二：向後相容模式

保持原有的角色類別定義，直接使用：

```bash
python run_media_interface_new.py --character Wobbuffet --prompt "your prompt"
```

## 主要變更

### 1. ContentProcessor 重構

舊版：
```python
process = ContentProcessor(character_class())
await process.etl_process(prompt=args.prompt)
```

新版：
```python
service_factory = ServiceFactory()
orchestration_service = service_factory.get_orchestration_service()
await orchestration_service.execute_workflow(character=character, prompt=prompt)
```

### 2. 角色配置外部化

舊版（硬編碼）：
```python
class WobbuffetProcess(BaseCharacter, SocialMediaMixin):
    character = 'wobbuffet'
    output_dir = f'/app/output_image'
    workflow_path = '/app/configs/workflow/nova-anime-xl.json'
    # ... 其他硬編碼配置
```

新版（配置檔案）：
- 所有配置都在 YAML 檔案中定義
- 支援權重隨機選擇（如 prompt_method_weights）
- 易於修改和管理

### 3. 資料庫操作封裝

舊版（直接在 ContentProcessor 中）：
```python
cursor = self.mysql_conn.cursor
query = "SELECT role_name_en FROM anime.anime_roles..."
cursor.execute(query, params)
```

新版（使用 Repository）：
```python
character_repository.get_random_character_from_group(group_name, workflow_name)
```

## 優點

1. **更好的可維護性** - 每個服務都有明確的職責
2. **更容易測試** - 服務之間解耦，可以獨立測試
3. **更靈活的配置** - 不需要修改程式碼就能調整角色設定
4. **更容易擴充** - 新增功能只需要添加新服務或擴充現有服務

## 注意事項

1. 確保所有環境變數都正確設定（`media_overload.env`）
2. YAML 配置檔案的路徑必須正確
3. 新架構仍然向後相容舊的角色類別定義
4. 資料庫連接會在 ServiceFactory 初始化時建立

## 未來擴充建議

1. 可以添加更多的策略類型（除了 text2img）
2. 可以支援更多的社群媒體平台
3. 可以添加快取機制提高效能
4. 可以添加更詳細的錯誤處理和重試機制 