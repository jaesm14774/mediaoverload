# MediaOverload

AI-powered automated content generation and multi-platform publishing system.

Generates diverse media content (images/videos) from text prompts using LLMs and ComfyUI, with automated social media publishing to Instagram and Twitter.

---

## Quick Start

**Fastest way to test:**
```bash
# Clone and setup
git clone https://github.com/your-repo/mediaoverload.git
cd mediaoverload
cp media_overload.env.example media_overload.env

# Start services
docker-compose up --build -d

# Generate content
python run_media_interface.py --character kirby --prompt "Kirby eating ramen"
```

**Try examples first:**
```bash
# Interactive notebook (recommended)
jupyter notebook examples/quick_draw_examples.ipynb

# Or run Python script
python examples/quick_draw_example.py
```

Examples skip time-consuming analysis steps - perfect for testing.

---

## What It Does

MediaOverload automates content creation from start to finish:

**Input** → Text prompt or news keyword
**Process** → AI generates descriptions → Creates images/videos → Analyzes quality
**Output** → Discord review → Auto-publish to Instagram/Twitter

### Core Features

**Smart Content Generation**
- Text-to-image, image-to-image, text-to-video workflows
- Text-to-image-to-video workflow (使用者選擇圖片後生成含音頻的影片，不做 AI 篩選)
- Multi-model support: Ollama, Gemini, OpenRouter
- ComfyUI integration with multiple workflows (Flux, SDXL, Wan2.2, etc.)

**Character System**
- Each character has unique style, workflows, and social accounts
- Group-based random character selection
- Two-character interaction scenes

**Quality Control**
- Vision model analyzes image-text matching (可選，Text2Image2Video 策略不使用)
- Discord-based human review workflow
- Automatic filtering by similarity threshold (Text2Image2Video 策略改為使用者手動選擇)

**Multi-Platform Publishing**
- Instagram with automatic format conversion
- Twitter with API v2 support
- Extensible platform architecture

---

## Documentation

**Getting Started**
- [Installation Guide](docs/installation.md) - Setup dependencies and services
- [Configuration Guide](docs/configuration.md) - Character configs and credentials
- [Quick Examples](examples/README.md) - 6 ready-to-run use cases

**Deep Dive**
- [Architecture Overview](docs/architecture.md) - System design and workflows
- [API Reference](docs/api.md) - Service interfaces and methods
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

---

## Key Concepts

### Character Configuration

Characters drive content generation. Each has a YAML config:

```yaml
character:
  name: kirby
  group_name: Kirby

generation:
  generation_type_weights:
    text2img: 0.6        # 60% probability
    text2image2image: 0.4  # 40% two-stage generation

  workflows:
    text2img: /app/configs/workflow/nova-anime-xl.json

  similarity_threshold: 0.7  # Quality filter
```

### Generation Strategies

**Text-to-Image**
Direct text → image generation using ComfyUI workflows.

**Image-to-Image**
Transform existing images with controllable denoising (0.5-0.7).

**Text→Image→Image (Two-Stage)**
1. Generate multiple images from text
2. AI selects best matches
3. Refine selected images with image-to-image

**Text-to-Video**
Generate videos with MMAudio sound effects.

**Text→Image→Video (Multi-Stage)**
1. Generate images from text (使用策略專用配置：固定 minimal, simple background style，不接受雙角色互動)
2. Generate article content based on images + descriptions (在第一階段圖片生成後立即生成)
3. Send all images to Discord for user review (預設為非勾選狀態，使用者檢核後才勾選)
4. User selects images via Discord
5. Generate video descriptions from selected images
6. Generate audio descriptions from images + video descriptions
7. Generate videos with audio using wan2.2_gguf_i2v workflow (每個影片使用不同的 seed)
8. Upload to social media

---

## System Requirements

**Required Services**
- ComfyUI (port 8188) - Image/video generation
- Ollama or cloud LLM - Text generation and analysis
- MySQL/PostgreSQL - Character and news data
- Discord - Review workflow

**Optional Services**
- Google Gemini API - Alternative LLM
- OpenRouter API - Access to free models

**Resources**
- GPU with 8GB+ VRAM recommended for video generation
- Storage for generated media and logs

---

## Development

**Add New Character**
1. Create `configs/characters/{name}.yaml`
2. Setup `configs/social_media/credentials/{name}/`
3. Configure platforms in YAML

**Add New Platform**
1. Implement platform class in `lib/social_media.py`
2. Register in `PublishingService`
3. Update character configs

**Custom ComfyUI Workflow**
1. Design workflow in ComfyUI
2. Export as JSON to `configs/workflow/`
3. Reference in character config

---

## Recent Updates

**v2.4.0** (Workflow 檢查)
- **nova-anime-xl.json workflow 變更檢查**：
  - ✅ 節點 ID 改變（237/240/241 → 260/269/272）不影響功能，系統使用動態節點查找
  - ✅ 結構改變（新增 LoraLoader 節點、LoRA 強度調整）不影響 text2img 功能
  - ⚠️ **重要發現**：workflow 包含三個獨立的生成流程，每次執行會生成 3 張圖片
    - 流程 1: noobaiXLNAIXL_vPred10Version + reiXL_NB11 LoRA (model0)
    - 流程 2: noobaiXLNAIXL_vPred10Version + reiXL_NB11 LoRA (model1)
    - 流程 3: novaAnimeXL_ilV60 + reiXL_NB11 LoRA (model3)
  - 📊 **影響**：如果 `images_per_description` 設為 8，實際會生成 8 × 3 = 24 張圖片
  - 💡 **建議**：如需單一圖片輸出，考慮修改 workflow 只保留一個生成流程，或調整 `images_per_description` 配置

**v2.3.2**
- **文章內容生成優化**：
  - 限制生成文章內容時最多使用3張圖片（而非全部圖片）
  - 減少 API 調用成本，提升生成效率
  - 適用於所有生成策略（Text2Image、Image2Image、Text2Image2Video 等）

**v2.3.1**
- **Text2Image2Video 策略修復**：
  - 修復影片生成後未生成基於影片的文章內容的問題
  - 修復影片審核時使用錯誤文章內容的問題（現在使用基於影片的內容）
  - 修復發布時未使用正確文章內容的問題（優先使用基於影片的內容）
  - 確保在影片生成後，會重新生成基於影片的文章內容並發送到 Discord 和社群媒體

**v2.3.0**
- **Text2Image2Video 策略優化**：
  - 在第一階段圖片生成後立即生成發文內文（使用圖 + 描述）
  - Discord 選擇預設為非勾選狀態，使用者檢核後才勾選可用圖片
  - 修正影片生成 seed 問題，確保每個影片使用不同的 seed
  - 支援策略專用配置：text2image2video 的 text2image 階段可設定固定 style 為 "minimalism style with pure background"
  - 強制不使用雙角色互動系統提示詞，確保背景乾淨簡單

**v2.2.0**
- **架構重構**：將業務邏輯從 `orchestration_service` 移回策略層，遵循單一職責原則
- **Text2Image2Video 策略優化**：
  - 移除 AI 圖片篩選步驟，改為透過 Discord 讓使用者手動選擇圖片
  - 節省 AI 分析成本，提升使用者控制權
  - 文章內容延遲生成：在影片生成後才生成文章內容（而非圖片階段）
- **策略介面擴展**：
  - 新增 `needs_user_review()` 方法：策略可指示是否需要使用者審核
  - 新增 `get_review_items(max_items)` 方法：策略提供審核項目（處理 Discord 10 張限制）
  - 新增 `continue_after_review(selected_indices)` 方法：策略處理使用者選擇後的後續流程
  - 新增 `should_generate_article_now()` 方法：策略控制文章內容生成時機

**v2.1.0**
- OpenRouter integration with free models
- Two-character interaction system
- New prompt templates (spiritual, dark humor, cinematic)
- Text→Image→Image two-stage generation
- Improved vision model analysis

---

## Project Structure

```
mediaoverload/
├── configs/           # Character and workflow configs
├── lib/              # Core services and strategies
├── examples/         # Ready-to-run examples
├── scheduler/        # Automated scheduling
└── docs/            # Detailed documentation
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Support

- **Documentation**: See `docs/` folder
- **Examples**: Run `examples/quick_draw_example.py`
- **Issues**: GitHub Issues
- **Discord**: Review bot setup in installation guide

---

**Note:** This README provides a high-level overview. See `docs/` for detailed guides on installation, configuration, and troubleshooting.
