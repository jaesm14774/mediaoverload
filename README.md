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
- Multi-model support: Ollama, Gemini, OpenRouter
- ComfyUI integration with multiple workflows (Flux, SDXL, etc.)

**Character System**
- Each character has unique style, workflows, and social accounts
- Group-based random character selection
- Two-character interaction scenes

**Quality Control**
- Vision model analyzes image-text matching
- Discord-based human review workflow
- Automatic filtering by similarity threshold

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
