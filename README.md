# MediaOverload

**AI-powered automated content generation and multi-platform publishing system.**

MediaOverload generates diverse media content (images/videos) from text prompts using LLMs and ComfyUI, and automatically publishes them to Instagram and Twitter.

## Table of Contents
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Documentation](#documentation)
- [Key Concepts](#key-concepts)
- [System Requirements](#system-requirements)
- [Development](#development)
- [Recent Updates](#recent-updates)
- [Project Structure](#project-structure)
- [Support](#support)

---

## Quick Start

**Fastest way to test:**

```bash
# 1. Clone and setup
git clone https://github.com/your-repo/mediaoverload.git
cd mediaoverload
cp media_overload.env.example media_overload.env

# 2. Start services
docker-compose up --build -d

# 3. Generate content
python run_media_interface.py --character kirby --prompt "Kirby eating ramen"
```

**Try examples first:**

The examples skip time-consuming analysis steps, making them perfect for testing.

```bash
# Interactive notebook (recommended)
jupyter notebook examples/quick_draw_examples.ipynb

# Or run Python script
python examples/quick_draw_example.py
```

---

## Core Features

MediaOverload automates content creation from start to finish:
**Input** → **Process** → **Output**

### Smart Content Generation
- **Multi-Format Workflows**: Text-to-image, image-to-image, and text-to-video.
- **Text-to-Image-to-Video**: User selects an image to generate a video with audio (no AI filtering).
- **Image Upscaling**: Optional SDXL-based upscaling workflow for enhanced image quality (configurable via environment variables).
- **Multi-Model Support**: Integrates with Ollama, Gemini, and OpenRouter.
- **ComfyUI Integration**: Supports multiple workflows (Flux, SDXL, Wan2.2, etc.).

### Character System
- **Unique Personas**: Each character has its own style, workflows, and social accounts.
- **Group Selection**: Randomly select characters from defined groups.
- **Interactions**: Generate scenes with two characters interacting.

### Quality Control
- **Vision Analysis**: AI analyzes image-text matching (optional).
- **Human Review**: Discord-based workflow for manual approval.
- **Smart Filtering**: Automatic filtering by similarity threshold (manual selection for video strategies).

### Multi-Platform Publishing
- **Instagram**: Automatic format conversion and publishing.
- **Twitter/X**: Full support via API v2.
- **Extensible**: Easy to add new platforms.

---

## Documentation

**Getting Started**
- [Installation Guide](docs/installation.md): Setup dependencies and services.
- [Configuration Guide](docs/configuration.md): Character configs and credentials.
- [Quick Examples](examples/README.md): 6 ready-to-run use cases.

**Deep Dive**
- [Architecture Overview](docs/architecture.md): System design and workflows.
- [Troubleshooting](docs/troubleshooting.md): Common issues and solutions.

---

## Key Concepts

### Character Configuration
Characters drive content generation. Each is defined by a YAML config:

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

- **Text-to-Image**: Direct text → image generation using ComfyUI.
- **Image-to-Image**: Transform existing images with controllable denoising (0.5-0.7).
- **Text→Image→Image (Two-Stage)**: Generate base images, filter by quality, then refine the best matches.
- **Text-to-Video**: Generate videos with MMAudio sound effects.
- **Text→Image→Video (Multi-Stage)**:
    1. **Generate Images**: Uses specific "minimal" style; no character interactions.
    2. **Generate Article**: Created immediately after image generation.
    3. **User Review**: Images sent to Discord; user manually selects the best ones.
    4. **Image Upscaling** (Optional): Selected images are upscaled using SDXL workflow if enabled.
    5. **Video Generation**: Selected (and optionally upscaled) images are converted to video with audio (unique seeds).
    6. **Publish**: Uploads the final video to social media.

---

## System Requirements

**Required Services**
- **ComfyUI** (port 8188): Image/video generation.
- **Ollama** or **Cloud LLM**: Text generation and analysis.
- **MySQL/PostgreSQL**: Character and news data.
- **Discord**: Review workflow.

**Resources**
- **GPU**: 8GB+ VRAM recommended (essential for video).
- **Storage**: Sufficient space for generated media and logs.

---

## Development

### Add New Character
1. Create `configs/characters/{name}.yaml`.
2. Setup `configs/social_media/credentials/{name}/`.
3. Configure platforms in the YAML file.

### Add New Platform
1. Implement the platform class in `lib/social_media/{platform}.py` (inheriting from `SocialMediaPlatform`).
2. Register it in `PublishingService`.
3. Update character configs.
4. Export the new class in `lib/social_media/__init__.py`.

### Custom ComfyUI Workflow
1. Design the workflow in ComfyUI.
2. Export as JSON to `configs/workflow/`.
3. Reference the file in your character config.

---

## Recent Updates

### v2.6.0 (Image Upscaling Support)
- **Upscale Workflow Integration**: Added optional SDXL-based image upscaling workflow.
  - **Text2Img Flow**: Images are upscaled after Discord review, before publishing to social media.
  - **Text2Img2Video Flow**: Images are upscaled after Discord review, before video generation.
  - **Character Config Control**: Enable/disable per character via `enable_upscale` in character YAML config.
  - **Configurable Workflow**: Set custom upscale workflow path per character via `upscale_workflow_path` in config.

### v2.5.0 (Architecture Refactor)
- **Modular Refactoring**:
    - Split `generate_strategies.py` into a dedicated package: `lib/media_auto/strategies/`.
    - Split `social_media.py` into a dedicated package: `lib/social_media/`.
    - Implemented `InstagramPlatform` and `TwitterPlatform` as separate classes.
- **Improved Maintainability**: Cleaner code structure and better separation of concerns.

### v2.3.0 - v2.4.0 (Video & Workflow Optimization)
- **Text2Image2Video**: Optimized workflow with immediate article generation and manual Discord selection.
- **Workflow Checks**: Enhanced validation for ComfyUI workflows (e.g., detecting multi-image outputs).
- **Cost Reduction**: Optimized article generation to use fewer images for context.

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

## Support

- **Documentation**: Check the `docs/` folder.
- **Examples**: Run `examples/quick_draw_example.py`.
- **Issues**: Submit via GitHub Issues.
- **Discord**: Ensure your review bot is set up correctly (see Installation Guide).

---

## Credits

- **Instagram Integration**: The Instagram functionality in this project (`lib/instagram`) is derived from the excellent [instagrapi](https://github.com/subzeroid/instagrapi) library. We have extracted and adapted specific components (login, upload, story) to suit our lightweight needs while maintaining the robust core logic of the original project.
