# Configuration Guide

Complete guide to configuring characters, workflows, and generation strategies.

---

## Character Configuration

Characters are the core of MediaOverload. Each character has a YAML config file in `configs/characters/`.

### Basic Structure

```yaml
character:
  name: kirby
  group_name: Kirby

generation:
  output_dir: /app/output_media
  generation_type_weights: {...}
  workflows: {...}
  similarity_threshold: 0.7
  prompt_method_weights: {...}
  image_system_prompt_weights: {...}
  style_weights: {...}

social_media:
  default_hashtags: [...]
  platforms: {...}

additional_params:
  image: {...}
  video: {...}
```

### Complete Example

See [kirby.yaml](../configs/characters/kirby.yaml) for a fully documented example with inline comments.

---

## Generation Type Weights

Controls which generation method to use. System randomly selects based on weights.

```yaml
generation_type_weights:
  text2img: 0.6                # 60% - Direct text to image
  image2image: 0.1             # 10% - Transform existing images
  text2image2image: 0.2        # 20% - Two-stage generation
  text2video: 0.1              # 10% - Text to video
```

**Set to 0 to disable a type.**

Weights are normalized automatically (don't need to sum to 1.0).

---

## Workflow Configuration

Maps each generation type to its ComfyUI workflow JSON file.

```yaml
workflows:
  text2img: /app/configs/workflow/nova-anime-xl.json
  text2video: /app/configs/workflow/wan2.1_t2v_audio.json
  image2image: /app/configs/workflow/image_to_image.json
  text2image2image: /app/configs/workflow/nova-anime-xl.json  # First stage
```

**Available workflows:**
- `nova-anime-xl.json` - Anime style (SDXL)
- `flux_krea_dev.json` - Fast generation (Flux)
- `flux_dev.json` - High quality (Flux)
- `image_to_image.json` - Image transformation
- `wan2.1_t2v_audio.json` - Video with audio

---

## Prompt Generation Methods

Controls how text prompts are created.

```yaml
prompt_method_weights:
  arbitrary: 0.1              # Random creative scenarios
  news: 0.9                   # Based on news articles
```

**Methods:**
- `arbitrary` - LLM generates creative prompts
- `news` - Fetches news from database, LLM creates related scenes

---

## Image System Prompts

Templates that guide the LLM to generate specific styles of descriptions.

```yaml
image_system_prompt_weights:
  conceptual_logo_design_prompt: 0.1              # Logo/icon design
  stable_diffusion_prompt: 0.2                    # Standard SDXL prompts
  two_character_interaction_generate_system_prompt: 0.25  # Character interactions
  warm_scene_description_system_prompt: 0.2       # Heartwarming scenes
  sticker_prompt_system_prompt: 0.25              # Sticker/emoji style
```

**Available prompts:**
- `stable_diffusion_prompt` - Optimized for SDXL
- `conceptual_logo_design_prompt` - Minimalist logos/icons
- `two_character_interaction_generate_system_prompt` - Two characters interacting
- `warm_scene_description_system_prompt` - Emotional/wholesome scenes
- `sticker_prompt_system_prompt` - Chibi sticker expressions
- `black_humor_system_prompt` - Dark comedy scenarios
- `cinematic_stable_diffusion_prompt` - Film-quality cinematography

See [image_system_guide.py](../configs/prompt/image_system_guide.py) for all available prompts.

---

## Style Weights

Visual style guidance for image generation.

```yaml
style_weights:
  "minimalism style with pure background": 0.2
  "watercolor painting style": 0.1
  "sketch-like illustration": 0.1
  "": 0.6  # Let LLM decide (60%)
```

Empty string `""` means no style constraint - LLM chooses freely.

---

## Social Media Configuration

### Default Hashtags

```yaml
social_media:
  default_hashtags:
    - "#kirby"
    - "#nintendo"
    - "#ai"
```

Appended to all posts for this character.

### Platform Setup

```yaml
platforms:
  instagram:
    config_folder_path: /app/configs/social_media/credentials/kirby
    enabled: true

  twitter:
    config_folder_path: /app/configs/social_media/credentials/kirby
    prefix: ""  # Optional prefix for tweets
    enabled: true
```

Each platform needs credentials in the specified folder:
- Instagram: `ig.env`
- Twitter: `twitter.env`

---

## Generation Parameters

### Image Generation

```yaml
additional_params:
  image:
    images_per_description: 8  # Generate 8 images per prompt

    custom_node_updates:       # Override ComfyUI workflow parameters
      - node_type: "PrimitiveInt"
        inputs:
          value: 1024           # Image resolution
```

### Image-to-Image Strategy

Used when `generation_type` is `image2image`:

```yaml
additional_params:
  image:
    images_per_input: 2        # Generate 2 variations per input image
    denoise: 0.6               # Denoising strength (0.5-0.7)
```

**Denoise values:**
- `0.5` - More similar to original
- `0.6` - Balanced (recommended)
- `0.7` - More creative freedom

### Text→Image→Image Two-Stage

Used when `generation_type` is `text2image2image`:

```yaml
additional_params:
  image:
    first_stage:
      images_per_description: 4  # Stage 1: Generate 4 base images

    second_stage:
      images_per_input: 4          # Stage 2: Refine each to 4 variations
      denoise: 0.7                 # Higher denoise for refinement

    i2i_workflow_path: /app/configs/workflow/image_to_image.json
```

**Process:**
1. Generate 4 images from text (first_stage)
2. AI selects best images based on similarity_threshold
3. Refine selected images with image-to-image (second_stage)

### Video Generation

```yaml
additional_params:
  video:
    videos_per_description: 2  # Generate 2 videos per prompt

    custom_node_updates:
      - node_type: "PrimitiveInt"
        inputs:
          value: 512            # Video resolution
      - node_type: "EmptyHunyuanLatentVideo"
        inputs:
          length: 97            # Frame count
```

---

## Quality Control

### Similarity Threshold

```yaml
similarity_threshold: 0.7  # 0.0-1.0
```

Vision model scores image-text matching. Images below threshold are filtered out.

**Guidelines:**
- `0.5-0.6` - Loose (more results)
- `0.7-0.8` - Balanced (recommended)
- `0.9+` - Strict (fewer but higher quality)

---

## Creating New Characters

### 1. Create Character Config

```bash
# Copy example
cp configs/characters/kirby.yaml configs/characters/newchar.yaml

# Edit config
nano configs/characters/newchar.yaml
```

### 2. Setup Credentials

```bash
# Create credentials directory
mkdir -p configs/social_media/credentials/newchar

# Copy credential templates
cp configs/social_media/credentials/ig.env.example \
   configs/social_media/credentials/newchar/ig.env

cp configs/social_media/credentials/twitter.env.example \
   configs/social_media/credentials/newchar/twitter.env

# Edit with actual credentials
nano configs/social_media/credentials/newchar/ig.env
nano configs/social_media/credentials/newchar/twitter.env
```

### 3. Test Character

```bash
python run_media_interface.py \
  --config configs/characters/newchar.yaml \
  --prompt "Test prompt"
```

---

## Character Groups

Characters can belong to groups for dynamic selection.

### Database Setup

```sql
INSERT INTO anime_roles (role_name_en, group_name, status, weight)
VALUES
  ('Kirby', 'Kirby', 1, 10),
  ('Meta Knight', 'Kirby', 1, 5),
  ('Waddle Dee', 'Kirby', 1, 8);
```

### Use in Config

```yaml
character:
  name: kirby
  group_name: Kirby  # System can randomly select from this group
```

When `group_name` is set, system can randomly pick any character from that group in the database.

---

## ComfyUI Workflow Customization

### 1. Design Workflow

- Open ComfyUI web interface
- Design your node graph
- Test with sample prompts

### 2. Export Workflow

- Click "Save (API Format)"
- Save as JSON file
- Place in `configs/workflow/`

### 3. Reference in Config

```yaml
workflows:
  text2img: /app/configs/workflow/my_custom_workflow.json
```

### 4. Parameter Overrides

Override specific node parameters:

```yaml
custom_node_updates:
  - node_type: "KSampler"
    inputs:
      steps: 30
      cfg: 7.5
  - node_type: "EmptyLatentImage"
    inputs:
      width: 1024
      height: 1024
```

---

## Advanced Configuration

### Multiple AI Model Providers

System automatically rotates between configured providers:

```env
# .env file
OLLAMA_API_BASE_URL=http://localhost:11434
GOOGLE_API_KEY=your_gemini_key
OPEN_ROUTER_TOKEN=your_openrouter_key
```

### News-Based Prompts

Requires news table in database:

```sql
CREATE TABLE news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    keyword VARCHAR(200),
    content TEXT,
    publish_date DATE
);
```

System fetches recent news and generates character-related scenes.

---

## Configuration Best Practices

**Start Simple**
- Use `text2img` only at first
- Add complexity after testing works

**Test Weights**
- Set all weights to 0 except one for testing
- Gradually enable more options

**Monitor Quality**
- Start with high similarity_threshold (0.8+)
- Lower if too many images filtered out

**Manage Costs**
- Use Ollama for local/free generation
- OpenRouter for occasional cloud use
- Gemini for vision analysis only

---

## Example Configurations

See `configs/characters/` for complete examples:

- `kirby.yaml` - Balanced configuration with multiple styles
- Check for other character examples in the directory

---

## Next Steps

- [Installation Guide](installation.md) - Setup environment
- [Architecture Overview](architecture.md) - System design
- [Troubleshooting](troubleshooting.md) - Common issues
