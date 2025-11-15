# Architecture Overview

System design and execution workflows for MediaOverload.

---

## System Architecture

MediaOverload uses a service-oriented architecture with clear separation of concerns.

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                         │
│  ┌──────────────────┐      ┌────────────────────────┐  │
│  │  Scheduler       │      │  Manual Execution      │  │
│  │  (automated)     │      │  (run_media_interface) │  │
│  └──────────────────┘      └────────────────────────┘  │
└────────────────┬────────────────────┬───────────────────┘
                 │                    │
                 v                    v
┌─────────────────────────────────────────────────────────┐
│              ServiceFactory (DI Container)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────┐
│              OrchestrationService                       │
│  (Coordinates entire content generation pipeline)       │
└──┬──────┬──────┬──────┬──────┬──────────────────────────┘
   │      │      │      │      │
   v      v      v      v      v
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│Pmt │ │Cnt │ │Rev │ │Pub │ │Not │  Services
│Svc │ │Svc │ │Svc │ │Svc │ │Svc │
└────┘ └────┘ └────┘ └────┘ └────┘
```

### Core Services

**OrchestrationService**
- Coordinates the entire pipeline
- Manages service lifecycle
- Handles errors and cleanup

**PromptService**
- Generates text prompts from user input or news
- Handles character group selection
- Integrates with LLMs

**ContentGenerationService**
- Orchestrates media generation
- Delegates to strategy pattern
- Manages quality filtering

**ReviewService**
- Sends content to Discord
- Waits for human approval
- Handles review reactions

**PublishingService**
- Converts media formats
- Publishes to social platforms
- Handles platform-specific requirements

**NotificationService**
- Sends status updates to Discord
- Reports errors and completion

---

## Generation Strategy Pattern

Different media types use different strategies with the same interface.

```
┌──────────────────────────────────────┐
│      StrategyFactory                 │
│  (Creates strategy based on type)    │
└──────────────┬───────────────────────┘
               │
               │ Creates based on generation_type
               v
┌──────────────────────────────────────────────────────┐
│            ContentStrategy (Interface)               │
│  - generate_description()                            │
│  - generate_media()                                  │
│  - analyze_media_text_match()                        │
│  - generate_article_content()                        │
└──────┬───────────────┬────────────────┬──────────────┘
       │               │                │
       v               v                v
┌──────────┐   ┌──────────────┐   ┌────────────┐
│ Text2Img │   │ Text2Video   │   │ Image2Img  │
│ Strategy │   │ Strategy     │   │ Strategy   │
└──────────┘   └──────────────┘   └────────────┘
       │               │                │
       v               v                v
┌──────────────────────────────────────────────────────┐
│              External Systems                        │
│  - Ollama/Gemini/OpenRouter (LLM)                   │
│  - ComfyUI (Image/Video Generation)                 │
└──────────────────────────────────────────────────────┘
```

### Strategy Types

**Text2ImageStrategy**
- Direct text → image via ComfyUI
- Standard generation workflow
- Fastest option

**Image2ImageStrategy**
- Transforms existing images
- Uses input_image_path
- Controllable denoise strength

**Text2Image2ImageStrategy**
- Two-stage generation
- First: text → multiple images
- Filter by similarity
- Second: refine images with i2i

**Text2VideoStrategy**
- Text → video via ComfyUI
- Optional MMAudio sound effects
- Slower but creates video content

---

## Execution Flow

### Complete Pipeline

```
1. Load Character Config
   ↓
2. [If group_name] Select random character from database
   ↓
3. Generate Prompt
   ├─ arbitrary: LLM creates creative prompt
   ├─ news: Fetch news → LLM creates scene
   └─ two_character: Fetch 2nd character → LLM creates interaction
   ↓
4. Select Generation Strategy
   ↓
5. Generate Descriptions
   - LLM expands prompt to detailed descriptions
   - Uses selected image_system_prompt template
   - Applies style guidance
   ↓
6. Generate Media
   ├─ text2img: ComfyUI generates images
   ├─ text2video: ComfyUI generates videos
   ├─ image2image: ComfyUI transforms input image
   └─ text2image2image: Two-stage process
   ↓
7. Analyze Quality
   - Vision model (Gemini/OpenRouter) analyzes each image
   - Compares against original description
   - Filters below similarity_threshold
   ↓
8. Generate Article
   - LLM creates caption/description
   - Adds hashtags
   - Formats for platform
   ↓
9. Discord Review
   - Posts media + caption to Discord
   - Waits for ✅ or ❌ reaction
   - Timeout after configured period
   ↓
10. [If approved] Publish
    - Convert formats if needed
    - Post to Instagram/Twitter
    - Handle platform errors
    ↓
11. Notification
    - Send success/failure to Discord
    - Log results
    ↓
12. Cleanup
    - Delete temporary files
    - Close connections
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User/Scheduler
    participant O as OrchestrationService
    participant P as PromptService
    participant C as ContentGenerationService
    participant S as Strategy
    participant ComfyUI
    participant LLM
    participant R as ReviewService
    participant Pub as PublishingService

    U->>O: Start generation
    O->>P: Generate prompt
    P->>LLM: Create prompt
    LLM-->>P: Return prompt
    P-->>O: Return prompt

    O->>C: Generate content
    C->>S: Load strategy
    S->>LLM: Generate descriptions
    LLM-->>S: Return descriptions
    S->>ComfyUI: Generate media
    ComfyUI-->>S: Return media files
    S->>LLM: Analyze quality
    LLM-->>S: Return scores
    S-->>C: Return results
    C-->>O: Return content

    O->>R: Request review
    R->>Discord: Post for approval
    Discord-->>R: User reaction
    R-->>O: Approval result

    O->>Pub: Publish content
    Pub->>Instagram: Post
    Instagram-->>Pub: Success
    Pub-->>O: Published

    O->>Discord: Send notification
    O-->>U: Complete
```

---

## Data Flow

### Character Configuration

```yaml
YAML Config File
    ↓
ConfigLoader.load_character_config()
    ↓
ConfigLoader.create_character_config()
    ↓ (processes weighted choices)
CharacterConfig dataclass
    ↓
ConfigurableCharacter instance
    ↓
OrchestrationService.run()
```

### Weighted Random Selection

System uses weighted random selection for variety:

```python
# Example: generation_type_weights
weights = {
    'text2img': 0.6,
    'text2image2image': 0.4
}
# Normalized to probabilities: [0.6, 0.4]
# Randomly selects: 60% text2img, 40% text2image2image
```

Applied to:
- `generation_type_weights`
- `prompt_method_weights`
- `image_system_prompt_weights`
- `style_weights`

Each execution may use different combinations.

---

## Database Schema

### Character Group Table

```sql
CREATE TABLE anime_roles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role_name_en VARCHAR(255) NOT NULL,
    group_name VARCHAR(255),
    status INT DEFAULT 1,
    weight INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Usage:**
- `group_name`: Groups characters together
- `weight`: Selection probability (higher = more likely)
- `status`: 1 = active, 0 = disabled

### News Table

```sql
CREATE TABLE news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    keyword VARCHAR(200),
    content TEXT,
    publish_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Usage:**
- Fetched when `prompt_method` is `news`
- LLM combines news with character to create scenes

---

## External Integrations

### ComfyUI

**Connection:** WebSocket + HTTP API
**Port:** 8188
**Purpose:** Image and video generation

**Workflow:**
1. Load workflow JSON
2. Update node parameters
3. Queue prompt via API
4. Monitor progress via WebSocket
5. Retrieve generated images

### LLM Providers

**Ollama (Local)**
- Models: llama3.2, qwen, etc.
- Free, unlimited use
- Requires GPU

**Google Gemini**
- Vision models for image analysis
- Paid API (generous free tier)
- Fast response times

**OpenRouter**
- Multiple free models
- Fallback when Ollama unavailable
- Rate limited

### Discord

**Review Bot:**
- Posts media for human review
- Listens for ✅/❌ reactions
- Requires bot token + channel ID

**Webhooks:**
- Status notifications
- Error reporting
- No bot required

### Social Media

**Instagram:**
- Uses unofficial API (instagrapi)
- Requires login credentials
- Cookie-based session

**Twitter:**
- Official API v2
- Requires developer account
- "Read and Write" permissions needed

---

## Error Handling

### Retry Mechanisms

**Database queries:** 3 retries with reconnection
**API calls:** Configurable retry with exponential backoff
**Vision API:** 10 retries (can be rate limited)

### Fallback Strategies

**LLM failures:**
- Ollama → OpenRouter → Gemini
- Different models tried sequentially

**ComfyUI errors:**
- Logged but generation stops
- User notified via Discord

**Publishing errors:**
- Logged with full details
- Content saved locally
- Can be manually retried

---

## Performance Considerations

### Bottlenecks

**Slowest operations:**
1. ComfyUI generation (30s - 5min)
2. Vision model analysis (2-10s per image)
3. Discord review (manual wait)

**Optimization:**
- Use `examples/quick_draw` to skip analysis
- Lower `images_per_description` for faster testing
- Use faster ComfyUI workflows (Flux Krea)

### Resource Usage

**GPU Memory:**
- ComfyUI: 4-8GB for images
- ComfyUI: 8-12GB for videos
- Ollama: 2-8GB depending on model

**Disk Space:**
- Generated media accumulates
- Log files grow daily
- Regular cleanup recommended

---

## Extending the System

### Adding New Platform

1. Implement `SocialMediaPlatform` interface
2. Add to platform_mapping in `ConfigurableCharacterWithSocialMedia`
3. Create credential template
4. Update character config

### Adding New Strategy

1. Inherit from `ContentStrategy`
2. Implement required methods
3. Register in `StrategyFactory`
4. Update character config schema

### Adding New Prompt Template

1. Add to `configs/prompt/image_system_guide.py`
2. Reference in character config `image_system_prompt_weights`

---

## Next Steps

- [Installation Guide](installation.md) - Setup environment
- [Configuration Guide](configuration.md) - Character configs
- [Troubleshooting](troubleshooting.md) - Debug issues
