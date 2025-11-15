# Installation Guide

Complete setup guide for MediaOverload system.

---

## Prerequisites

**Required Services:**
- Docker & Docker Compose
- ComfyUI (running on port 8188)
- Ollama or cloud LLM service
- MySQL or PostgreSQL database
- Discord account (for review workflow)

**System Requirements:**
- Python 3.12+
- GPU with 8GB+ VRAM (recommended for video generation)
- 10GB+ free disk space

---

## Quick Install (Docker)

```bash
# Clone repository
git clone https://github.com/your-repo/mediaoverload.git
cd mediaoverload

# Copy example configs
cp media_overload.env.example media_overload.env
cp configs/social_media/discord/Discord.env.example configs/social_media/discord/Discord.env

# Setup character credentials
mkdir -p configs/social_media/credentials/kirby
cp configs/social_media/credentials/ig.env.example configs/social_media/credentials/kirby/ig.env
cp configs/social_media/credentials/twitter.env.example configs/social_media/credentials/kirby/twitter.env

# Edit environment files with your credentials
# Windows: notepad media_overload.env
# Linux/Mac: nano media_overload.env

# Start services
docker-compose up --build -d
```

---

## Manual Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**
- `PyYAML` - Configuration
- `schedule` - Task scheduling
- `python-dotenv` - Environment variables
- `SQLAlchemy` - Database ORM
- `Pillow` - Image processing
- `ollama` - Local LLM
- `google-generativeai` - Gemini API
- `websocket-client` - ComfyUI communication

### 2. Setup ComfyUI

```bash
# Install ComfyUI (if not already installed)
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Start ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

**Required ComfyUI models:**
- Place SDXL/Flux models in `ComfyUI/models/checkpoints/`
- Download required ControlNet/LoRA models if needed

### 3. Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models
ollama pull llama3.2:latest
ollama pull llama3.2-vision:latest
ollama pull llava:13b
ollama pull qwen2.5vl:7b
```

### 4. Setup Database

**MySQL:**
```bash
# Install MySQL
# Ubuntu: sudo apt install mysql-server
# Mac: brew install mysql

# Create database
mysql -u root -p
CREATE DATABASE mediaoverload;
CREATE USER 'mediauser'@'localhost' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON mediaoverload.* TO 'mediauser'@'localhost';
FLUSH PRIVILEGES;
```

**PostgreSQL:**
```bash
# Install PostgreSQL
# Ubuntu: sudo apt install postgresql
# Mac: brew install postgresql

# Create database
createdb mediaoverload
createuser mediauser -P
```

**Run migrations:**
```sql
-- Character table
CREATE TABLE anime_roles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    role_name_en VARCHAR(255) NOT NULL,
    group_name VARCHAR(255),
    status INT DEFAULT 1,
    weight INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- News table
CREATE TABLE news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    keyword VARCHAR(200),
    content TEXT,
    publish_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Environment Configuration

### Main Config (`media_overload.env`)

```env
# Database
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=mediauser
DB_PASSWORD=yourpassword
DB_NAME=mediaoverload

# Discord
DISCORD_REVIEW_BOT_TOKEN=your_discord_bot_token
DISCORD_REVIEW_CHANNEL_ID=your_channel_id

# AI Services
OLLAMA_API_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
COMFYUI_API_URL=http://localhost:8188

# Optional: Google Gemini
GOOGLE_API_KEY=your_google_api_key

# Optional: OpenRouter
OPEN_ROUTER_TOKEN=your_openrouter_api_key
```

### Discord Webhook (`configs/social_media/discord/Discord.env`)

```env
DISCORD_NOTIFICATION_WEBHOOK_URL=your_webhook_url
DISCORD_ERROR_WEBHOOK_URL=your_error_webhook_url
```

### Character Credentials

**Instagram** (`configs/social_media/credentials/{character}/ig.env`)
```env
IG_USERNAME=your_instagram_username
IG_PASSWORD=your_instagram_password
IG_USER_ID=your_user_id
IG_ACCOUNT_COOKIE_FILE_PATH=ig_account.json
```

**Twitter** (`configs/social_media/credentials/{character}/twitter.env`)
```env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

---

## Discord Bot Setup

### 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Go to "Bot" section
4. Click "Add Bot"
5. Copy bot token to `DISCORD_REVIEW_BOT_TOKEN`

### 2. Setup Bot Permissions

Required permissions:
- Read Messages/View Channels
- Send Messages
- Attach Files
- Add Reactions
- Use Slash Commands

### 3. Invite Bot to Server

1. Go to OAuth2 → URL Generator
2. Select scopes: `bot`, `applications.commands`
3. Select permissions above
4. Copy generated URL and open in browser
5. Select your server and authorize

### 4. Get Channel ID

1. Enable Developer Mode in Discord settings
2. Right-click on review channel
3. Click "Copy ID"
4. Paste into `DISCORD_REVIEW_CHANNEL_ID`

---

## Twitter API Setup

### Get API Keys

1. Apply for Twitter Developer Account at [developer.twitter.com](https://developer.twitter.com)
2. Create new app in Developer Portal
3. Go to "Keys and tokens"
4. Generate API Key and Secret
5. Generate Access Token and Secret

### Important: Set Permissions

**Critical:** Set app permissions to "Read and Write"

1. Go to app settings
2. Navigate to "User authentication settings"
3. Set permissions to **"Read and Write"**
4. Save changes
5. **Regenerate** Access Token and Secret (old tokens won't work with new permissions)

**Common Issues:**
- `403 Forbidden` → Check permissions are "Read and Write"
- `403 Forbidden` → Regenerate tokens after changing permissions
- Rate limits → Free tier has strict limits, wait between posts

---

## Verification

Test your setup:

```bash
# Test database connection
python -c "from lib.database import db_pool; print('DB OK')"

# Test Ollama
curl http://localhost:11434/api/tags

# Test ComfyUI
curl http://localhost:8188/system_stats

# Generate test content
python run_media_interface.py --character kirby --prompt "Test image"
```

---

## Troubleshooting

**ComfyUI not responding:**
- Check ComfyUI is running: `ps aux | grep comfyui`
- Check port 8188 is accessible
- View ComfyUI logs for errors

**Ollama connection failed:**
- Verify Ollama service: `systemctl status ollama`
- Test API: `curl http://localhost:11434/api/tags`
- Check firewall settings

**Database connection error:**
- Verify credentials in `.env`
- Check database service is running
- Test connection: `mysql -u mediauser -p mediaoverload`

**Discord bot not responding:**
- Verify bot token is correct
- Check bot has channel permissions
- Enable "Message Content Intent" in bot settings

---

## Next Steps

- [Configuration Guide](configuration.md) - Setup characters and workflows
- [Quick Examples](../examples/README.md) - Try example scripts
- [Architecture Overview](architecture.md) - Understand system design
