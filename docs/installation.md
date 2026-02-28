# Installation Guide

This guide covers the complete setup process for MediaOverload, from prerequisites to verification.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Install (Docker)](#quick-install-docker)
- [Manual Installation](#manual-installation)
- [Environment Configuration](#environment-configuration)
- [Discord Bot Setup](#discord-bot-setup)
- [Twitter API Setup](#twitter-api-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: 3.12 or higher
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for video generation)
- **Disk Space**: 10GB+ free

### Required Services

- **Docker & Docker Compose** (for containerized setup)
- **ComfyUI**: Running locally on port 8188
- **Ollama** (or a cloud LLM key): For text generation
- **Database**: MySQL or PostgreSQL
- **Discord Account**: For the review workflow

---

## Quick Install (Docker)

**Docker is the fastest way to get started.**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/mediaoverload.git
cd mediaoverload

# 2. Copy example configuration files
cp media_overload.env.example media_overload.env
cp configs/social_media/discord/Discord.env.example configs/social_media/discord/Discord.env

# 3. Set up character credentials (example for 'kirby')
mkdir -p configs/social_media/credentials/kirby
cp configs/social_media/credentials/ig.env.example configs/social_media/credentials/kirby/ig.env
cp configs/social_media/credentials/twitter.env.example configs/social_media/credentials/kirby/twitter.env

# 4. Edit environment files with actual credentials
# Windows: notepad media_overload.env
# Linux/Mac: nano media_overload.env

# 5. Start services
docker-compose up --build -d
```

---

## Manual Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**

| Package | Purpose |
|---------|---------|
| `PyYAML` | Configuration management |
| `schedule` | Task scheduling |
| `python-dotenv` | Environment variable loading |
| `SQLAlchemy` | Database ORM |
| `Pillow` | Image processing |
| `ollama` | Local LLM interface |

### 2. Set Up ComfyUI

1. **Install ComfyUI**:
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    pip install -r requirements.txt
    ```
2. **Start ComfyUI**:
    ```bash
    python main.py --listen 0.0.0.0 --port 8188
    ```
3. **Install models**: Place SDXL/Flux checkpoints in `ComfyUI/models/checkpoints/`.

### 3. Set Up Ollama

1. **Install**: [Download Ollama](https://ollama.com/download)
2. **Start the service**: `ollama serve`
3. **Pull models**:
    ```bash
    ollama pull llama3.2:latest
    ollama pull llama3.2-vision:latest
    ```

### 4. Set Up Database

**MySQL example:**

```bash
mysql -u root -p
```

```sql
CREATE DATABASE mediaoverload;
CREATE USER 'mediauser'@'localhost' IDENTIFIED BY 'yourpassword';
GRANT ALL PRIVILEGES ON mediaoverload.* TO 'mediauser'@'localhost';
FLUSH PRIVILEGES;
```

**Create tables:**

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

# Optional: Cloud APIs
GOOGLE_API_KEY=your_google_api_key
OPEN_ROUTER_TOKEN=your_openrouter_api_key
```

### Character Credentials

**Instagram** (`configs/social_media/credentials/{character}/ig.env`):

```env
IG_USERNAME=your_instagram_username
IG_PASSWORD=your_instagram_password
IG_USER_ID=your_user_id
IG_ACCOUNT_COOKIE_FILE_PATH=ig_account.json
```

**Twitter** (`configs/social_media/credentials/{character}/twitter.env`):

```env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

---

## Discord Bot Setup

1. **Create a bot**: Go to the [Discord Developer Portal](https://discord.com/developers/applications), create a new application, and add a bot.
2. **Copy the token**: Paste it into `DISCORD_REVIEW_BOT_TOKEN`.
3. **Enable intents**: Turn on "Message Content Intent" under "Privileged Gateway Intents".
4. **Invite the bot**: Use the OAuth2 URL Generator with scopes `bot` and `applications.commands`. Grant permissions: `Send Messages`, `Attach Files`, `Add Reactions`, `View Channels`.
5. **Get the channel ID**: Enable Developer Mode in Discord, right-click the review channel, and select "Copy ID". Paste the ID into `DISCORD_REVIEW_CHANNEL_ID`.

---

## Twitter API Setup

**Permissions are the most common source of errors.** Follow the steps carefully.

1. Apply for a [Twitter Developer Account](https://developer.twitter.com).
2. Create an app in the developer portal.
3. Navigate to "User authentication settings".
4. **Set app permissions to "Read and Write".** The default is often "Read only".
5. Save changes.
6. **Regenerate the Access Token and Secret.** Tokens created before the permission change will not work.

---

## Verification

Run the following commands to verify the setup:

```bash
# 1. Test database connection
python -c "from lib.database import db_pool; print('DB OK')"

# 2. Test Ollama
curl http://localhost:11434/api/tags

# 3. Test ComfyUI
curl http://localhost:8188/system_stats

# 4. Generate test content
python run_media_interface.py --character kirby --prompt "Test image"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ComfyUI not responding | Check if ComfyUI is running on port 8188 and that the firewall allows connections. |
| Twitter 403 Forbidden | Set "Read and Write" permissions in the developer portal, then regenerate tokens. |
| Database connection failed | Verify credentials in `.env` and ensure the database service is running. |

For detailed solutions, see the [Troubleshooting Guide](troubleshooting.md).
