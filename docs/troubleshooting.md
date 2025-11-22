# Troubleshooting Guide

This guide provides solutions to common issues encountered when running MediaOverload.

## Table of Contents
- [Quick Diagnostics](#quick-diagnostics)
- [Service Issues](#service-issues)
    - [ComfyUI Connection Failed](#comfyui-connection-failed)
    - [Ollama Connection Error](#ollama-connection-error)
    - [Database Connection Failed](#database-connection-failed)
- [Social Media Issues](#social-media-issues)
    - [Twitter Publishing Failed](#twitter-publishing-failed)
    - [Instagram Publishing Failed](#instagram-publishing-failed)
    - [Discord Bot Not Responding](#discord-bot-not-responding)
- [Generation Issues](#generation-issues)
    - [Image Generation Failed](#image-generation-failed)
    - [Vision Analysis Errors](#vision-analysis-errors)
- [Debugging Techniques](#debugging-techniques)

---

## Quick Diagnostics

Run these commands to check the status of your system:

```bash
# Check AI Services
curl http://localhost:11434/api/tags      # Ollama
curl http://localhost:8188/system_stats   # ComfyUI

# Test Database
python -c "from lib.database import db_pool; print('DB OK')"

# View Recent Logs
tail -f logs/$(date +%Y-%m-%d).log
```

---

## Service Issues

### ComfyUI Connection Failed

**Symptoms:**
- `Connection refused` errors in logs.
- `ComfyUI not responding`.

**Solutions:**
1.  **Check Process**: Ensure ComfyUI is running (`ps aux | grep comfyui` or check Docker container).
2.  **Verify Port**: It must be listening on port `8188`.
3.  **Check Bind Address**: Ensure it was started with `--listen 0.0.0.0` (especially if using Docker).
4.  **Firewall**: Verify that port 8188 is allowed through your firewall.
5.  **Config**: Check `COMFYUI_API_URL` in your `.env` file. It should be `http://localhost:8188` (no trailing slash).

### Ollama Connection Error

**Symptoms:**
- `Connection error` during prompt generation.
- `Ollama API unreachable`.

**Solutions:**
1.  **Service Status**: Check if Ollama is running (`ollama serve`).
2.  **Model Availability**: Run `ollama list` to ensure `llama3.2` (or your configured model) is installed.
3.  **API URL**: Verify `OLLAMA_API_BASE_URL` in `.env`.

### Database Connection Failed

**Symptoms:**
- `Can't connect to MySQL server`.
- `sqlalchemy.exc.OperationalError`.

**Solutions:**
1.  **Service Status**: Ensure MySQL/PostgreSQL is running.
2.  **Credentials**: Double-check `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in `.env`.
3.  **Host**: Use `localhost` or the Docker service name (e.g., `db`) depending on your setup.
4.  **Permissions**: Verify the user has `ALL PRIVILEGES` on the database.

---

## Social Media Issues

### Twitter Publishing Failed

**Symptoms:**
- `403 Forbidden`.
- `You are not permitted to perform this action`.

**Solutions:**
1.  **Permissions**: Go to the Twitter Developer Portal. Under "User authentication settings", ensure permissions are set to **"Read and Write"**.
2.  **Regenerate Tokens**: Changing permissions **does not** update existing tokens. You **must** regenerate your Access Token and Secret after changing permissions.
3.  **API Plan**: Ensure you are not hitting the rate limits of the Free tier.

### Instagram Publishing Failed

**Symptoms:**
- `Challenge required`.
- Login failed.

**Solutions:**
1.  **2FA/Challenge**: Instagram often flags automated logins. Check your email/phone for a verification code. You may need to log in manually on a browser first to "trust" the location.
2.  **Credentials**: Verify username (not email) and password in `ig.env`.
3.  **Session File**: Delete the `ig_account.json` file to force a fresh login attempt.
4.  **Account Age**: New accounts have strict limits. Wait 24 hours before heavy automation.

### Discord Bot Not Responding

**Symptoms:**
- Bot doesn't post images.
- Bot doesn't react to emojis.

**Solutions:**
1.  **Intents**: In the Discord Developer Portal, enable **"Message Content Intent"**.
2.  **Permissions**: Ensure the bot has `View Channel`, `Send Messages`, and `Attach Files` permissions in the specific channel.
3.  **Channel ID**: Verify `DISCORD_REVIEW_CHANNEL_ID` is correct.

---

## Generation Issues

### Image Generation Failed

**Symptoms:**
- `Failed to generate images`.
- Images missing from output folder.

**Solutions:**
1.  **ComfyUI Logs**: Check the console output of ComfyUI. It often prints detailed node errors (e.g., missing model, invalid input).
2.  **Workflow File**: Ensure the JSON workflow file path in your config is correct and accessible.
3.  **Models**: Verify that all checkpoints and LoRAs used in the workflow are present in `ComfyUI/models/`.
4.  **GPU Memory**: Run `nvidia-smi`. If VRAM is full, reduce resolution or batch size (`images_per_description`).

### Vision Analysis Errors

**Symptoms:**
- All images are filtered out.
- High API costs.

**Solutions:**
1.  **Threshold**: Lower `similarity_threshold` (e.g., from 0.9 to 0.7).
2.  **API Quotas**: Check if you've exceeded the free tier limits for Gemini or OpenRouter.
3.  **Skip Analysis**: For testing, use `examples/quick_draw_example.py` which bypasses this step.

---

## Debugging Techniques

### Enable Debug Logging

Add this to your Python script to see detailed logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Log Files

Logs are stored in the `logs/` directory, organized by date.

```bash
# Search for errors
grep ERROR logs/*.log
```

### Test Individual Components

You can test specific parts of the system in isolation using Python:

```python
# Test Config Loading
from lib.config_loader import ConfigLoader
config = ConfigLoader.load_character_config('configs/characters/kirby.yaml')
print(config)

# Test Database
from lib.database import db_pool
conn = db_pool.get_connection('mysql')
print("Database connected")
```
