# Troubleshooting Guide

Solutions to common issues when running MediaOverload.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Service Issues](#service-issues)
- [Social Media Issues](#social-media-issues)
- [Generation Issues](#generation-issues)
- [Debugging Techniques](#debugging-techniques)

---

## Quick Diagnostics

Run the following commands to check system status:

```bash
# Check AI services
curl http://localhost:11434/api/tags      # Ollama
curl http://localhost:8188/system_stats   # ComfyUI

# Test database
python -c "from lib.database import db_pool; print('DB OK')"

# View recent logs
tail -f logs/$(date +%Y-%m-%d).log
```

---

## Service Issues

### ComfyUI Connection Failed

**Symptoms:**
- `Connection refused` errors in logs
- `ComfyUI not responding`

**Solutions:**

1. **Check process** — Ensure ComfyUI is running (`ps aux | grep comfyui` or check the Docker container).
2. **Verify port** — ComfyUI must be listening on port `8188`.
3. **Check bind address** — Start ComfyUI with `--listen 0.0.0.0` (required when using Docker).
4. **Firewall** — Verify that port 8188 is allowed.
5. **Config** — Check `COMFYUI_API_URL` in the `.env` file. Use `http://localhost:8188` (no trailing slash).

### Ollama Connection Error

**Symptoms:**
- `Connection error` during prompt generation
- `Ollama API unreachable`

**Solutions:**

1. **Service status** — Check if Ollama is running (`ollama serve`).
2. **Model availability** — Run `ollama list` to confirm `llama3.2` (or the configured model) is installed.
3. **API URL** — Verify `OLLAMA_API_BASE_URL` in `.env`.

### Database Connection Failed

**Symptoms:**
- `Can't connect to MySQL server`
- `sqlalchemy.exc.OperationalError`

**Solutions:**

1. **Service status** — Ensure MySQL/PostgreSQL is running.
2. **Credentials** — Double-check `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in `.env`.
3. **Host** — Use `localhost` or the Docker service name (e.g., `db`) depending on the setup.
4. **Permissions** — Verify the database user has `ALL PRIVILEGES` on the target database.

---

## Social Media Issues

### Twitter Publishing Failed

**Symptoms:**
- `403 Forbidden`
- `You are not permitted to perform this action`

**Solutions:**

1. **Permissions** — In the Twitter Developer Portal under "User authentication settings", ensure permissions are set to **"Read and Write"**.
2. **Regenerate tokens** — Changing permissions does **not** update existing tokens. Regenerate the Access Token and Secret after changing permissions.
3. **API plan** — Check if the Free tier rate limits have been exceeded.

### Instagram Publishing Failed

**Symptoms:**
- `Challenge required`
- Login failed

**Solutions:**

1. **2FA/Challenge** — Instagram often flags automated logins. Check email/phone for a verification code. Log in manually on a browser first to "trust" the location.
2. **Credentials** — Verify username (not email) and password in `ig.env`.
3. **Session file** — Delete `ig_account.json` to force a fresh login attempt.
4. **Account age** — New accounts have strict limits. Wait 24 hours before attempting automation.

### Discord Bot Not Responding

**Symptoms:**
- Bot does not post images
- Bot does not react to emojis

**Solutions:**

1. **Intents** — In the Discord Developer Portal, enable **"Message Content Intent"**.
2. **Permissions** — Ensure the bot has `View Channel`, `Send Messages`, and `Attach Files` permissions in the target channel.
3. **Channel ID** — Verify `DISCORD_REVIEW_CHANNEL_ID` is correct.

---

## Generation Issues

### Image Generation Failed

**Symptoms:**
- `Failed to generate images`
- Images missing from output folder

**Solutions:**

1. **ComfyUI logs** — Check the console output of ComfyUI. Node errors (e.g., missing model, invalid input) are printed there.
2. **Workflow file** — Ensure the JSON workflow file path in the config is correct and accessible.
3. **Models** — Verify that all checkpoints and LoRAs referenced in the workflow are present in `ComfyUI/models/`.
4. **GPU memory** — Run `nvidia-smi`. If VRAM is full, reduce resolution or batch size (`images_per_description`).

### Vision Analysis Errors

**Symptoms:**
- All images are filtered out
- High API costs

**Solutions:**

1. **Threshold** — Lower `similarity_threshold` (e.g., from 0.9 to 0.7).
2. **API quotas** — Check if the free tier limits for Gemini or OpenRouter have been exceeded.
3. **Skip analysis** — Use `examples/quick_draw_example.py` for testing. The quick draw module bypasses analysis.

---

## Debugging Techniques

### Enable Debug Logging

Add the following to a Python script to see detailed logs:

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

Test specific parts of the system in isolation:

```python
# Test config loading
from lib.config_loader import ConfigLoader
config = ConfigLoader.load_character_config('configs/characters/kirby.yaml')
print(config)

# Test database connection
from lib.database import db_pool
conn = db_pool.get_connection('mysql')
print("Database connected")
```
