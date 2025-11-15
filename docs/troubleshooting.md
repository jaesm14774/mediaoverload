# Troubleshooting Guide

Solutions to common issues and debugging tips.

---

## Quick Diagnostics

Run these commands to check system status:

```bash
# Check services
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8188/system_stats  # ComfyUI

# Test database
python -c "from lib.database import db_pool; print('DB OK')"

# View logs
tail -f logs/$(date +%Y-%m-%d).log
```

---

## Common Issues

### ComfyUI Connection Failed

**Symptoms:**
- `Connection refused` errors
- `ComfyUI not responding`

**Solutions:**

1. **Check if ComfyUI is running:**
   ```bash
   ps aux | grep comfyui
   # Or on Windows: tasklist | findstr python
   ```

2. **Start ComfyUI:**
   ```bash
   cd /path/to/ComfyUI
   python main.py --listen 0.0.0.0 --port 8188
   ```

3. **Verify port is accessible:**
   ```bash
   curl http://localhost:8188/system_stats
   # Should return JSON with system info
   ```

4. **Check firewall:**
   - Allow port 8188 through firewall
   - Windows: Add exception in Windows Defender
   - Linux: `sudo ufw allow 8188`

5. **Check URL in config:**
   ```env
   COMFYUI_API_URL=http://localhost:8188  # Correct
   # NOT: http://localhost:8188/
   ```

---

### Ollama Connection Error

**Symptoms:**
- `Connection error` when generating prompts
- `Ollama API unreachable`

**Solutions:**

1. **Check Ollama service:**
   ```bash
   # Linux
   systemctl status ollama

   # Mac/manual
   ps aux | grep ollama
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Test API:**
   ```bash
   curl http://localhost:11434/api/tags
   # Should list installed models
   ```

4. **Pull required models:**
   ```bash
   ollama pull llama3.2:latest
   ollama pull llama3.2-vision:latest
   ```

5. **Check API URL:**
   ```env
   OLLAMA_API_BASE_URL=http://localhost:11434  # No trailing slash
   ```

---

### Database Connection Failed

**Symptoms:**
- `Can't connect to MySQL server`
- `sqlalchemy.exc.OperationalError`

**Solutions:**

1. **Check database service:**
   ```bash
   # MySQL
   systemctl status mysql
   # Or: ps aux | grep mysql

   # PostgreSQL
   systemctl status postgresql
   ```

2. **Test connection manually:**
   ```bash
   mysql -h localhost -u mediauser -p mediaoverload
   # Enter password
   ```

3. **Verify credentials:**
   ```env
   DB_HOST=localhost  # Not 127.0.0.1 if using socket
   DB_PORT=3306
   DB_USER=mediauser
   DB_PASSWORD=yourpassword
   DB_NAME=mediaoverload
   ```

4. **Check user permissions:**
   ```sql
   SHOW GRANTS FOR 'mediauser'@'localhost';
   # Should have ALL PRIVILEGES on mediaoverload.*
   ```

5. **Recreate user if needed:**
   ```sql
   DROP USER 'mediauser'@'localhost';
   CREATE USER 'mediauser'@'localhost' IDENTIFIED BY 'newpassword';
   GRANT ALL PRIVILEGES ON mediaoverload.* TO 'mediauser'@'localhost';
   FLUSH PRIVILEGES;
   ```

---

### Discord Bot Not Responding

**Symptoms:**
- Bot doesn't post to channel
- No reactions detected
- `403 Forbidden` errors

**Solutions:**

1. **Verify bot token:**
   - Copy token from Discord Developer Portal
   - Paste exactly (no extra spaces)
   - Token starts with `MTk...` or similar

2. **Check bot permissions:**
   - Bot needs these in the channel:
     - View Channel
     - Send Messages
     - Attach Files
     - Add Reactions
   - Right-click channel → Edit Channel → Permissions

3. **Enable privileged intents:**
   - Go to Discord Developer Portal
   - Bot → Privileged Gateway Intents
   - Enable "Message Content Intent"
   - Enable "Server Members Intent"

4. **Verify channel ID:**
   ```python
   # Test in Python
   print(DISCORD_REVIEW_CHANNEL_ID)
   # Should be a long number like: 1234567890123456789
   ```

5. **Check bot is in server:**
   - Use OAuth2 URL generator to re-invite
   - Select bot + applications.commands
   - Add all required permissions

---

### Image Generation Failed

**Symptoms:**
- `Failed to generate images`
- Images not appearing in output folder
- ComfyUI errors in logs

**Solutions:**

1. **Check ComfyUI logs:**
   ```bash
   # In ComfyUI directory
   tail -f comfyui.log
   ```

2. **Verify workflow file exists:**
   ```bash
   ls -la configs/workflow/nova-anime-xl.json
   ```

3. **Check model files:**
   - Models must be in `ComfyUI/models/checkpoints/`
   - VAE in `ComfyUI/models/vae/`
   - Check workflow for required model names

4. **Test workflow in ComfyUI UI:**
   - Load workflow JSON in ComfyUI web interface
   - Run manually with test prompt
   - Fix any node errors

5. **Check GPU memory:**
   ```bash
   nvidia-smi
   # Should show free VRAM
   ```
   - If OOM, lower resolution in config
   - Close other GPU applications

6. **Verify custom_node_updates syntax:**
   ```yaml
   custom_node_updates:
     - node_type: "PrimitiveInt"
       inputs:
         value: 1024  # Must match workflow node names
   ```

---

### Twitter Publishing Failed

**Symptoms:**
- `403 Forbidden` errors
- `You are not permitted to perform this action`
- `Invalid or expired token`

**Solutions:**

1. **Set correct permissions:**
   - Go to Twitter Developer Portal
   - App Settings → User authentication settings
   - Set to **"Read and Write"** (not "Read only")
   - Save changes

2. **Regenerate tokens:**
   - After changing permissions, old tokens won't work
   - Go to Keys and tokens
   - Click "Regenerate" for Access Token and Secret
   - Update in `twitter.env`

3. **Check API plan:**
   - Free tier: Can use API v2 for tweets
   - Paid tier: Can use both v1.1 and v2
   - System automatically tries v2 first

4. **Handle rate limits:**
   ```
   429 Too Many Requests
   ```
   - System auto-retries after rate limit period
   - Reduce posting frequency
   - Wait time shown in logs

5. **Check tweet length:**
   - Tweets limited to 280 characters
   - System auto-truncates with "..."
   - Check article generation length

6. **Verify credentials:**
   ```env
   TWITTER_API_KEY=your_key  # No quotes
   TWITTER_API_SECRET=your_secret
   TWITTER_ACCESS_TOKEN=your_token
   TWITTER_ACCESS_TOKEN_SECRET=your_token_secret
   ```

---

### Instagram Publishing Failed

**Symptoms:**
- Login failed
- `Challenge required`
- Session expired

**Solutions:**

1. **Check credentials:**
   ```env
   IG_USERNAME=your_username  # Not email
   IG_PASSWORD=your_password
   ```

2. **Handle login challenges:**
   - Instagram may require 2FA
   - Check email/phone for verification code
   - May need to login manually first

3. **Session cookies:**
   - First login creates `ig_account.json`
   - Reuses session for subsequent posts
   - Delete file to force fresh login

4. **Account restrictions:**
   - New accounts have strict limits
   - Wait 24h between first posts
   - Don't post too frequently

5. **Image format:**
   - System converts to JPEG automatically
   - Check file size < 8MB
   - Resolution between 320x320 and 1080x1080

---

### Vision Analysis Errors

**Symptoms:**
- `Vision model failed`
- All images filtered out
- High API costs

**Solutions:**

1. **Check similarity threshold:**
   ```yaml
   similarity_threshold: 0.9  # Too high?
   # Try: 0.6-0.7 for more results
   ```

2. **Verify API keys:**
   ```env
   GOOGLE_API_KEY=your_key  # For Gemini
   OPEN_ROUTER_TOKEN=your_token  # For OpenRouter
   ```

3. **Check API quotas:**
   - Gemini: Free tier has daily limits
   - OpenRouter: Free models have rate limits
   - View limits in respective dashboards

4. **Use examples to skip analysis:**
   ```bash
   python examples/quick_draw_example.py
   # Skips vision analysis for faster testing
   ```

5. **Reduce images to analyze:**
   ```yaml
   images_per_description: 3  # Instead of 10
   ```

---

## Debugging Techniques

### Enable Debug Logging

```python
# In your code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Log Files

```bash
# Today's log
cat logs/$(date +%Y-%m-%d).log

# Search for errors
grep ERROR logs/*.log

# Follow live
tail -f logs/$(date +%Y-%m-%d).log
```

### Test Individual Components

```python
# Test config loading
from lib.config_loader import ConfigLoader
config = ConfigLoader.load_character_config('configs/characters/kirby.yaml')
print(config)

# Test database
from lib.database import db_pool
conn = db_pool.get_connection('mysql')
print("Database connected")

# Test LLM
import ollama
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Hello'}
])
print(response)
```

### Monitor Resources

```bash
# GPU usage
nvidia-smi -l 1

# CPU/Memory
htop

# Disk space
df -h

# Network
netstat -tuln | grep -E '8188|11434|3306'
```

---

## Environment Issues

### Python Version Mismatch

**Required:** Python 3.12+

```bash
python --version
# Should show 3.12.x or higher
```

**Solution:**
```bash
# Install correct version
# Ubuntu: sudo apt install python3.12
# Mac: brew install python@3.12

# Use with venv
python3.12 -m venv venv
source venv/bin/activate
```

### Missing Dependencies

```bash
# Reinstall all
pip install -r requirements.txt

# Specific packages
pip install SQLAlchemy pymysql pillow
```

### Path Issues

**Symptoms:**
- `File not found` errors
- Config files not loading

**Solutions:**
```python
# Use absolute paths in configs
workflow_path: /app/configs/workflow/file.json  # ✓
workflow_path: configs/workflow/file.json       # ✗

# Or relative to project root
workflow_path: ./configs/workflow/file.json
```

---

## Performance Issues

### Slow Generation

**Causes:**
- Low GPU memory → Smaller batch sizes
- CPU bottleneck → Use GPU acceleration
- Network latency → Use local models

**Solutions:**
```yaml
# Reduce image count
images_per_description: 3  # Instead of 10

# Use faster workflow
workflows:
  text2img: /app/configs/workflow/flux_krea_dev.json  # Faster
```

### High Memory Usage

```bash
# Check memory
free -h

# Clear cache
sync; echo 3 > /proc/sys/vm/drop_caches
```

**Solutions:**
- Lower image resolution
- Use smaller LLM models
- Reduce `images_per_description`

### Disk Space Full

```bash
# Check usage
du -sh output_media/ logs/

# Clean old files
find output_media/ -mtime +7 -delete  # Delete > 7 days
find logs/ -mtime +30 -delete  # Delete old logs
```

---

## Getting Help

### Collect Information

Before asking for help, gather:

1. **Error message:**
   ```bash
   grep ERROR logs/*.log | tail -20
   ```

2. **System info:**
   ```bash
   python --version
   nvidia-smi
   docker --version
   ```

3. **Configuration:**
   ```bash
   cat media_overload.env | grep -v PASSWORD
   ```

4. **Recent logs:**
   ```bash
   tail -100 logs/$(date +%Y-%m-%d).log
   ```

### Report Issues

Include in GitHub issue:
- Error message (full traceback)
- Steps to reproduce
- Expected vs actual behavior
- System info
- Relevant config (remove secrets!)

---

## Next Steps

- [Installation Guide](installation.md) - Setup instructions
- [Configuration Guide](configuration.md) - Config options
- [Architecture Overview](architecture.md) - System design
