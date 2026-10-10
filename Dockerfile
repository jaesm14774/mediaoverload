FROM python:3.12-slim

WORKDIR /app

# 安裝系統依賴 (包含 ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ffmpeg \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 設定時區為 Asia/Taipei
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 複製整個專案（除了 .dockerignore 中排除的文件）
COPY . .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 運行排程器
CMD ["python", "scheduler/scheduler.py"]
