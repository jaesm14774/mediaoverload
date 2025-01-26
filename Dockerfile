FROM python:3.12-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製整個專案（除了 .dockerignore 中排除的文件）
COPY . .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 設置環境變數
ENV PYTHONPATH=/app

# 運行排程器
CMD ["python", "scheduler/scheduler.py"] 