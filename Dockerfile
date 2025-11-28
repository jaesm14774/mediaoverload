FROM python:3.12-slim

WORKDIR /app

# 安裝系統依賴 (包含 ffmpeg)
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    default-libmysqlclient-dev \
    libpq-dev \
    unixodbc-dev \
    curl \
    gnupg2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安裝 MSSQL ODBC 驅動程式
RUN set -eux; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-archive-keyring.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && ACCEPT_EULA=Y apt-get install -y mssql-tools18 \
    && echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc \
    && apt-get install -y unixodbc-dev \
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