# ── Veronica AI — Dockerfile ─────────────────────────────────────────────────
# Многоэтапная сборка: зависимости отдельно от кода → быстрый rebuild
FROM python:3.12-slim AS builder

WORKDIR /build

# Системные зависимости для компиляции (bcrypt, cryptography и т.д.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Runtime образ ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /install /usr/local

# Копируем код приложения
COPY main.py .
COPY static/ static/

# Создаём папку для персистентных данных (БД монтируется как volume)
RUN mkdir -p /app/data

# Переменные окружения по умолчанию (переопределяются через .env или docker-compose)
ENV SQLITE_DB_PATH=/app/data/veronica.db \
    ALLOWED_ORIGINS=http://localhost:80,http://localhost:8000 \
    MAX_UPLOAD_SIZE_MB=10 \
    FILE_TTL_SECONDS=3600 \
    JWT_EXPIRE_MINUTES=1440 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck — используем встроенный /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
