FROM python:3.12-slim

# 1. Системные либы для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Устанавливаем uv через pip (план Б из-за блокировок)
RUN pip install uv

WORKDIR /app

# 3. Копируем конфиги и ставим зависимости
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# 4. Копируем код (теперь быстро благодаря .dockerignore)
COPY . .

EXPOSE 8501

# 5. Запуск
ENTRYPOINT ["uv", "run", "streamlit", "run", "inference/app.py", "--server.port=8501", "--server.address=0.0.0.0"]