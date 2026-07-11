# Equitex Intelligence — production image
# Build:  docker build -t equitex .
# Run:    docker compose up -d   (recommended — mounts /data)

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for layer caching
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# App code (flat layout, app.py at repo root)
COPY . .

# Persistent storage for journal/watchlist — mount a volume here
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", \
    "--server.headless=true"]
