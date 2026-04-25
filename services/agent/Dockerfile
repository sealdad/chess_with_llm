# Agent Service Dockerfile
# Lightweight Python image with chess engine

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    stockfish \
    curl \
    # ffmpeg for audio processing (pydub)
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/agent/main.py .
COPY services/agent/voice_events.py .
COPY services/agent/voice_processor.py .
COPY services/agent/audio/ ./audio/
COPY services/agent/websocket/ ./websocket/
COPY services/agent/conversation/ ./conversation/
COPY services/agent/static/ ./static/

# Environment variables
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV VISION_SERVICE_URL=http://vision:8001
ENV ROBOT_SERVICE_URL=http://robot:8002

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
