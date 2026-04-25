# Vision Service Dockerfile
# Includes PyTorch, OpenCV, YOLO, and RealSense SDK

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    # Build tools for pupil-apriltags
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Intel RealSense SDK (optional - comment out if not using hardware)
RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && \
    apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev || true && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/vision/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
# Note: chess_vision module is mounted as volume in docker-compose
COPY services/vision/main.py .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
