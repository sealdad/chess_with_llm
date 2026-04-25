FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (include cuda-nvcc so deepspeed can build)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git sox libsox-dev ffmpeg curl \
    cuda-nvcc-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone CosyVoice repo for the library code and install its deps
RUN git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice && \
    cd /opt/CosyVoice && \
    pip install --no-cache-dir -r requirements.txt

# Add CosyVoice to Python path
ENV PYTHONPATH="/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

# Copy pre-downloaded model if it exists (glob pattern won't fail if missing)
COPY pretrained_model[s] /app/pretrained_models_tmp/
RUN if [ -d "/app/pretrained_models_tmp/Fun-CosyVoice3-0.5B-2512" ] && \
       [ -n "$(ls -A /app/pretrained_models_tmp/Fun-CosyVoice3-0.5B-2512/ 2>/dev/null)" ]; then \
        echo "Using pre-downloaded model" && \
        mkdir -p /app/pretrained_models && \
        mv /app/pretrained_models_tmp/Fun-CosyVoice3-0.5B-2512 /app/pretrained_models/Fun-CosyVoice3-0.5B-2512; \
    else \
        echo "No local model found, downloading from HuggingFace..." && \
        mkdir -p /app/pretrained_models && \
        python3 -c "from huggingface_hub import snapshot_download; \
        snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
        local_dir='/app/pretrained_models/Fun-CosyVoice3-0.5B-2512')"; \
    fi && \
    rm -rf /app/pretrained_models_tmp

# Verify CosyVoice imports work (fail build early if deps are missing)
RUN python3 -c "from cosyvoice.cli.cosyvoice import CosyVoice; print('[OK] CosyVoice imports work')"

# Copy service code
COPY main.py .

# Create reference voices directory
RUN mkdir -p /app/reference_voices

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
