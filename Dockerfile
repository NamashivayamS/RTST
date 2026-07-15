# ==============================================================================
# Dockerfile for RealTimeSpeechTranslator
# ==============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Install Python and system dependencies
# ------------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    gnupg2 \
    apt-transport-https \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC driver for SQL Server (msodbcsql18) and unixodbc-dev
RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------------------------
# Working directory
# ------------------------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------------------------
# Copy requirements first for Docker layer caching
# ------------------------------------------------------------------------------

COPY requirements.txt .

# ------------------------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------------------------

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------------------------
# Download SpaCy model
# ------------------------------------------------------------------------------

RUN python3 -m spacy download en_core_web_sm

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

COPY download_models.py .
RUN python3 download_models.py && rm download_models.py

ENV HF_TOKEN=""

# ------------------------------------------------------------------------------
# Offline HuggingFace configuration
# ------------------------------------------------------------------------------

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------------------
# Copy project
# ------------------------------------------------------------------------------

COPY . .

# ------------------------------------------------------------------------------
# Expose FastAPI port
# ------------------------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------------------------
# Start application
# ------------------------------------------------------------------------------

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]