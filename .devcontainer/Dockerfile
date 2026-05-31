# Base image with CUDA 12.6 support (Ubuntu 24.04 ships Python 3.12 natively)
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# Install Python 3.12 and system deps
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python3-dev curl gcc libffi-dev zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Create a virtualenv at a fixed path so Poetry and VS Code can find it
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"

# Install Poetry
RUN curl -sSL https://install.python-poetry.org -o install-poetry.py \
    && python install-poetry.py \
    && rm install-poetry.py

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies into the venv at build time so rebuilds use Docker cache
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-interaction --no-ansi
