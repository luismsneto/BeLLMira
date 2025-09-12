# ----------------------------
# Stage 1: Build dependencies
# ----------------------------
FROM python:3.12-slim AS builder

# Install Poetry
RUN apt-get update && apt-get install -y curl gcc libffi-dev
RUN curl -sSL https://install.python-poetry.org -o install-poetry.py
RUN python3 install-poetry.py

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set workdir
WORKDIR /app

# Copy only pyproject.toml and poetry.lock first to install deps
COPY pyproject.toml poetry.lock* /app/

# Install dependencies into a virtual environment in /opt
#RUN poetry config virtualenvs.create false \
#    && poetry install --only main --no-interaction --no-ansi
#RUN poetry install --only main --no-interaction --no-ansi
# --------------------------
# Stage 2: Run-time image
# --------------------------
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy installed deps from builder stage
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your source code
COPY . /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default run command (adjust if needed)
#CMD ["python", "main.py"]
