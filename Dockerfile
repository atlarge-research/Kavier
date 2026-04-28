# Dockerfile for Kavier - LLM Performance Simulator
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not create virtual env (we're in container)
RUN poetry config virtualenvs.create false

# Install dependencies only (skip root project - setuptools will handle it)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application (including src/)
COPY . .

# Install the project using pip (setuptools-based)
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["/bin/bash"]
