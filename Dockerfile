# Dockerfile for Kavier - LLM Performance Simulator
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the application (including src/). pyproject.toml is the single source of
# dependencies; pip resolves and installs them from it.
COPY . .

# Install the project (setuptools-based) plus its dependencies from pyproject.
RUN python -m pip install --upgrade pip && \
    pip install -e .

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["/bin/bash"]
