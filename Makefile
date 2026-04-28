.PHONY: setup up down shell test clean help

# Default target
.DEFAULT_GOAL := help

# Setup: Build Docker image
setup:
	@echo "Building Kavier Docker image..."
	docker-compose build
	@echo "✅ Setup complete! Run 'make up' to start the container."

# Up: Start container in interactive mode
up:
	@echo "Starting Kavier container..."
	docker-compose up -d
	@echo "✅ Container started! Run 'make shell' to enter the container."
	@echo "   Or run 'make test' to run tests."

# Down: Stop and remove containers
down:
	@echo "Stopping Kavier container..."
	docker-compose down
	@echo "✅ Container stopped and removed."

# Shell: Enter the running container
shell:
	@echo "Entering Kavier container..."
	docker-compose exec kavier /bin/bash

# Test: Run tests inside container
test:
	@echo "Running tests..."
	docker-compose exec kavier python src/tests/test_training_components.py

# Clean: Remove containers, images, and volumes
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --rmi all
	@echo "✅ Cleanup complete."

# Help: Show available commands
help:
	@echo "Kavier Docker Commands:"
	@echo "  make setup    - Build Docker image"
	@echo "  make up       - Start container"
	@echo "  make down     - Stop container"
	@echo "  make shell    - Enter container shell"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove all Docker resources"
	@echo "  make help     - Show this help message"