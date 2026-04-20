#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Building Docker image..."
docker build -t vehicle-detection-app .

echo "Stopping and removing existing container (if any)..."
docker stop vehicle-detection-container 2>/dev/null || true
docker rm vehicle-detection-container 2>/dev/null || true

echo "Starting new container on port 8502..."
docker run -d \
  --name vehicle-detection-container \
  --restart unless-stopped \
  -p 8502:8502 \
  vehicle-detection-app

echo "================================================"
echo "Deployment successful!"
echo "Application is running at http://localhost:8502"
echo "To view logs, run: docker logs -f vehicle-detection-container"
echo "================================================"
