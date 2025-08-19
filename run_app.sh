#!/bin/bash

# List of ports to check
ports=(3000 8000)

echo "🔍 Checking ports: ${ports[*]}"

for port in "${ports[@]}"; do
  if lsof -i ":$port" > /dev/null; then
    echo "❌ Port $port is IN USE"
  else
    echo "✅ Port $port is FREE"
  fi
done

echo "🔍 Checking if Docker is installed..."

# Check if docker command exists
if ! command -v docker &> /dev/null; then
  echo "❌ Docker is NOT installed. Please install Docker first."
  exit 1
else
  echo "✅ Docker is installed."
fi

echo "🔍 Verifying Docker can run commands (checking 'docker ps')..."

# Check if Docker daemon is running and accessible
if docker ps &> /dev/null; then
  echo "✅ Docker daemon is running and accessible."
else
  echo "❌ Docker command failed. Docker may not be running or you may not have sufficient permissions."
  echo "   Try running: 'open --background -a Docker' or check your Docker setup."
  exit 1
fi

# Build Backend
echo "🛠️ Building backend Docker image..."
docker build backend -t theogregoir/who-should-i-hire-backend:latest

# Build Frontend
echo "🛠️ Building frontend Docker image..."
docker build frontend -t theogregoir/who-should-i-hire-frontend:latest

# Run Backend
echo "🚀 Starting backend container on port 8000..."
docker run -d --rm \
    -p 8000:8000 \
    --env-file ./backend/.env \
    -v ./backend/data:/data \
    -v ./backend:/app \
    theogregoir/who-should-i-hire-backend:latest
echo "Backend container launched. It will accept incoming calls on http://localhost:8000 !"

# Run Frontend
echo "🚀 Starting frontend container on port 3000..."
docker run -d --rm \
  --name semantic-frontend \
  -p 3000:3000 \
  theogregoir/who-should-i-hire-frontend:latest &
echo "Frontend container launched. You can access it at http://localhost:3000 !"