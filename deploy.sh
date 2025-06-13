#!/bin/bash
# Quantitative Trading Platform Deployment Script
# Developed by: Malavath Hanmanth Nayak
# Contact: hanmanthnayak.95@gmail.com
# LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

echo "🚀 Quantitative Trading Platform Deployment"
echo "Developer: Malavath Hanmanth Nayak"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker environment is ready"

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t quant-trading-platform:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Run with Docker Compose (Production)
echo "🚀 Starting application with Docker Compose (Production)..."
docker compose -f docker-compose.production.yml up -d

if [ $? -eq 0 ]; then
    echo "✅ Application started successfully"
    echo "🌐 Streamlit Dashboard: http://localhost:8501"
    echo "📊 API Server: http://localhost:8000"
    echo ""
    echo "🔗 Developer Links:"
    echo "   LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/"
    echo "   Email: hanmanthnayak.95@gmail.com"
    echo ""
    echo "📋 To stop the application: docker compose -f docker-compose.production.yml down"
    echo "📋 To view logs: docker compose -f docker-compose.production.yml logs -f"
    echo "📋 To check status: docker compose -f docker-compose.production.yml ps"
else
    echo "❌ Failed to start application"
    exit 1
fi
