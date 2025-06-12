# Quantitative Trading Platform Deployment Script (Windows PowerShell)
# Developed by: Malavath Hanmanth Nayak
# Contact: hanmanthnayak.95@gmail.com
# LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

Write-Host "🚀 Quantitative Trading Platform Deployment" -ForegroundColor Green
Write-Host "Developer: Malavath Hanmanth Nayak" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Yellow

# Check if Docker is installed
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is available
try {
    docker compose version | Out-Null
    Write-Host "✅ Docker Compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not available. Please install Docker Desktop with Compose." -ForegroundColor Red
    exit 1
}

Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
docker build -t quant-trading-platform:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker image built successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Starting application with Docker Compose..." -ForegroundColor Yellow
docker compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Application started successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Streamlit Dashboard: http://localhost:8501" -ForegroundColor Cyan
    Write-Host "📊 API Server: http://localhost:8000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔗 Developer Links:" -ForegroundColor Magenta
    Write-Host "   LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/" -ForegroundColor Blue
    Write-Host "   Email: hanmanthnayak.95@gmail.com" -ForegroundColor Blue
    Write-Host ""
    Write-Host "📋 Management Commands:" -ForegroundColor Yellow
    Write-Host "   Stop: docker compose down" -ForegroundColor White
    Write-Host "   Logs: docker compose logs -f" -ForegroundColor White
    Write-Host "   Status: docker compose ps" -ForegroundColor White
} else {
    Write-Host "❌ Failed to start application" -ForegroundColor Red
    exit 1
}
