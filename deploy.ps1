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

Write-Host "🧹 Pruning unused Docker resources..." -ForegroundColor Yellow
docker system prune -f
Write-Host "✅ Unused Docker resources cleaned" -ForegroundColor Green

# Load environment variables if .env file exists
if (Test-Path ".env") {
    Write-Host "🔑 Loading environment variables from .env file..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($value) {
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
                Write-Host "  - $key loaded" -ForegroundColor DarkGray
            }
        }
    }
    Write-Host "✅ Environment variables loaded successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️ No .env file found, using default values (not recommended for production)" -ForegroundColor Yellow
}

Write-Host "📦 Building Docker images..." -ForegroundColor Yellow
Write-Host "🔍 Testing dashboard Dockerfile build first..." -ForegroundColor Cyan
docker build -t test-dashboard -f Dockerfile.dashboard.production .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dashboard Docker image built successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Dashboard Docker build failed" -ForegroundColor Red
    Write-Host "🔧 Attempting to fix common issues..." -ForegroundColor Yellow
    
    # Create a compatible yfinance version file if needed
    @"
# Streamlit Cloud Compatible Requirements
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
requests>=2.31.0
python-dotenv>=1.0.0
yfinance==0.2.30
ta>=0.10.0
"@ | Out-File -FilePath "requirements-streamlit.txt" -Encoding utf8
    
    Write-Host "✏️ Updated requirements-streamlit.txt with compatible yfinance version" -ForegroundColor Green
    Write-Host "🔄 Retrying dashboard build..." -ForegroundColor Cyan
    docker build -t test-dashboard -f Dockerfile.dashboard.production .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Dashboard build still failing. Please check the logs above for details." -ForegroundColor Red
        exit 1
    }
}

Write-Host "📦 Building all production Docker images..." -ForegroundColor Yellow
docker compose -f docker-compose.production.yml build --progress=plain

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ All Docker images built successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Starting application with Docker Compose (Production)..." -ForegroundColor Yellow
docker compose -f docker-compose.production.yml up -d

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
    Write-Host "   Stop: docker compose -f docker-compose.production.yml down" -ForegroundColor White
    Write-Host "   Logs: docker compose -f docker-compose.production.yml logs -f" -ForegroundColor White
    Write-Host "   Status: docker compose -f docker-compose.production.yml ps" -ForegroundColor White
} else {
    Write-Host "❌ Failed to start application" -ForegroundColor Red
    exit 1
}
