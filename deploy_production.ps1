# Production Deployment Script for Quantitative Trading Platform
# Usage: ./deploy_production.ps1

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Quantitative Trading Platform - Production Deployment" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Developer: Malavath Hanmanth Nayak" -ForegroundColor Yellow
Write-Host "Date: $(Get-Date)" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Cyan

# Check Docker and Docker Compose
try {
    docker --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Docker is not installed or not running!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Docker is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed or not running!" -ForegroundColor Red
    exit 1
}

# Ask for optional environment variables
$deployEnv = $args[0]
if (-not $deployEnv) {
    $deployEnv = Read-Host "Enter deployment environment (production/staging) [production]"
    if (-not $deployEnv) { $deployEnv = "production" }
}

# Create a .env file if it doesn't exist
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating .env file for configuration..." -ForegroundColor Yellow
    $dbPassword = Read-Host "Enter database password [default: postgres]"
    if (-not $dbPassword) { $dbPassword = "postgres" }
    
    $redisPassword = Read-Host "Enter Redis password [default: redispassword]"
    if (-not $redisPassword) { $redisPassword = "redispassword" }
    
    $jwtSecret = Read-Host "Enter JWT secret key [default: jwt_secret_key_change_me]"
    if (-not $jwtSecret) { $jwtSecret = "jwt_secret_key_change_me" }
    
    $apiSecret = Read-Host "Enter API secret key [default: api_secret_key_change_me]"
    if (-not $apiSecret) { $apiSecret = "api_secret_key_change_me" }
    
    $grafanaPassword = Read-Host "Enter Grafana admin password [default: admin]"
    if (-not $grafanaPassword) { $grafanaPassword = "admin" }
    
    # Write to .env file
    @"
# Environment: $deployEnv
# Generated on: $(Get-Date)
ENVIRONMENT=$deployEnv
DB_PASSWORD=$dbPassword
REDIS_PASSWORD=$redisPassword
JWT_SECRET_KEY=$jwtSecret
API_SECRET_KEY=$apiSecret
GRAFANA_ADMIN_PASSWORD=$grafanaPassword
"@ | Out-File -FilePath $envFile -Encoding utf8
    
    Write-Host "✅ .env file created" -ForegroundColor Green
} else {
    Write-Host "✅ Using existing .env configuration" -ForegroundColor Green
}

# Check if the docker-compose.production.yml file exists
if (-not (Test-Path "docker-compose.production.yml")) {
    Write-Host "❌ docker-compose.production.yml file not found!" -ForegroundColor Red
    exit 1
}

# Clean up old containers and volumes
Write-Host "Stopping any existing containers..." -ForegroundColor Yellow
docker compose -f docker-compose.production.yml down --remove-orphans

# Fix the requirements file for the yfinance dependency issue
Write-Host "Fixing requirements for yfinance dependency..." -ForegroundColor Yellow
@"
# Streamlit Cloud Compatible Requirements with Fixed Version
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

# Build or rebuild containers without using cache
Write-Host "Building Docker images (this may take a while)..." -ForegroundColor Yellow
docker build -t trading-dashboard-test -f Dockerfile.dashboard.production . --progress=plain

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Dashboard image build failed. Fixing and retrying..." -ForegroundColor Red
    
    # Update the Dockerfile.dashboard.production to use a more flexible approach
    (Get-Content -Path Dockerfile.dashboard.production) -replace "RUN pip install --no-cache-dir -r requirements-streamlit.txt", "RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-streamlit.txt || pip install --no-cache-dir -r requirements-streamlit.txt --use-deprecated=legacy-resolver" | Set-Content -Path Dockerfile.dashboard.production
    
    # Try building again
    docker build -t trading-dashboard-test -f Dockerfile.dashboard.production . --progress=plain
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Dashboard build still failing. Please check the error messages above." -ForegroundColor Red
        exit 1
    }
}

Write-Host "✅ Dashboard image built successfully. Building full environment..." -ForegroundColor Green
docker compose -f docker-compose.production.yml build --progress=plain

# Start the containers in detached mode
Write-Host "Starting production containers..." -ForegroundColor Yellow
docker compose -f docker-compose.production.yml up -d

# Check the status
Write-Host "Checking container status..." -ForegroundColor Yellow
docker compose -f docker-compose.production.yml ps

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Deployment completed! Use the following commands for management:" -ForegroundColor Green
Write-Host "- View logs: docker compose -f docker-compose.production.yml logs -f" -ForegroundColor White
Write-Host "- Stop services: docker compose -f docker-compose.production.yml down" -ForegroundColor White
Write-Host "- Restart services: docker compose -f docker-compose.production.yml restart" -ForegroundColor White
Write-Host "=====================================" -ForegroundColor Cyan
