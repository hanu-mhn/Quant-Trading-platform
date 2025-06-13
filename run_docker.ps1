# Script to run Docker Compose with error capturing
$logFile = "docker_compose_log.txt"
Write-Host "Running Docker Compose with output logging to $logFile"
docker compose -f docker-compose.production.yml up --build 2>&1 | Tee-Object -FilePath $logFile
