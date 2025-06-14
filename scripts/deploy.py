#!/usr/bin/env python3
"""
Production deployment script for the quantitative trading platform.
This script automates the deployment process including environment setup,
security configuration, and service orchestration.
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
import secrets
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingPlatformDeployer:
    """Main deployment orchestrator for the trading platform."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.env_file = self.project_root / f".env.{environment}"
        self.docker_compose_file = self.project_root / "docker-compose.yml"
        
    def generate_secrets(self) -> Dict[str, str]:
        """Generate secure random secrets for the deployment."""
        logger.info("Generating deployment secrets...")
        
        secrets_config = {
            'DB_PASSWORD': secrets.token_urlsafe(32),
            'JWT_SECRET_KEY': secrets.token_urlsafe(64),
            'GRAFANA_ADMIN_PASSWORD': secrets.token_urlsafe(16),
            'POSTGRES_PASSWORD': secrets.token_urlsafe(32),
            'REDIS_PASSWORD': secrets.token_urlsafe(32),
            'API_SECRET_KEY': secrets.token_urlsafe(64),
            'ENCRYPTION_KEY': secrets.token_urlsafe(32),
            'WEBHOOK_SECRET': secrets.token_urlsafe(32),
            'SESSION_SECRET': secrets.token_urlsafe(32),
            'MONITORING_TOKEN': secrets.token_urlsafe(24)
        }
        
        # Generate SSL certificate paths (if using HTTPS)
        ssl_config = {
            'SSL_CERT_PATH': '/etc/ssl/certs/trading-platform.crt',
            'SSL_KEY_PATH': '/etc/ssl/private/trading-platform.key',
            'SSL_CA_PATH': '/etc/ssl/certs/ca-certificates.crt'
        }
        
        secrets_config.update(ssl_config)
        return secrets_config
    
    def create_environment_file(self, secrets: Dict[str, str]):
        """Create environment configuration file."""
        logger.info(f"Creating environment file: {self.env_file}")
        
        env_config = {
            # Application Configuration
            'ENVIRONMENT': self.environment,
            'DEBUG': 'false' if self.environment == 'production' else 'true',
            'LOG_LEVEL': 'INFO' if self.environment == 'production' else 'DEBUG',
            
            # Database Configuration
            'DATABASE_URL': f"postgresql://trader:{secrets['DB_PASSWORD']}@database:5432/trading_platform",
            'DB_HOST': 'database',
            'DB_PORT': '5432',
            'DB_NAME': 'trading_platform',
            'DB_USER': 'trader',
            'DB_PASSWORD': secrets['DB_PASSWORD'],
            
            # Redis Configuration
            'REDIS_URL': f"redis://:{secrets['REDIS_PASSWORD']}@redis:6379/0",
            'REDIS_HOST': 'redis',
            'REDIS_PORT': '6379',
            'REDIS_PASSWORD': secrets['REDIS_PASSWORD'],
            
            # API Configuration
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
            'API_WORKERS': '4',
            'API_SECRET_KEY': secrets['API_SECRET_KEY'],
            'JWT_SECRET_KEY': secrets['JWT_SECRET_KEY'],
            'JWT_EXPIRATION_HOURS': '24',
            
            # Dashboard Configuration
            'DASHBOARD_HOST': '0.0.0.0',
            'DASHBOARD_PORT': '8501',
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_ENABLE_CORS': 'false',
            
            # Monitoring Configuration
            'GRAFANA_ADMIN_USER': 'admin',
            'GRAFANA_ADMIN_PASSWORD': secrets['GRAFANA_ADMIN_PASSWORD'],
            'PROMETHEUS_RETENTION_TIME': '30d',
            'PROMETHEUS_RETENTION_SIZE': '10GB',
            
            # Security Configuration
            'ENCRYPTION_KEY': secrets['ENCRYPTION_KEY'],
            'SESSION_SECRET': secrets['SESSION_SECRET'],
            'WEBHOOK_SECRET': secrets['WEBHOOK_SECRET'],
            'CORS_ORIGINS': 'http://localhost:3000,https://your-domain.com',
            
            # Trading Configuration
            'PAPER_TRADING_ENABLED': 'true',
            'LIVE_TRADING_ENABLED': 'false',  # Enable only after thorough testing
            'MAX_POSITION_SIZE': '0.1',  # 10% of portfolio
            'MAX_DAILY_LOSS': '0.05',   # 5% daily loss limit
            'RISK_FREE_RATE': '0.02',   # 2% risk-free rate
            
            # Broker Configuration (Interactive Brokers)
            'IB_GATEWAY_HOST': 'ib-gateway',
            'IB_GATEWAY_PORT': '4001',
            'IB_CLIENT_ID': '1',
            'IB_ACCOUNT_ID': '',  # Set this manually
            
            # Market Data Configuration
            'MARKET_DATA_PROVIDER': 'yahoo',  # yahoo, alpha_vantage, iex
            'ALPHA_VANTAGE_API_KEY': '',      # Set if using Alpha Vantage
            'IEX_API_KEY': '',                # Set if using IEX
            
            # Notification Configuration
            'ENABLE_EMAIL_NOTIFICATIONS': 'false',
            'SMTP_HOST': '',
            'SMTP_PORT': '587',
            'SMTP_USER': '',
            'SMTP_PASSWORD': '',
            'NOTIFICATION_EMAIL': '',
            
            # Backup Configuration
            'BACKUP_ENABLED': 'true',
            'BACKUP_SCHEDULE': '0 2 * * *',  # Daily at 2 AM
            'BACKUP_RETENTION_DAYS': '30',
            'S3_BACKUP_BUCKET': '',          # Optional S3 backup
            'S3_ACCESS_KEY': '',
            'S3_SECRET_KEY': '',
            
            # Performance Configuration
            'WORKER_PROCESSES': '4',
            'WORKER_CONNECTIONS': '1000',
            'MAX_REQUESTS': '1000',
            'MAX_REQUESTS_JITTER': '100',
            'PRELOAD_APP': 'true',
            
            # SSL Configuration
            'SSL_ENABLED': 'false',  # Set to true for production with valid certificates
            'SSL_CERT_PATH': secrets['SSL_CERT_PATH'],
            'SSL_KEY_PATH': secrets['SSL_KEY_PATH'],
            'SSL_CA_PATH': secrets['SSL_CA_PATH']
        }
        
        # Write environment file
        with open(self.env_file, 'w') as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")
        
        # Secure the environment file
        os.chmod(self.env_file, 0o600)
        logger.info("Environment file created and secured")
    
    def deploy_services(self):
        """Deploy all services using Docker Compose."""
        logger.info("Deploying all services...")
        
        # Start all services
        self.run_command(['docker-compose', 'up', '-d'])
        
        logger.info("All services deployed successfully")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a system command with logging."""
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.stdout:
            logger.info(f"Command output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr.strip()}")
        
        return result

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy quantitative trading platform")
    parser.add_argument('--environment', '-e', default='production',
                       choices=['development', 'staging', 'production'],
                       help='Deployment environment')
    
    args = parser.parse_args()
    
    try:
        deployer = TradingPlatformDeployer(args.environment)
        
        # Generate secrets and create environment file
        secrets = deployer.generate_secrets()
        deployer.create_environment_file(secrets)
        
        # Deploy all services
        deployer.deploy_services()
        
        logger.info("Deployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()