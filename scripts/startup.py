#!/usr/bin/env python3
"""
Production startup script for the quantitative trading platform.
This script provides a unified interface for starting, stopping, and managing
the trading platform in production environments.
"""

import os
import sys
import time
import subprocess
import argparse
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingPlatformManager:
    """Production management interface for the trading platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docker_compose_file = self.project_root / "docker-compose.yml"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met for running the platform."""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            logger.info("✓ Docker is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ Docker not found. Please install Docker.")
            return False
        
        # Check Docker Compose
        try:
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
            logger.info("✓ Docker Compose is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ Docker Compose not found. Please install Docker Compose.")
            return False
        
        # Check required files
        required_files = [
            'docker-compose.yml',
            'requirements.txt',
            'scripts/init_db.sql',
            'nginx/nginx.conf'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                logger.error(f"✗ Required file missing: {file_path}")
                return False
            logger.info(f"✓ {file_path} found")
        
        return True
    
    def start_platform(self, environment: str = "production") -> bool:
        """Start the trading platform with all services."""
        logger.info(f"Starting trading platform in {environment} mode...")
        
        if not self.check_prerequisites():
            return False
        
        # Check if environment file exists
        env_file = self.project_root / f".env.{environment}"
        if not env_file.exists():
            logger.warning(f"Environment file not found: {env_file}")
            logger.info("Generating environment configuration...")
            
            # Run deployment script to generate environment
            try:
                subprocess.run([
                    'python', 'scripts/deploy.py', 
                    '--environment', environment,
                    '--generate-secrets-only'
                ], check=True, cwd=self.project_root)
                logger.info("✓ Environment configuration generated")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to generate environment: {e}")
                return False
        
        # Start services
        try:
            logger.info("Starting Docker services...")
            result = subprocess.run([
                'docker-compose', '--env-file', f'.env.{environment}', 'up', '-d'
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            logger.info("✓ Services started successfully")
            
            # Wait for services to be ready
            logger.info("Waiting for services to initialize...")
            time.sleep(30)
            
            # Check service health
            if self.check_service_health():
                logger.info("🎉 Trading platform started successfully!")
                self.display_access_info()
                return True
            else:
                logger.error("❌ Some services failed to start properly")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to start services: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def stop_platform(self) -> bool:
        """Stop the trading platform and all services."""
        logger.info("Stopping trading platform...")
        
        try:
            subprocess.run([
                'docker-compose', 'down'
            ], check=True, capture_output=True, cwd=self.project_root)
            
            logger.info("✓ Platform stopped successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to stop platform: {e}")
            return False
    
    def restart_platform(self, environment: str = "production") -> bool:
        """Restart the trading platform."""
        logger.info("Restarting trading platform...")
        
        if self.stop_platform():
            time.sleep(5)  # Brief pause
            return self.start_platform(environment)
        
        return False
    
    def check_service_health(self) -> bool:
        """Check the health of all services."""
        logger.info("Checking service health...")
        
        services = {
            'API': 'http://localhost:8000/health',
            'Dashboard': 'http://localhost:8501',
            'Prometheus': 'http://localhost:9090/-/healthy',
            'Grafana': 'http://localhost:3000/api/health'
        }
        
        all_healthy = True
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} is healthy")
                else:
                    logger.warning(f"⚠ {service_name} returned status {response.status_code}")
                    all_healthy = False
            except requests.RequestException as e:
                logger.error(f"✗ {service_name} is not responding: {e}")
                all_healthy = False
        
        return all_healthy
    
    def show_status(self) -> None:
        """Show the current status of the platform."""
        logger.info("Checking platform status...")
        
        try:
            # Get Docker Compose status
            result = subprocess.run([
                'docker-compose', 'ps'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            print("\n" + "="*60)
            print("TRADING PLATFORM STATUS")
            print("="*60)
            print(result.stdout)
            
            # Check service health
            self.check_service_health()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get status: {e}")
    
    def show_logs(self, service: Optional[str] = None, follow: bool = False) -> None:
        """Show logs for services."""
        logger.info(f"Showing logs for {service or 'all services'}...")
        
        cmd = ['docker-compose', 'logs']
        
        if follow:
            cmd.append('-f')
        
        if service:
            cmd.append(service)
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("Log viewing interrupted")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to show logs: {e}")
    
    def backup_data(self) -> bool:
        """Create a backup of the platform data."""
        logger.info("Creating platform backup...")
        
        backup_script = self.project_root / 'scripts' / 'backup.sh'
        
        if backup_script.exists():
            try:
                subprocess.run(['bash', str(backup_script)], check=True)
                logger.info("✓ Backup completed successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Backup failed: {e}")
                return False
        else:
            logger.error("✗ Backup script not found")
            return False
    
    def update_platform(self) -> bool:
        """Update the platform to the latest version."""
        logger.info("Updating trading platform...")
        
        try:
            # Pull latest code
            subprocess.run(['git', 'pull'], check=True, cwd=self.project_root)
            
            # Pull latest Docker images
            subprocess.run(['docker-compose', 'pull'], check=True, cwd=self.project_root)
            
            # Rebuild and restart
            subprocess.run(['docker-compose', 'up', '-d', '--build'], 
                         check=True, cwd=self.project_root)
            
            logger.info("✓ Platform updated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Update failed: {e}")
            return False
    
    def display_access_info(self) -> None:
        """Display access information for the platform."""
        print("\n" + "="*60)
        print("🚀 TRADING PLATFORM ACCESS INFORMATION")
        print("="*60)
        print("📊 Trading Dashboard: http://localhost:8501")
        print("🔌 REST API:          http://localhost:8000")
        print("📈 API Documentation: http://localhost:8000/docs")
        print("📊 Grafana:          http://localhost:3000")
        print("📈 Prometheus:       http://localhost:9090")
        print("🌐 Main Portal:      http://localhost")
        print("\n💡 Default Credentials:")
        print("   Grafana: admin / [check .env file]")
        print("\n📋 Useful Commands:")
        print("   View logs:    python scripts/startup.py logs")
        print("   Check status: python scripts/startup.py status")
        print("   Stop platform: python scripts/startup.py stop")
        print("="*60)

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Trading Platform Manager")
    parser.add_argument('command', choices=[
        'start', 'stop', 'restart', 'status', 'logs', 'backup', 'update', 'health'
    ], help='Command to execute')
    parser.add_argument('--environment', '-e', default='production',
                       choices=['development', 'staging', 'production'],
                       help='Environment to use')
    parser.add_argument('--service', '-s', help='Specific service for logs command')
    parser.add_argument('--follow', '-f', action='store_true',
                       help='Follow logs in real-time')
    
    args = parser.parse_args()
    
    manager = TradingPlatformManager()
    
    try:
        if args.command == 'start':
            success = manager.start_platform(args.environment)
            sys.exit(0 if success else 1)
            
        elif args.command == 'stop':
            success = manager.stop_platform()
            sys.exit(0 if success else 1)
            
        elif args.command == 'restart':
            success = manager.restart_platform(args.environment)
            sys.exit(0 if success else 1)
            
        elif args.command == 'status':
            manager.show_status()
            
        elif args.command == 'logs':
            manager.show_logs(args.service, args.follow)
            
        elif args.command == 'backup':
            success = manager.backup_data()
            sys.exit(0 if success else 1)
            
        elif args.command == 'update':
            success = manager.update_platform()
            sys.exit(0 if success else 1)
            
        elif args.command == 'health':
            healthy = manager.check_service_health()
            sys.exit(0 if healthy else 1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
