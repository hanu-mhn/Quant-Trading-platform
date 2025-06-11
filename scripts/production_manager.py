#!/usr/bin/env python3
"""
Production Startup Script for Quantitative Trading Platform

This script provides a comprehensive production management interface for:
- Starting/stopping services
- Health monitoring
- Log management
- Backup operations
- Performance monitoring
"""

import os
import sys
import time
import json
import yaml
import subprocess
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor
import argparse


class ProductionManager:
    """Production environment management for the trading platform."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / 'config' / 'production.yaml'
        self.services_file = self.project_root / 'config' / 'services.yaml'
        self.pid_dir = self.project_root / 'run'
        self.log_dir = self.project_root / 'logs'
        
        # Create directories
        self.pid_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        self.services = self.load_services_config()
        
        # Service status cache
        self.service_status = {}
        
    def setup_logging(self):
        """Setup logging for production manager."""
        log_file = self.log_dir / 'production.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ProductionManager')
    
    def load_config(self) -> Dict[str, Any]:
        """Load production configuration."""
        default_config = {
            'environment': 'production',
            'health_check': {
                'interval': 30,
                'timeout': 10,
                'retries': 3
            },
            'monitoring': {
                'enabled': True,
                'metrics_interval': 60,
                'alert_thresholds': {
                    'cpu_percent': 80,
                    'memory_percent': 85,
                    'disk_percent': 90
                }
            },
            'backup': {
                'enabled': True,
                'schedule': '0 2 * * *',  # Daily at 2 AM
                'retention_days': 30
            },
            'logging': {
                'level': 'INFO',
                'max_size_mb': 100,
                'backup_count': 10
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def load_services_config(self) -> Dict[str, Any]:
        """Load services configuration."""
        default_services = {
            'database': {
                'command': 'docker-compose up -d database',
                'health_check': 'http://localhost:5432',
                'dependencies': [],
                'startup_time': 30
            },
            'redis': {
                'command': 'docker-compose up -d redis',
                'health_check': 'tcp://localhost:6379',
                'dependencies': [],
                'startup_time': 10
            },
            'api_server': {
                'command': 'python -m uvicorn src.api.api_server:app --host 0.0.0.0 --port 8000',
                'health_check': 'http://localhost:8000/health',
                'dependencies': ['database', 'redis'],
                'startup_time': 20
            },
            'dashboard': {
                'command': 'streamlit run src/dashboard/trading_dashboard.py --server.port 8501',
                'health_check': 'http://localhost:8501',
                'dependencies': ['api_server'],
                'startup_time': 15
            },
            'prometheus': {
                'command': 'docker-compose up -d prometheus',
                'health_check': 'http://localhost:9090',
                'dependencies': [],
                'startup_time': 15
            },
            'grafana': {
                'command': 'docker-compose up -d grafana',
                'health_check': 'http://localhost:3000',
                'dependencies': ['prometheus'],
                'startup_time': 20
            }
        }
        
        if self.services_file.exists():
            try:
                with open(self.services_file, 'r') as f:
                    user_services = yaml.safe_load(f)
                    default_services.update(user_services)
            except Exception as e:
                self.logger.warning(f"Error loading services config: {e}. Using defaults.")
        
        return default_services
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            self.logger.error(f"Unknown service: {service_name}")
            return False
        
        service_config = self.services[service_name]
        
        # Check dependencies
        for dep in service_config.get('dependencies', []):
            if not self.is_service_running(dep):
                self.logger.info(f"Starting dependency: {dep}")
                if not self.start_service(dep):
                    self.logger.error(f"Failed to start dependency: {dep}")
                    return False
        
        self.logger.info(f"Starting service: {service_name}")
        
        try:
            # Start the service
            command = service_config['command']
            
            if command.startswith('docker-compose'):
                # Use docker-compose
                process = subprocess.Popen(
                    command.split(),
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                # Start as background process
                log_file = self.log_dir / f"{service_name}.log"
                
                with open(log_file, 'a') as f:
                    process = subprocess.Popen(
                        command.split(),
                        cwd=self.project_root,
                        stdout=f,
                        stderr=subprocess.STDOUT
                    )
                
                # Save PID
                pid_file = self.pid_dir / f"{service_name}.pid"
                with open(pid_file, 'w') as f:
                    f.write(str(process.pid))
            
            # Wait for startup
            startup_time = service_config.get('startup_time', 30)
            self.logger.info(f"Waiting {startup_time}s for {service_name} to start...")
            time.sleep(startup_time)
            
            # Verify service is running
            if self.check_service_health(service_name):
                self.logger.info(f"Service {service_name} started successfully")
                return True
            else:
                self.logger.error(f"Service {service_name} failed health check")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            self.logger.error(f"Unknown service: {service_name}")
            return False
        
        self.logger.info(f"Stopping service: {service_name}")
        
        try:
            service_config = self.services[service_name]
            command = service_config['command']
            
            if command.startswith('docker-compose'):
                # Stop docker-compose service
                stop_command = command.replace('up -d', 'stop').split()
                subprocess.run(stop_command, cwd=self.project_root, check=True)
            else:
                # Stop process using PID
                pid_file = self.pid_dir / f"{service_name}.pid"
                if pid_file.exists():
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    
                    try:
                        # Try graceful shutdown first
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(5)
                        
                        # Check if still running
                        if psutil.pid_exists(pid):
                            os.kill(pid, signal.SIGKILL)
                        
                        pid_file.unlink()
                        
                    except (ProcessLookupError, psutil.NoSuchProcess):
                        # Process already dead
                        pid_file.unlink()
            
            self.logger.info(f"Service {service_name} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service {service_name}: {e}")
            return False
    
    def is_service_running(self, service_name: str) -> bool:
        """Check if a service is running."""
        if service_name not in self.services:
            return False
        
        service_config = self.services[service_name]
        command = service_config['command']
        
        if command.startswith('docker-compose'):
            # Check docker container status
            try:
                result = subprocess.run([
                    'docker-compose', 'ps', '--services', '--filter', 'status=running'
                ], cwd=self.project_root, capture_output=True, text=True)
                
                return service_name in result.stdout
                
            except subprocess.CalledProcessError:
                return False
        else:
            # Check PID file
            pid_file = self.pid_dir / f"{service_name}.pid"
            if not pid_file.exists():
                return False
            
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                return psutil.pid_exists(pid)
                
            except (ValueError, FileNotFoundError):
                return False
    
    def check_service_health(self, service_name: str) -> bool:
        """Check service health via HTTP endpoint or other means."""
        if service_name not in self.services:
            return False
        
        health_check = self.services[service_name].get('health_check')
        if not health_check:
            return self.is_service_running(service_name)
        
        try:
            if health_check.startswith('http'):
                response = requests.get(
                    health_check,
                    timeout=self.config['health_check']['timeout']
                )
                return response.status_code == 200
                
            elif health_check.startswith('tcp'):
                # TCP connection check
                import socket
                host, port = health_check.replace('tcp://', '').split(':')
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config['health_check']['timeout'])
                result = sock.connect_ex((host, int(port)))
                sock.close()
                return result == 0
            
            else:
                return self.is_service_running(service_name)
                
        except Exception as e:
            self.logger.warning(f"Health check failed for {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all services in dependency order."""
        self.logger.info("Starting all services...")
        
        # Build dependency graph
        dependency_order = self.get_startup_order()
        
        success = True
        for service_name in dependency_order:
            if not self.start_service(service_name):
                success = False
                break
        
        if success:
            self.logger.info("All services started successfully")
        else:
            self.logger.error("Failed to start all services")
        
        return success
    
    def stop_all_services(self) -> bool:
        """Stop all services in reverse dependency order."""
        self.logger.info("Stopping all services...")
        
        # Reverse dependency order for shutdown
        dependency_order = list(reversed(self.get_startup_order()))
        
        success = True
        for service_name in dependency_order:
            if self.is_service_running(service_name):
                if not self.stop_service(service_name):
                    success = False
        
        if success:
            self.logger.info("All services stopped successfully")
        else:
            self.logger.error("Failed to stop all services")
        
        return success
    
    def get_startup_order(self) -> List[str]:
        """Get services in dependency order for startup."""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(service_name):
            if service_name in visited:
                return
            
            visited.add(service_name)
            
            for dep in self.services.get(service_name, {}).get('dependencies', []):
                visit(dep)
            
            order.append(service_name)
        
        for service_name in self.services:
            visit(service_name)
        
        return order
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        status = {}
        
        for service_name in self.services:
            status[service_name] = {
                'running': self.is_service_running(service_name),
                'healthy': self.check_service_health(service_name),
                'uptime': self.get_service_uptime(service_name)
            }
        
        return status
    
    def get_service_uptime(self, service_name: str) -> Optional[float]:
        """Get service uptime in seconds."""
        pid_file = self.pid_dir / f"{service_name}.pid"
        if not pid_file.exists():
            return None
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                return time.time() - process.create_time()
        
        except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
            pass
        
        return None
    
    def monitor_system(self) -> Dict[str, Any]:
        """Monitor system resources."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'network': psutil.net_io_counters()._asdict(),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_backup(self) -> bool:
        """Create system backup."""
        try:
            from .backup_restore import BackupManager
            
            backup_manager = BackupManager()
            backup_path = backup_manager.create_backup('daily')
            
            self.logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        self.logger.info(f"Restarting service: {service_name}")
        
        # Stop service
        if self.is_service_running(service_name):
            if not self.stop_service(service_name):
                return False
        
        # Wait a moment
        time.sleep(2)
        
        # Start service
        return self.start_service(service_name)
    
    def show_logs(self, service_name: str, lines: int = 50):
        """Show service logs."""
        log_file = self.log_dir / f"{service_name}.log"
        
        if not log_file.exists():
            print(f"No log file found for service: {service_name}")
            return
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                print(f"=== Last {len(recent_lines)} lines of {service_name} ===")
                for line in recent_lines:
                    print(line.rstrip())
                    
        except Exception as e:
            print(f"Error reading log file: {e}")


def main():
    """Main function for production management."""
    parser = argparse.ArgumentParser(description='Production Management for Trading Platform')
    parser.add_argument('command', choices=[
        'start', 'stop', 'restart', 'status', 'health', 
        'logs', 'monitor', 'backup', 'start-all', 'stop-all'
    ], help='Command to execute')
    parser.add_argument('--service', help='Specific service name')
    parser.add_argument('--lines', type=int, default=50, help='Number of log lines to show')
    
    args = parser.parse_args()
    
    manager = ProductionManager()
    
    try:
        if args.command == 'start':
            if not args.service:
                print("Error: --service is required for start command")
                return 1
            success = manager.start_service(args.service)
            return 0 if success else 1
        
        elif args.command == 'stop':
            if not args.service:
                print("Error: --service is required for stop command")
                return 1
            success = manager.stop_service(args.service)
            return 0 if success else 1
        
        elif args.command == 'restart':
            if not args.service:
                print("Error: --service is required for restart command")
                return 1
            success = manager.restart_service(args.service)
            return 0 if success else 1
        
        elif args.command == 'start-all':
            success = manager.start_all_services()
            return 0 if success else 1
        
        elif args.command == 'stop-all':
            success = manager.stop_all_services()
            return 0 if success else 1
        
        elif args.command == 'status':
            status = manager.get_service_status()
            
            print(f"{'Service':<15} {'Running':<10} {'Healthy':<10} {'Uptime':<15}")
            print("-" * 50)
            
            for service_name, service_status in status.items():
                running = "✓" if service_status['running'] else "✗"
                healthy = "✓" if service_status['healthy'] else "✗"
                uptime = f"{service_status['uptime']:.1f}s" if service_status['uptime'] else "N/A"
                
                print(f"{service_name:<15} {running:<10} {healthy:<10} {uptime:<15}")
        
        elif args.command == 'health':
            status = manager.get_service_status()
            all_healthy = all(s['healthy'] for s in status.values())
            
            if all_healthy:
                print("✓ All services are healthy")
                return 0
            else:
                print("✗ Some services are unhealthy")
                for service_name, service_status in status.items():
                    if not service_status['healthy']:
                        print(f"  - {service_name}: unhealthy")
                return 1
        
        elif args.command == 'logs':
            if not args.service:
                print("Error: --service is required for logs command")
                return 1
            manager.show_logs(args.service, args.lines)
        
        elif args.command == 'monitor':
            system_info = manager.monitor_system()
            
            print("=== System Monitoring ===")
            print(f"CPU Usage: {system_info['cpu_percent']:.1f}%")
            print(f"Memory Usage: {system_info['memory']['percent']:.1f}%")
            print(f"Disk Usage: {system_info['disk']['percent']:.1f}%")
            print(f"Timestamp: {system_info['timestamp']}")
        
        elif args.command == 'backup':
            success = manager.create_backup()
            return 0 if success else 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
