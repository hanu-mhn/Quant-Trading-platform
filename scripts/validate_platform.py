#!/usr/bin/env python3
"""
Comprehensive validation script for the quantitative trading platform.
This script validates all components, configurations, and dependencies
to ensure the platform is ready for deployment and operation.
"""

import os
import sys
import subprocess
import json
import yaml
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util
import psutil
import sqlite3
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlatformValidator:
    """Comprehensive platform validation orchestrator."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_environment(self) -> bool:
        """Validate the system environment and dependencies."""
        logger.info("Validating system environment...")
        
        results = {}
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 10):
            results['python_version'] = f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        else:
            self.errors.append(f"Python version {python_version.major}.{python_version.minor} is too old. Requires 3.10+")
            results['python_version'] = f"✗ Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        
        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        cpu_count = psutil.cpu_count()
        
        results['system_resources'] = {
            'memory_gb': f"✓ {memory_gb:.1f} GB RAM" if memory_gb >= 8 else f"⚠ {memory_gb:.1f} GB RAM (recommend 8GB+)",
            'disk_gb': f"✓ {disk_gb:.1f} GB free" if disk_gb >= 50 else f"⚠ {disk_gb:.1f} GB free (recommend 50GB+)",
            'cpu_count': f"✓ {cpu_count} CPU cores"
        }
        
        if memory_gb < 4:
            self.errors.append(f"Insufficient memory: {memory_gb:.1f}GB (minimum 4GB required)")
        elif memory_gb < 8:
            self.warnings.append(f"Low memory: {memory_gb:.1f}GB (recommend 8GB+)")
            
        if disk_gb < 20:
            self.errors.append(f"Insufficient disk space: {disk_gb:.1f}GB (minimum 20GB required)")
        elif disk_gb < 50:
            self.warnings.append(f"Low disk space: {disk_gb:.1f}GB (recommend 50GB+)")
        
        # Check Docker availability
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            results['docker'] = "✓ Docker available"
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.errors.append("Docker not found or not working")
            results['docker'] = "✗ Docker unavailable"
        
        # Check Docker Compose availability
        try:
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
            results['docker_compose'] = "✓ Docker Compose available"
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.errors.append("Docker Compose not found or not working")
            results['docker_compose'] = "✗ Docker Compose unavailable"
        
        self.validation_results['environment'] = results
        return len(self.errors) == 0
    
    def validate_python_dependencies(self) -> bool:
        """Validate Python package dependencies."""
        logger.info("Validating Python dependencies...")
        
        results = {}
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.errors.append("requirements.txt not found")
            self.validation_results['python_dependencies'] = {"requirements_file": "✗ Not found"}
            return False
        
        # Read requirements
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        results['total_packages'] = len(requirements)
        missing_packages = []
          # Check critical packages
        critical_packages = {
            'pandas': 'pandas', 
            'numpy': 'numpy', 
            'scipy': 'scipy', 
            'scikit-learn': 'sklearn',
            'fastapi': 'fastapi', 
            'uvicorn': 'uvicorn', 
            'streamlit': 'streamlit',
            'psycopg2-binary': 'psycopg2', 
            'redis': 'redis', 
            'sqlalchemy': 'sqlalchemy',
            'pytest': 'pytest', 
            'docker': 'docker', 
            'prometheus-client': 'prometheus_client'
        }
        
        for package_name, import_name in critical_packages.items():
            try:
                __import__(import_name)
                results[package_name] = f"✓ {package_name}"
            except ImportError:
                missing_packages.append(package_name)
                results[package_name] = f"✗ {package_name} missing"
        
        if missing_packages:
            self.errors.append(f"Missing critical packages: {', '.join(missing_packages)}")
        
        self.validation_results['python_dependencies'] = results
        return len(missing_packages) == 0
    
    def validate_file_structure(self) -> bool:
        """Validate project file structure and required files."""
        logger.info("Validating file structure...")
        
        results = {}
        
        # Required files and directories
        required_paths = {
            'docker-compose.yml': 'file',
            'requirements.txt': 'file',
            'Dockerfile': 'file',
            'Dockerfile.dashboard': 'file',
            'src/': 'directory',
            'tests/': 'directory',
            'scripts/': 'directory',
            'monitoring/': 'directory',
            'nginx/': 'directory',
            'src/api/api_server.py': 'file',
            'src/dashboard/trading_dashboard.py': 'file',
            'src/trading/paper_trading.py': 'file',
            'scripts/init_db.sql': 'file',
            'nginx/nginx.conf': 'file',
            'monitoring/prometheus/prometheus.yml': 'file'
        }
        
        for path, path_type in required_paths.items():
            full_path = self.project_root / path
            
            if path_type == 'file':
                if full_path.is_file():
                    results[path] = f"✓ {path}"
                else:
                    self.errors.append(f"Required file missing: {path}")
                    results[path] = f"✗ {path} missing"
            else:  # directory
                if full_path.is_dir():
                    results[path] = f"✓ {path}"
                else:
                    self.errors.append(f"Required directory missing: {path}")
                    results[path] = f"✗ {path} missing"
        
        self.validation_results['file_structure'] = results
        return all('✓' in result for result in results.values())
    
    def validate_configuration_files(self) -> bool:
        """Validate configuration files syntax and content."""
        logger.info("Validating configuration files...")
        
        results = {}
        
        # Validate docker-compose.yml
        docker_compose_file = self.project_root / 'docker-compose.yml'
        try:
            with open(docker_compose_file, 'r') as f:
                docker_config = yaml.safe_load(f)
            
            required_services = ['database', 'redis', 'trading_app', 'nginx', 'prometheus', 'grafana']
            missing_services = [svc for svc in required_services if svc not in docker_config.get('services', {})]
            
            if missing_services:
                self.errors.append(f"Missing services in docker-compose.yml: {', '.join(missing_services)}")
                results['docker_compose'] = f"✗ Missing services: {', '.join(missing_services)}"
            else:
                results['docker_compose'] = "✓ Valid docker-compose.yml"
        except Exception as e:
            self.errors.append(f"Invalid docker-compose.yml: {e}")
            results['docker_compose'] = f"✗ Invalid syntax: {e}"
        
        # Validate Prometheus configuration
        prometheus_config_file = self.project_root / 'monitoring/prometheus/prometheus.yml'
        try:
            with open(prometheus_config_file, 'r') as f:
                prometheus_config = yaml.safe_load(f)
            
            if 'scrape_configs' in prometheus_config:
                results['prometheus_config'] = "✓ Valid prometheus.yml"
            else:
                self.warnings.append("Prometheus config missing scrape_configs")
                results['prometheus_config'] = "⚠ Missing scrape_configs"
        except Exception as e:
            self.errors.append(f"Invalid prometheus.yml: {e}")
            results['prometheus_config'] = f"✗ Invalid: {e}"
        
        # Validate Nginx configuration
        nginx_config_file = self.project_root / 'nginx/nginx.conf'
        try:
            with open(nginx_config_file, 'r') as f:
                nginx_config = f.read()
            
            if 'upstream' in nginx_config and 'server' in nginx_config:
                results['nginx_config'] = "✓ Valid nginx.conf"
            else:
                self.warnings.append("Nginx config may be incomplete")
                results['nginx_config'] = "⚠ Incomplete configuration"
        except Exception as e:
            self.errors.append(f"Invalid nginx.conf: {e}")
            results['nginx_config'] = f"✗ Invalid: {e}"
        
        self.validation_results['configuration_files'] = results
        return len([r for r in results.values() if '✗' in r]) == 0
    
    def validate_python_modules(self) -> bool:
        """Validate Python modules can be imported and have basic functionality."""
        logger.info("Validating Python modules...")
        
        results = {}
        
        # Test critical modules
        critical_modules = {
            'src.api.api_server': 'API Server',
            'src.dashboard.trading_dashboard': 'Dashboard',
            'src.trading.paper_trading': 'Paper Trading',
            'src.utils.logging_system': 'Logging System',
            'src.brokers.interactive_brokers.ib_broker': 'IB Broker',
            'tests.test_suite': 'Test Suite'        }
        
        for module_path, module_name in critical_modules.items():
            try:
                # Add project root to Python path
                sys.path.insert(0, str(self.project_root))
                
                # Convert module path to file path
                file_path = self.project_root / module_path.replace('.', os.sep) 
                file_path = file_path.with_suffix('.py')
                
                spec = importlib.util.spec_from_file_location(
                    module_path.replace('.', '_'),
                    str(file_path)
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    results[module_name] = f"✓ {module_name}"
                else:
                    self.errors.append(f"Cannot load module: {module_path}")
                    results[module_name] = f"✗ {module_name} - Cannot load"
            except Exception as e:
                self.errors.append(f"Error importing {module_path}: {e}")
                results[module_name] = f"✗ {module_name} - {str(e)[:50]}..."
        
        self.validation_results['python_modules'] = results
        return all('✓' in result for result in results.values())
    
    def validate_database_setup(self) -> bool:
        """Validate database initialization script."""
        logger.info("Validating database setup...")
        
        results = {}
        
        # Check database init script
        init_script = self.project_root / 'scripts/init_db.sql'
        if not init_script.exists():
            self.errors.append("Database initialization script not found")
            results['init_script'] = "✗ init_db.sql missing"
            self.validation_results['database_setup'] = results
            return False
        
        try:
            with open(init_script, 'r') as f:
                script_content = f.read()
            
            # Check for required schema elements
            required_elements = [
                'CREATE SCHEMA',
                'CREATE TABLE',
                'CREATE INDEX',
                'users',
                'accounts',
                'orders',
                'positions',
                'securities'
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in script_content:
                    missing_elements.append(element)
            
            if missing_elements:
                self.warnings.append(f"Database script may be missing: {', '.join(missing_elements)}")
                results['init_script'] = f"⚠ May be missing: {', '.join(missing_elements)}"
            else:
                results['init_script'] = "✓ Complete database schema"
                
        except Exception as e:
            self.errors.append(f"Error reading database script: {e}")
            results['init_script'] = f"✗ Error: {e}"
        
        self.validation_results['database_setup'] = results
        return '✗' not in results.get('init_script', '')
    
    def validate_test_suite(self) -> bool:
        """Run the test suite to validate functionality."""
        logger.info("Running test suite validation...")
        
        results = {}
        
        try:
            # Run pytest with basic configuration
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', '-v', '--tb=short', '--maxfail=5'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)
            
            if result.returncode == 0:
                results['test_execution'] = "✓ All tests passed"
            else:
                test_output = result.stdout + result.stderr
                failed_tests = test_output.count('FAILED')
                self.warnings.append(f"{failed_tests} tests failed")
                results['test_execution'] = f"⚠ {failed_tests} tests failed"
                
        except subprocess.TimeoutExpired:
            self.warnings.append("Test suite execution timed out")
            results['test_execution'] = "⚠ Tests timed out"
        except Exception as e:
            self.errors.append(f"Error running tests: {e}")
            results['test_execution'] = f"✗ Error: {e}"
        
        self.validation_results['test_suite'] = results
        return '✗' not in str(results)
    
    def validate_docker_build(self) -> bool:
        """Validate Docker images can be built successfully."""
        logger.info("Validating Docker build process...")
        
        results = {}
        
        # Test main application Docker build
        try:
            result = subprocess.run([
                'docker', 'build', '-t', 'trading-platform-test:latest', '.'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=600)
            
            if result.returncode == 0:
                results['main_app_build'] = "✓ Main application builds successfully"
            else:
                self.errors.append(f"Main application Docker build failed: {result.stderr}")
                results['main_app_build'] = "✗ Build failed"
                
        except subprocess.TimeoutExpired:
            self.errors.append("Docker build timed out")
            results['main_app_build'] = "✗ Build timed out"
        except Exception as e:
            self.errors.append(f"Docker build error: {e}")
            results['main_app_build'] = f"✗ Error: {e}"
        
        # Test dashboard Docker build
        try:
            result = subprocess.run([
                'docker', 'build', '-f', 'Dockerfile.dashboard', 
                '-t', 'trading-dashboard-test:latest', '.'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=600)
            
            if result.returncode == 0:
                results['dashboard_build'] = "✓ Dashboard builds successfully"
            else:
                self.errors.append(f"Dashboard Docker build failed: {result.stderr}")
                results['dashboard_build'] = "✗ Build failed"
                
        except subprocess.TimeoutExpired:
            self.errors.append("Dashboard Docker build timed out")
            results['dashboard_build'] = "✗ Build timed out"
        except Exception as e:
            self.errors.append(f"Dashboard Docker build error: {e}")
            results['dashboard_build'] = f"✗ Error: {e}"
        
        self.validation_results['docker_build'] = results
        return all('✓' in result for result in results.values())
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report = []
        report.append("=" * 80)
        report.append("QUANTITATIVE TRADING PLATFORM VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Project root: {self.project_root}")
        report.append("")
        
        # Summary
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        if total_errors == 0 and total_warnings == 0:
            report.append("🎉 VALIDATION PASSED - Platform is ready for deployment!")
        elif total_errors == 0:
            report.append(f"⚠️  VALIDATION PASSED WITH WARNINGS - {total_warnings} warnings found")
        else:
            report.append(f"❌ VALIDATION FAILED - {total_errors} errors, {total_warnings} warnings")
        
        report.append("")
        
        # Detailed results
        for category, results in self.validation_results.items():
            report.append(f"📋 {category.upper().replace('_', ' ')}")
            report.append("-" * 40)
            
            if isinstance(results, dict):
                for key, value in results.items():
                    report.append(f"  {key}: {value}")
            else:
                report.append(f"  {results}")
            report.append("")
        
        # Errors
        if self.errors:
            report.append("❌ ERRORS")
            report.append("-" * 40)
            for error in self.errors:
                report.append(f"  • {error}")
            report.append("")
        
        # Warnings
        if self.warnings:
            report.append("⚠️  WARNINGS")
            report.append("-" * 40)
            for warning in self.warnings:
                report.append(f"  • {warning}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        if total_errors > 0:
            report.append("  • Fix all errors before deploying to production")
        if total_warnings > 0:
            report.append("  • Review and address warnings for optimal performance")
        report.append("  • Run integration tests after deployment")
        report.append("  • Monitor system performance during initial operation")
        report.append("  • Set up automated backups and monitoring alerts")
        report.append("  • Review security configurations for production use")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_validation(self) -> bool:
        """Run complete platform validation."""
        logger.info("Starting comprehensive platform validation...")
        
        validation_steps = [
            ("Environment", self.validate_environment),
            ("Python Dependencies", self.validate_python_dependencies),
            ("File Structure", self.validate_file_structure),
            ("Configuration Files", self.validate_configuration_files),
            ("Python Modules", self.validate_python_modules),
            ("Database Setup", self.validate_database_setup),
            ("Test Suite", self.validate_test_suite),
            ("Docker Build", self.validate_docker_build)
        ]
        
        overall_success = True
        
        for step_name, step_function in validation_steps:
            logger.info(f"Validating: {step_name}")
            try:
                step_result = step_function()
                if not step_result:
                    overall_success = False
            except Exception as e:
                logger.error(f"Validation step '{step_name}' failed with exception: {e}")
                self.errors.append(f"{step_name} validation failed: {e}")
                overall_success = False
        
        # Generate and save report
        report = self.generate_validation_report()
          # Save report to file
        report_file = self.project_root / 'validation_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print report
        print(report)
        
        logger.info(f"Validation completed. Report saved to: {report_file}")
        
        return overall_success

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate quantitative trading platform")
    parser.add_argument('--skip-docker', action='store_true',
                       help='Skip Docker build validation')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip test suite execution')
    
    args = parser.parse_args()
    
    validator = PlatformValidator()
    
    # Temporarily skip certain validations if requested
    if args.skip_docker:
        validator.validate_docker_build = lambda: True
    if args.skip_tests:
        validator.validate_test_suite = lambda: True
    
    success = validator.run_validation()
    
    if success:
        logger.info("✅ Platform validation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Platform validation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()