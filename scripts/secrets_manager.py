#!/usr/bin/env python3
"""
Secrets Management and Environment Setup Script

This script generates secure secrets, manages environment configurations,
and sets up the platform for production or development deployment.
"""

import os
import secrets
import string
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import base64
from datetime import datetime
from cryptography.fernet import Fernet


class SecretsManager:
    """Manage secrets generation and environment configuration."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.secrets_file = self.project_root / 'secrets' / 'secrets.yaml'
        self.secrets_dir = self.project_root / 'secrets'
        
        # Ensure secrets directory exists
        self.secrets_dir.mkdir(exist_ok=True)
        
        # Create .gitignore for secrets if it doesn't exist
        gitignore_file = self.secrets_dir / '.gitignore'
        if not gitignore_file.exists():
            gitignore_file.write_text('*\n!.gitignore\n')
    
    def generate_random_string(self, length: int = 32) -> str:
        """Generate a cryptographically secure random string."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_encryption_key(self) -> str:
        """Generate a Fernet-compatible encryption key."""
        return Fernet.generate_key().decode()
    
    def generate_jwt_secret(self, length: int = 64) -> str:
        """Generate a JWT secret key."""
        alphabet = string.ascii_letters + string.digits + '!@#$%^&*'
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_database_password(self, length: int = 24) -> str:
        """Generate a secure database password."""
        alphabet = string.ascii_letters + string.digits + '!@#$%^&*'
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_all_secrets(self) -> Dict[str, Any]:
        """Generate all required secrets for the platform."""
        secrets_data = {
            'api': {
                'secret_key': self.generate_random_string(64),
                'jwt_secret_key': self.generate_jwt_secret(),
                'encryption_key': self.generate_encryption_key()
            },
            'database': {
                'postgres_password': self.generate_database_password(),
                'root_password': self.generate_database_password()
            },
            'monitoring': {
                'grafana_admin_password': self.generate_random_string(16),
                'prometheus_web_password': self.generate_random_string(16)
            },
            'redis': {
                'password': self.generate_database_password()
            },
            'backup': {
                'encryption_key': self.generate_encryption_key()
            },
            'generated_at': str(datetime.now()),
            'version': '1.0.0'
        }
        
        return secrets_data
    
    def save_secrets(self, secrets_data: Dict[str, Any]):
        """Save secrets to encrypted file."""
        # Save as YAML
        with open(self.secrets_file, 'w') as f:
            yaml.dump(secrets_data, f, default_flow_style=False)
        
        # Set restrictive permissions
        if os.name != 'nt':  # Unix/Linux
            os.chmod(self.secrets_file, 0o600)
        
        print(f"✓ Secrets saved to: {self.secrets_file}")
    
    def load_secrets(self) -> Dict[str, Any]:
        """Load secrets from file."""
        if not self.secrets_file.exists():
            raise FileNotFoundError(f"Secrets file not found: {self.secrets_file}")
        
        with open(self.secrets_file, 'r') as f:
            return yaml.safe_load(f)
    
    def create_environment_file(self, environment: str = 'development', 
                               custom_config: Dict[str, Any] = None):
        """Create environment file with secrets."""
        secrets_data = self.load_secrets() if self.secrets_file.exists() else self.generate_all_secrets()
        
        if not self.secrets_file.exists():
            self.save_secrets(secrets_data)
        
        env_file = self.project_root / f'.env.{environment}'
        
        # Load existing environment file if it exists
        if env_file.exists():
            print(f"Environment file already exists: {env_file}")
            return
        
        # Environment-specific configurations
        env_configs = {
            'development': {
                'ENVIRONMENT': 'development',
                'DEBUG': 'true',
                'LOG_LEVEL': 'DEBUG',
                'DATABASE_HOST': 'localhost',
                'DATABASE_NAME': 'trading_db_dev',
                'API_WORKERS': '1',
                'PAPER_TRADING': 'true',
                'SSL_ENABLED': 'false'
            },
            'production': {
                'ENVIRONMENT': 'production',
                'DEBUG': 'false',
                'LOG_LEVEL': 'INFO',
                'DATABASE_HOST': 'database',
                'DATABASE_NAME': 'trading_db',
                'API_WORKERS': '4',
                'PAPER_TRADING': 'false',
                'SSL_ENABLED': 'true'
            }
        }
        
        base_config = env_configs.get(environment, env_configs['development'])
        
        # Merge with custom config
        if custom_config:
            base_config.update(custom_config)
        
        # Add secrets to configuration
        config = {
            **base_config,
            'API_SECRET_KEY': secrets_data['api']['secret_key'],
            'JWT_SECRET_KEY': secrets_data['api']['jwt_secret_key'],
            'ENCRYPTION_KEY': secrets_data['api']['encryption_key'],
            'DATABASE_PASSWORD': secrets_data['database']['postgres_password'],
            'REDIS_PASSWORD': secrets_data['redis']['password'],
            'GRAFANA_ADMIN_PASSWORD': secrets_data['monitoring']['grafana_admin_password']
        }
        
        # Write environment file
        with open(env_file, 'w') as f:
            f.write(f"# {environment.title()} Environment Configuration\n")
            f.write(f"# Generated automatically - DO NOT EDIT MANUALLY\n")
            f.write(f"# Generated at: {datetime.now()}\n\n")
            
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        
        print(f"✓ Environment file created: {env_file}")
    
    def setup_ssl_certificates(self):
        """Generate self-signed SSL certificates for development."""
        ssl_dir = self.project_root / 'ssl'
        ssl_dir.mkdir(exist_ok=True)
        
        cert_file = ssl_dir / 'trading-platform.crt'
        key_file = ssl_dir / 'trading-platform.key'
        
        if cert_file.exists() and key_file.exists():
            print(f"SSL certificates already exist in: {ssl_dir}")
            return
        
        try:
            import subprocess
            
            # Generate private key
            subprocess.run([
                'openssl', 'genrsa', '-out', str(key_file), '2048'
            ], check=True, capture_output=True)
            
            # Generate certificate
            subprocess.run([
                'openssl', 'req', '-new', '-x509', '-key', str(key_file),
                '-out', str(cert_file), '-days', '365', '-subj',
                '/C=US/ST=State/L=City/O=Organization/CN=localhost'
            ], check=True, capture_output=True)
            
            print(f"✓ SSL certificates generated in: {ssl_dir}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ OpenSSL not found. Please install OpenSSL to generate SSL certificates.")
            print("  For development, you can disable SSL in the environment configuration.")
    
    def create_docker_secrets(self):
        """Create Docker secrets files."""
        secrets_data = self.load_secrets()
        docker_secrets_dir = self.project_root / 'docker' / 'secrets'
        docker_secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual secret files for Docker Swarm
        secret_files = {
            'postgres_password': secrets_data['database']['postgres_password'],
            'redis_password': secrets_data['redis']['password'],
            'jwt_secret': secrets_data['api']['jwt_secret_key'],
            'api_secret': secrets_data['api']['secret_key'],
            'grafana_admin_password': secrets_data['monitoring']['grafana_admin_password']
        }
        
        for secret_name, secret_value in secret_files.items():
            secret_file = docker_secrets_dir / f"{secret_name}.txt"
            with open(secret_file, 'w') as f:
                f.write(secret_value)
            
            # Set restrictive permissions
            if os.name != 'nt':  # Unix/Linux
                os.chmod(secret_file, 0o600)
        
        print(f"✓ Docker secrets created in: {docker_secrets_dir}")
    
    def validate_environment(self, environment: str = 'development'):
        """Validate environment configuration."""
        env_file = self.project_root / f'.env.{environment}'
        
        if not env_file.exists():
            print(f"❌ Environment file not found: {env_file}")
            return False
        
        required_vars = [
            'DATABASE_PASSWORD', 'API_SECRET_KEY', 'JWT_SECRET_KEY',
            'ENCRYPTION_KEY', 'ENVIRONMENT'
        ]
        
        env_vars = {}
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
        
        missing_vars = [var for var in required_vars if var not in env_vars]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        # Validate secret strength
        weak_secrets = []
        if len(env_vars.get('DATABASE_PASSWORD', '')) < 16:
            weak_secrets.append('DATABASE_PASSWORD')
        if len(env_vars.get('API_SECRET_KEY', '')) < 32:
            weak_secrets.append('API_SECRET_KEY')
        if len(env_vars.get('JWT_SECRET_KEY', '')) < 32:
            weak_secrets.append('JWT_SECRET_KEY')
        
        if weak_secrets:
            print(f"⚠ Weak secrets detected: {weak_secrets}")
        
        print(f"✓ Environment configuration validated: {env_file}")
        return True


def main():
    """Main function for secrets management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Secrets Management for Trading Platform')
    parser.add_argument('command', choices=['generate', 'create-env', 'validate', 'setup-ssl', 'docker-secrets'],
                       help='Command to execute')
    parser.add_argument('--environment', choices=['development', 'production'], default='development',
                       help='Environment type')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing files')
    
    args = parser.parse_args()
    
    secrets_manager = SecretsManager()
    
    if args.command == 'generate':
        print("🔐 Generating secrets...")
        secrets_data = secrets_manager.generate_all_secrets()
        secrets_manager.save_secrets(secrets_data)
        print("✓ Secrets generated successfully!")
    
    elif args.command == 'create-env':
        print(f"📝 Creating {args.environment} environment file...")
        secrets_manager.create_environment_file(args.environment)
        print(f"✓ Environment file created for {args.environment}!")
    
    elif args.command == 'validate':
        print(f"🔍 Validating {args.environment} environment...")
        if secrets_manager.validate_environment(args.environment):
            print("✓ Environment validation successful!")
        else:
            print("❌ Environment validation failed!")
            return 1
    
    elif args.command == 'setup-ssl':
        print("🔒 Setting up SSL certificates...")
        secrets_manager.setup_ssl_certificates()
        print("✓ SSL setup completed!")
    
    elif args.command == 'docker-secrets':
        print("🐳 Creating Docker secrets...")
        secrets_manager.create_docker_secrets()
        print("✓ Docker secrets created!")
    
    return 0


if __name__ == '__main__':
    exit(main())
