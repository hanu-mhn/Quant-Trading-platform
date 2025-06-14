#!/usr/bin/env python3
"""
Backup and Restore Script for Quantitative Trading Platform

This script provides comprehensive backup and restore functionality for:
- Database data
- Configuration files
- Trading logs
- Model files
- System state
"""

import os
import sys
import shutil
import tarfile
import gzip
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import logging
import argparse


class BackupManager:
    """Comprehensive backup and restore manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.backup_dir = self.project_root / 'backups'
        self.config_file = config_file or self.project_root / 'config' / 'backup_config.yaml'
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
    def setup_logging(self):
        """Setup logging for backup operations."""
        log_file = self.backup_dir / 'backup.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict[str, Any]:
        """Load backup configuration."""
        default_config = {
            'retention': {
                'daily_backups': 7,
                'weekly_backups': 4,
                'monthly_backups': 12
            },
            'compression': True,
            'encryption': False,
            'include_paths': [
                'data/processed',
                'data/external',
                'logs',
                'src/config',
                'secrets',
                'models'
            ],
            'exclude_patterns': [
                '*.pyc',
                '__pycache__',
                '*.log',
                'temp',
                '.git'
            ],
            'database': {
                'enabled': True,
                'format': 'sql'  # sql or binary
            },
            'remote_storage': {
                'enabled': False,
                'type': 's3',  # s3, ftp, sftp
                'bucket': 'trading-platform-backups',
                'prefix': 'backups/'
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
    
    def create_backup(self, backup_type: str = 'manual', 
                     include_database: bool = True,
                     include_files: bool = True) -> Path:
        """
        Create a comprehensive backup.
        
        Args:
            backup_type: Type of backup (manual, daily, weekly, monthly)
            include_database: Whether to include database backup
            include_files: Whether to include file system backup
            
        Returns:
            Path to the created backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{backup_type}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Starting {backup_type} backup: {backup_name}")
        
        try:
            # Create metadata
            metadata = self.create_metadata(backup_type)
            
            # Backup database
            if include_database and self.config['database']['enabled']:
                db_backup_file = self.backup_database(backup_path)
                metadata['database_backup'] = str(db_backup_file.name)
            
            # Backup files
            if include_files:
                files_backup = self.backup_files(backup_path)
                metadata['files_backup'] = str(files_backup.name)
            
            # Save metadata
            metadata_file = backup_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Create compressed archive
            if self.config['compression']:
                archive_path = self.create_archive(backup_path)
                shutil.rmtree(backup_path)
                backup_path = archive_path
            
            # Upload to remote storage
            if self.config['remote_storage']['enabled']:
                self.upload_to_remote(backup_path)
            
            self.logger.info(f"Backup completed successfully: {backup_path}")
            
            # Cleanup old backups
            self.cleanup_old_backups(backup_type)
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def create_metadata(self, backup_type: str) -> Dict[str, Any]:
        """Create backup metadata."""
        return {
            'backup_type': backup_type,
            'timestamp': datetime.now().isoformat(),
            'platform_version': '1.0.0',
            'python_version': sys.version,
            'host': os.uname().nodename if hasattr(os, 'uname') else 'windows',
            'size_bytes': 0,
            'files_count': 0,
            'compression': self.config['compression'],
            'encryption': self.config['encryption']
        }
    
    def backup_database(self, backup_path: Path) -> Path:
        """Backup database using pg_dump."""
        self.logger.info("Backing up database...")
        
        # Get database configuration from environment
        db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5432'),
            'database': os.getenv('DATABASE_NAME', 'trading_db'),
            'username': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', '')
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.config['database']['format'] == 'sql':
            db_backup_file = backup_path / f"database_{timestamp}.sql"
            
            # Create pg_dump command
            cmd = [
                'pg_dump',
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['username']}",
                '--no-password',
                '--verbose',
                '--clean',
                '--no-acl',
                '--no-owner',
                db_config['database']
            ]
            
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = db_config['password']
            
            try:
                with open(db_backup_file, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          env=env, check=True, text=True)
                
                self.logger.info(f"Database backup created: {db_backup_file}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Database backup failed: {e.stderr}")
                raise
            except FileNotFoundError:
                self.logger.warning("pg_dump not found. Skipping database backup.")
                # Create a placeholder file
                db_backup_file.write_text("# Database backup skipped - pg_dump not available")
        
        else:  # binary format
            db_backup_file = backup_path / f"database_{timestamp}.dump"
            
            cmd = [
                'pg_dump',
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['username']}",
                '--no-password',
                '--verbose',
                '--format=custom',
                '--compress=9',
                f"--file={db_backup_file}",
                db_config['database']
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = db_config['password']
            
            try:
                subprocess.run(cmd, env=env, check=True)
                self.logger.info(f"Database backup created: {db_backup_file}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Database backup failed: {e}")
                raise
            except FileNotFoundError:
                self.logger.warning("pg_dump not found. Skipping database backup.")
                db_backup_file.write_text("Database backup skipped - pg_dump not available")
        
        return db_backup_file
    
    def backup_files(self, backup_path: Path) -> Path:
        """Backup file system data."""
        self.logger.info("Backing up files...")
        
        files_backup_dir = backup_path / 'files'
        files_backup_dir.mkdir(exist_ok=True)
        
        files_count = 0
        total_size = 0
        
        for include_path in self.config['include_paths']:
            source_path = self.project_root / include_path
            
            if not source_path.exists():
                self.logger.warning(f"Path not found: {source_path}")
                continue
            
            dest_path = files_backup_dir / include_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_file():
                shutil.copy2(source_path, dest_path)
                files_count += 1
                total_size += source_path.stat().st_size
            else:
                shutil.copytree(source_path, dest_path, 
                              ignore=shutil.ignore_patterns(*self.config['exclude_patterns']))
                
                # Count files and calculate size
                for file_path in dest_path.rglob('*'):
                    if file_path.is_file():
                        files_count += 1
                        total_size += file_path.stat().st_size
        
        self.logger.info(f"Backed up {files_count} files ({total_size / 1024 / 1024:.2f} MB)")
        
        return files_backup_dir
    
    def create_archive(self, backup_path: Path) -> Path:
        """Create compressed archive of backup."""
        self.logger.info("Creating compressed archive...")
        
        archive_path = backup_path.with_suffix('.tar.gz')
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        return archive_path
    
    def cleanup_old_backups(self, backup_type: str):
        """Remove old backups based on retention policy."""
        self.logger.info(f"Cleaning up old {backup_type} backups...")
        
        retention_days = {
            'daily': self.config['retention']['daily_backups'],
            'weekly': self.config['retention']['weekly_backups'] * 7,
            'monthly': self.config['retention']['monthly_backups'] * 30,
            'manual': 30  # Keep manual backups for 30 days
        }
        
        cutoff_date = datetime.now() - timedelta(days=retention_days.get(backup_type, 30))
        
        pattern = f"backup_{backup_type}_*"
        old_backups = list(self.backup_dir.glob(pattern))
        
        removed_count = 0
        for backup_file in old_backups:
            try:
                # Extract timestamp from filename
                # Format: backup_type_YYYYMMDD_HHMMSS.tar.gz
                parts = backup_file.name.split('_')
                if len(parts) >= 4:
                    date_part = parts[2]
                    time_part = parts[3]
                    
                    # Remove extension from time part
                    if '.' in time_part:
                        time_part = time_part.split('.')[0]
                    
                    timestamp_str = f"{date_part}_{time_part}"
                else:
                    # Fallback for old format
                    timestamp_str = parts[-1]
                    if '.' in timestamp_str:
                        timestamp_str = timestamp_str.split('.')[0]
                
                backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                if backup_date < cutoff_date:
                    if backup_file.is_file():
                        backup_file.unlink()
                    else:
                        shutil.rmtree(backup_file)
                    
                    removed_count += 1
                    self.logger.info(f"Removed old backup: {backup_file.name}")
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not parse backup date for {backup_file.name}: {e}")
        
        self.logger.info(f"Removed {removed_count} old backups")
    
    def restore_backup(self, backup_path: Path, 
                      restore_database: bool = True,
                      restore_files: bool = True):
        """
        Restore from backup.
        
        Args:
            backup_path: Path to backup file or directory
            restore_database: Whether to restore database
            restore_files: Whether to restore files
        """
        self.logger.info(f"Starting restore from: {backup_path}")
        
        # Extract archive if necessary
        work_dir = None
        if backup_path.suffix == '.gz':
            work_dir = self.backup_dir / f"restore_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            work_dir.mkdir(exist_ok=True)
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(work_dir)
            
            # Find the extracted backup directory
            extracted_dirs = [d for d in work_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                backup_path = extracted_dirs[0]
        
        try:
            # Load metadata
            metadata_file = backup_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Restoring backup from {metadata['timestamp']}")
            
            # Restore database
            if restore_database:
                db_backup_files = list(backup_path.glob('database_*'))
                if db_backup_files:
                    self.restore_database(db_backup_files[0])
            
            # Restore files
            if restore_files:
                files_backup_dir = backup_path / 'files'
                if files_backup_dir.exists():
                    self.restore_files(files_backup_dir)
            
            self.logger.info("Restore completed successfully")
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            raise
        
        finally:
            # Cleanup temporary directory
            if work_dir and work_dir.exists():
                shutil.rmtree(work_dir)
    
    def restore_database(self, db_backup_file: Path):
        """Restore database from backup."""
        self.logger.info(f"Restoring database from: {db_backup_file}")
        
        # Get database configuration
        db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': os.getenv('DATABASE_PORT', '5432'),
            'database': os.getenv('DATABASE_NAME', 'trading_db'),
            'username': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', '')
        }
        
        env = os.environ.copy()
        env['PGPASSWORD'] = db_config['password']
        
        if db_backup_file.suffix == '.sql':
            # Restore from SQL dump
            cmd = [
                'psql',
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['username']}",
                '--no-password',
                db_config['database']
            ]
            
            with open(db_backup_file, 'r') as f:
                subprocess.run(cmd, stdin=f, env=env, check=True)
        
        else:
            # Restore from binary dump
            cmd = [
                'pg_restore',
                f"--host={db_config['host']}",
                f"--port={db_config['port']}",
                f"--username={db_config['username']}",
                '--no-password',
                '--verbose',
                '--clean',
                '--no-acl',
                '--no-owner',
                f"--dbname={db_config['database']}",
                str(db_backup_file)
            ]
            
            subprocess.run(cmd, env=env, check=True)
        
        self.logger.info("Database restored successfully")
    
    def restore_files(self, files_backup_dir: Path):
        """Restore files from backup."""
        self.logger.info(f"Restoring files from: {files_backup_dir}")
        
        restored_count = 0
        
        for backup_item in files_backup_dir.rglob('*'):
            if backup_item.is_file():
                # Calculate relative path from backup root
                rel_path = backup_item.relative_to(files_backup_dir)
                dest_path = self.project_root / rel_path
                
                # Create destination directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(backup_item, dest_path)
                restored_count += 1
        
        self.logger.info(f"Restored {restored_count} files")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []
        
        for backup_item in self.backup_dir.iterdir():
            if backup_item.name.startswith('backup_'):
                try:
                    # Extract backup info from filename
                    # Format: backup_type_YYYYMMDD_HHMMSS.tar.gz
                    parts = backup_item.name.split('_')
                    if len(parts) >= 4:
                        backup_type = parts[1]
                        date_part = parts[2]
                        time_part = parts[3]
                        
                        # Remove extension from time part
                        if '.' in time_part:
                            time_part = time_part.split('.')[0]
                        
                        timestamp_str = f"{date_part}_{time_part}"
                    else:
                        # Fallback for old format
                        backup_type = parts[1]
                        timestamp_str = parts[2]
                        if '.' in timestamp_str:
                            timestamp_str = timestamp_str.split('.')[0]
                    
                    backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    backup_info = {
                        'name': backup_item.name,
                        'path': str(backup_item),
                        'type': backup_type,
                        'date': backup_date.isoformat(),
                        'size_mb': self.get_backup_size(backup_item)
                    }
                    
                    # Load metadata if available
                    if backup_item.is_dir():
                        metadata_file = backup_item / 'metadata.json'
                    else:
                        metadata_file = None
                    
                    if metadata_file and metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            backup_info.update(metadata)
                    
                    backups.append(backup_info)
                    
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Could not parse backup info for {backup_item.name}: {e}")
        
        # Sort by date (newest first)
        backups.sort(key=lambda x: x['date'], reverse=True)
        
        return backups
    
    def get_backup_size(self, backup_path: Path) -> float:
        """Get backup size in MB."""
        total_size = 0
        
        if backup_path.is_file():
            total_size = backup_path.stat().st_size
        else:
            for file_path in backup_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return total_size / 1024 / 1024  # Convert to MB


def main():
    """Main function for backup management."""
    parser = argparse.ArgumentParser(description='Backup and Restore for Trading Platform')
    parser.add_argument('command', choices=['backup', 'restore', 'list', 'cleanup'],
                       help='Command to execute')
    parser.add_argument('--type', choices=['manual', 'daily', 'weekly', 'monthly'], 
                       default='manual', help='Backup type')
    parser.add_argument('--backup-path', help='Path to backup file for restore')
    parser.add_argument('--no-database', action='store_true', help='Skip database backup/restore')
    parser.add_argument('--no-files', action='store_true', help='Skip files backup/restore')
    parser.add_argument('--config', help='Path to backup configuration file')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(Path(args.config) if args.config else None)
    
    try:
        if args.command == 'backup':
            backup_path = backup_manager.create_backup(
                backup_type=args.type,
                include_database=not args.no_database,
                include_files=not args.no_files
            )
            print(f"✓ Backup created: {backup_path}")
        
        elif args.command == 'restore':
            if not args.backup_path:
                print("Error: --backup-path is required for restore")
                return 1
            
            backup_path = Path(args.backup_path)
            if not backup_path.exists():
                print(f"Error: Backup path not found: {backup_path}")
                return 1
            
            backup_manager.restore_backup(
                backup_path,
                restore_database=not args.no_database,
                restore_files=not args.no_files
            )
            print("✓ Restore completed")
        
        elif args.command == 'list':
            backups = backup_manager.list_backups()
            if not backups:
                print("No backups found")
            else:
                print(f"{'Name':<30} {'Type':<10} {'Date':<20} {'Size (MB)':<10}")
                print("-" * 70)
                for backup in backups:
                    print(f"{backup['name']:<30} {backup['type']:<10} "
                          f"{backup['date'][:19]:<20} {backup['size_mb']:<10.2f}")
        
        elif args.command == 'cleanup':
            backup_manager.cleanup_old_backups(args.type)
            print(f"✓ Cleanup completed for {args.type} backups")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
