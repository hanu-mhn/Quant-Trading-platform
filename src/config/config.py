"""
Configuration management for the trading platform.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    external_data_path: str = "data/external"
    default_timeframe: str = "1D"
    max_cache_size_mb: int = 1000
    cache_expiry_hours: int = 24


@dataclass
class BacktestConfig:
    """Backtesting configuration settings."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    benchmark_symbol: str = "NIFTY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    rebalance_frequency: str = "monthly"


@dataclass
class LiveTradingConfig:
    """Live trading configuration settings."""
    paper_trading: bool = True
    max_orders_per_day: int = 100
    max_position_size_percent: float = 10.0
    risk_limit_percent: float = 2.0
    auto_cancel_hours: int = 24
    heartbeat_interval_seconds: int = 30


@dataclass
class BrokerConfig:
    """Broker configuration settings."""
    default_broker: str = "zerodha"
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    base_url: str = ""
    timeout_seconds: int = 30
    rate_limit_per_second: int = 10


@dataclass
class RiskConfig:
    """Risk management configuration settings."""
    max_portfolio_risk_percent: float = 5.0
    max_single_position_percent: float = 10.0
    max_sector_exposure_percent: float = 25.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 6.0
    var_confidence_level: float = 0.95
    max_drawdown_percent: float = 10.0


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    log_level: str = "INFO"
    log_file_path: str = "logs/trading_platform.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_logging: bool = True


@dataclass
class NotificationConfig:
    """Notification configuration settings."""
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_to: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


class Config:
    """Main configuration class that manages all settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path or "src/config/config.yaml"
        
        # Initialize with default configurations
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.live_trading = LiveTradingConfig()
        self.broker = BrokerConfig()
        self.risk = RiskConfig()
        self.logging = LoggingConfig()
        self.notification = NotificationConfig()
        
        # Load configuration from file if it exists
        self.load_config()
        
        # Override with environment variables
        self._load_from_env()
    
    def load_config(self):
        """Load configuration from file."""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self._update_from_dict(config_data)
                print(f"Configuration loaded from {config_file}")
            except Exception as e:
                print(f"Error loading configuration from {config_file}: {e}")
                print("Using default configuration")
        else:
            print(f"Configuration file {config_file} not found. Using default configuration.")
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = file_path or self.config_path
        config_file = Path(save_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = self.to_dict()
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, default=str)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Error saving configuration to {config_file}: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if 'data' in config_data:
            self.data = DataConfig(**config_data['data'])
        if 'backtest' in config_data:
            self.backtest = BacktestConfig(**config_data['backtest'])
        if 'live_trading' in config_data:
            self.live_trading = LiveTradingConfig(**config_data['live_trading'])
        if 'broker' in config_data:
            self.broker = BrokerConfig(**config_data['broker'])
        if 'risk' in config_data:
            self.risk = RiskConfig(**config_data['risk'])
        if 'logging' in config_data:
            self.logging = LoggingConfig(**config_data['logging'])
        if 'notification' in config_data:
            self.notification = NotificationConfig(**config_data['notification'])
    
    def _load_from_env(self):
        """Load sensitive settings from environment variables."""
        # Broker settings
        if os.getenv('BROKER_API_KEY'):
            self.broker.api_key = os.getenv('BROKER_API_KEY')
        if os.getenv('BROKER_API_SECRET'):
            self.broker.api_secret = os.getenv('BROKER_API_SECRET')
        if os.getenv('BROKER_ACCESS_TOKEN'):
            self.broker.access_token = os.getenv('BROKER_ACCESS_TOKEN')
        
        # Notification settings
        if os.getenv('EMAIL_USERNAME'):
            self.notification.email_username = os.getenv('EMAIL_USERNAME')
        if os.getenv('EMAIL_PASSWORD'):
            self.notification.email_password = os.getenv('EMAIL_PASSWORD')
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self.notification.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if os.getenv('TELEGRAM_CHAT_ID'):
            self.notification.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'backtest': asdict(self.backtest),
            'live_trading': asdict(self.live_trading),
            'broker': asdict(self.broker),
            'risk': asdict(self.risk),
            'logging': asdict(self.logging),
            'notification': asdict(self.notification),
            'last_updated': datetime.now().isoformat()
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate data paths
        if not self.data.raw_data_path:
            errors.append("Raw data path is required")
        
        # Validate backtesting settings
        if self.backtest.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        if not (0 <= self.backtest.commission_rate <= 1):
            errors.append("Commission rate must be between 0 and 1")
        
        # Validate live trading settings
        if not (0 < self.live_trading.max_position_size_percent <= 100):
            errors.append("Max position size percent must be between 0 and 100")
        
        # Validate risk settings
        if not (0 < self.risk.max_portfolio_risk_percent <= 100):
            errors.append("Max portfolio risk percent must be between 0 and 100")
        
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_data_path(self, data_type: str = "raw") -> Path:
        """Get path for specific data type."""
        base_path = Path(".")
        if data_type == "raw":
            return base_path / self.data.raw_data_path
        elif data_type == "processed":
            return base_path / self.data.processed_data_path
        elif data_type == "external":
            return base_path / self.data.external_data_path
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def get_log_path(self) -> Path:
        """Get logging file path."""
        return Path(self.logging.log_file_path)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from file."""
    global config
    config.load_config()


def save_config(file_path: Optional[str] = None):
    """Save current configuration to file."""
    config.save_config(file_path)
