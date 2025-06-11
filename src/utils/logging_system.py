"""
Comprehensive Logging and Monitoring System

This module provides advanced logging capabilities, performance monitoring,
and system health checks for the quantitative trading platform.
"""

import logging
import logging.handlers
import os
import sys
import time
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import pandas as pd
from collections import defaultdict, deque
import warnings
import traceback
from contextlib import contextmanager

warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_threads: int
    open_files: int


@dataclass
class TradingMetrics:
    """Trading-specific metrics"""
    timestamp: datetime
    orders_placed: int
    orders_filled: int
    orders_cancelled: int
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    portfolio_value: float
    cash_balance: float
    position_count: int
    active_strategies: int
    data_feed_latency_ms: float
    order_execution_time_ms: float


class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors and enhanced information"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record):
        # Create the log format
        log_format = (
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
        )
        
        if self.use_colors and hasattr(record, 'levelname'):
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            log_format = f"{level_color}{log_format}{reset_color}"
        
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class DatabaseHandler(logging.Handler):
    """Custom logging handler that stores logs in SQLite database"""
    
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the logs database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                logger_name TEXT,
                module TEXT,
                function TEXT,
                line_number INTEGER,
                message TEXT,
                extra_data TEXT
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
            ON logs(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_logs_level 
            ON logs(level)
        ''')
        
        conn.commit()
        conn.close()
    
    def emit(self, record):
        """Emit a log record to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare log data
            extra_data = {}
            if hasattr(record, '__dict__'):
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                 'pathname', 'filename', 'module', 'exc_info',
                                 'exc_text', 'stack_info', 'lineno', 'funcName',
                                 'created', 'msecs', 'relativeCreated', 'thread',
                                 'threadName', 'processName', 'process', 'getMessage']:
                        extra_data[key] = str(value)
            
            cursor.execute('''
                INSERT INTO logs (timestamp, level, logger_name, module, function, 
                                line_number, message, extra_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.fromtimestamp(record.created).isoformat(),
                record.levelname,
                record.name,
                record.module,
                record.funcName,
                record.lineno,
                self.format(record),
                json.dumps(extra_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Don't let logging errors crash the application
            print(f"Database logging error: {e}")


class PerformanceMonitor:
    """System performance monitoring"""
    
    def __init__(self, collection_interval: int = 60):
        """
        Initialize performance monitor
        
        Args:
            collection_interval: Seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self._monitoring = False
        self._monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize baseline metrics
        self.process = psutil.Process()
        self.baseline_net_io = psutil.net_io_counters()
        self.baseline_disk_io = psutil.disk_io_counters()
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Process-specific metrics
        process_memory = self.process.memory_info()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read = current_disk_io.read_bytes - self.baseline_disk_io.read_bytes
        disk_write = current_disk_io.write_bytes - self.baseline_disk_io.write_bytes
        
        # Network I/O
        current_net_io = psutil.net_io_counters()
        net_sent = current_net_io.bytes_sent - self.baseline_net_io.bytes_sent
        net_recv = current_net_io.bytes_recv - self.baseline_net_io.bytes_recv
        
        # Threading and file handles
        active_threads = threading.active_count()
        try:
            open_files = len(self.process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=process_memory.rss / 1024 / 1024,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_sent=net_sent,
            network_recv=net_recv,
            active_threads=active_threads,
            open_files=open_files
        )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and log alerts"""
        # High CPU usage
        if metrics.cpu_percent > 80:
            self.logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # High memory usage
        if metrics.memory_percent > 80:
            self.logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # High process memory
        if metrics.memory_mb > 1024:  # 1GB
            self.logger.warning(f"High process memory usage: {metrics.memory_mb:.1f} MB")
        
        # Too many threads
        if metrics.active_threads > 50:
            self.logger.warning(f"High thread count: {metrics.active_threads}")
        
        # Too many open files
        if metrics.open_files > 100:
            self.logger.warning(f"High open file count: {metrics.open_files}")
    
    def get_recent_metrics(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """
        Get recent performance metrics
        
        Args:
            minutes: Number of recent minutes to retrieve
            
        Returns:
            List of performance metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'collection_interval': self.collection_interval,
            'total_samples': len(self.metrics_history),
            'recent_samples': len(recent_metrics),
            'cpu_usage': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            },
            'memory_usage': {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0
            },
            'current_threads': recent_metrics[-1].active_threads if recent_metrics else 0,
            'current_open_files': recent_metrics[-1].open_files if recent_metrics else 0
        }


class TradingMonitor:
    """Trading-specific monitoring and metrics"""
    
    def __init__(self):
        self.trading_metrics = deque(maxlen=1440)  # 24 hours of data
        self.alert_thresholds = {
            'max_daily_loss': -5000,  # $5000 loss
            'max_drawdown': -0.1,     # 10% drawdown
            'min_success_rate': 0.3,   # 30% success rate
            'max_open_positions': 50
        }
        self.logger = logging.getLogger(__name__)
    
    def record_metrics(self, metrics: TradingMetrics):
        """Record trading metrics"""
        self.trading_metrics.append(metrics)
        self._check_trading_alerts(metrics)
    
    def _check_trading_alerts(self, metrics: TradingMetrics):
        """Check for trading-related alerts"""
        # Daily loss check
        if metrics.realized_pnl < self.alert_thresholds['max_daily_loss']:
            self.logger.critical(f"Daily loss limit exceeded: ${metrics.realized_pnl:,.2f}")
        
        # Position count check
        if metrics.position_count > self.alert_thresholds['max_open_positions']:
            self.logger.warning(f"High position count: {metrics.position_count}")
        
        # Data feed latency check
        if metrics.data_feed_latency_ms > 1000:  # 1 second
            self.logger.warning(f"High data feed latency: {metrics.data_feed_latency_ms:.1f}ms")
        
        # Order execution time check
        if metrics.order_execution_time_ms > 5000:  # 5 seconds
            self.logger.warning(f"Slow order execution: {metrics.order_execution_time_ms:.1f}ms")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        if not self.trading_metrics:
            return {}
        
        recent = list(self.trading_metrics)[-60:]  # Last hour
        
        total_orders = sum(m.orders_placed for m in recent)
        total_fills = sum(m.orders_filled for m in recent)
        fill_rate = (total_fills / total_orders) if total_orders > 0 else 0
        
        return {
            'total_samples': len(self.trading_metrics),
            'recent_samples': len(recent),
            'orders_placed_1h': total_orders,
            'orders_filled_1h': total_fills,
            'fill_rate_1h': fill_rate,
            'current_portfolio_value': recent[-1].portfolio_value if recent else 0,
            'current_pnl': recent[-1].total_pnl if recent else 0,
            'current_positions': recent[-1].position_count if recent else 0,
            'avg_data_latency_ms': sum(m.data_feed_latency_ms for m in recent) / len(recent) if recent else 0,
            'avg_execution_time_ms': sum(m.order_execution_time_ms for m in recent) / len(recent) if recent else 0
        }


class LoggingManager:
    """Centralized logging management"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_database_logging: bool = True,
                 enable_console_colors: bool = True):
        """
        Initialize logging manager
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum log level
            max_file_size: Maximum size per log file
            backup_count: Number of backup files to keep
            enable_database_logging: Whether to enable database logging
            enable_console_colors: Whether to use colored console output
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_database_logging = enable_database_logging
        self.enable_console_colors = enable_console_colors
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize monitors
        self.performance_monitor = PerformanceMonitor()
        self.trading_monitor = TradingMonitor()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging system initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = CustomFormatter(use_colors=self.enable_console_colors)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "trading_platform.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler (separate file for errors)
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "errors.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # Database handler
        if self.enable_database_logging:
            db_handler = DatabaseHandler(str(self.log_dir / "logs.db"))
            db_handler.setLevel(self.log_level)
            root_logger.addHandler(db_handler)
        
        # Trading-specific handler
        trading_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "trading.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(file_formatter)
        
        # Add trading handler to specific loggers
        trading_loggers = [
            'src.core.backtester',
            'src.core.live_trader',
            'src.brokers',
            'src.strategies',
            'src.portfolio'
        ]
        
        for logger_name in trading_loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(trading_handler)
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.performance_monitor.start_monitoring()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.performance_monitor.stop_monitoring()
        self.logger.info("System monitoring stopped")
    
    def record_trading_metrics(self, metrics: TradingMetrics):
        """Record trading metrics"""
        self.trading_monitor.record_metrics(metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance': self.performance_monitor.get_performance_summary(),
            'trading': self.trading_monitor.get_trading_summary(),
            'logging': {
                'log_level': logging.getLevelName(self.log_level),
                'log_dir': str(self.log_dir),
                'handlers_count': len(logging.getLogger().handlers)
            }
        }
    
    def query_logs(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[str] = None,
                   logger_name: Optional[str] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        Query logs from database
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            level: Log level filter
            logger_name: Logger name filter
            limit: Maximum number of records
            
        Returns:
            DataFrame of log records
        """
        if not self.enable_database_logging:
            return pd.DataFrame()
        
        db_path = self.log_dir / "logs.db"
        if not db_path.exists():
            return pd.DataFrame()
        
        conn = sqlite3.connect(str(db_path))
        
        query = "SELECT * FROM logs WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if level:
            query += " AND level = ?"
            params.append(level)
        
        if logger_name:
            query += " AND logger_name LIKE ?"
            params.append(f"%{logger_name}%")
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        self.logger.info(f"Starting operation: {operation_name}")
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(f"Completed operation: {operation_name} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed operation: {operation_name} after {duration:.3f}s - {e}")
            raise
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with full traceback"""
        tb_str = traceback.format_exc()
        self.logger.error(f"Exception in {context}: {exception}\nTraceback:\n{tb_str}")
    
    def create_alert(self, message: str, level: str = "WARNING", extra_data: Optional[Dict] = None):
        """Create a structured alert"""
        logger = logging.getLogger("ALERT")
        
        if extra_data:
            # Add extra data to log record
            record = logging.LogRecord(
                name="ALERT",
                level=getattr(logging, level),
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            
            for key, value in extra_data.items():
                setattr(record, key, value)
                
            logger.handle(record)
        else:
            getattr(logger, level.lower())(message)


# Global logging manager instance
_logging_manager = None

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def setup_logging(log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 enable_monitoring: bool = True) -> LoggingManager:
    """
    Setup logging for the application
    
    Args:
        log_dir: Log directory path
        log_level: Minimum log level
        enable_monitoring: Whether to start system monitoring
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    _logging_manager = LoggingManager(log_dir=log_dir, log_level=log_level)
    
    if enable_monitoring:
        _logging_manager.start_monitoring()
    
    return _logging_manager


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging_manager = setup_logging("test_logs", logging.DEBUG)
    
    # Create test logger
    logger = logging.getLogger("test_module")
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test timed operation
    with logging_manager.timed_operation("test_operation"):
        time.sleep(1)
        logger.info("Doing some work...")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logging_manager.log_exception(e, "test_context")
    
    # Test alert creation
    logging_manager.create_alert(
        "High CPU usage detected",
        level="WARNING",
        extra_data={'cpu_percent': 85.5, 'threshold': 80}
    )
    
    # Record test trading metrics
    trading_metrics = TradingMetrics(
        timestamp=datetime.now(),
        orders_placed=10,
        orders_filled=8,
        orders_cancelled=2,
        total_pnl=1250.75,
        unrealized_pnl=350.25,
        realized_pnl=900.50,
        portfolio_value=105000.00,
        cash_balance=25000.00,
        position_count=15,
        active_strategies=3,
        data_feed_latency_ms=45.2,
        order_execution_time_ms=125.8
    )
    
    logging_manager.record_trading_metrics(trading_metrics)
    
    # Get system status
    status = logging_manager.get_system_status()
    logger.info(f"System status: {status}")
    
    # Wait a bit for monitoring
    time.sleep(5)
    
    # Query recent logs
    recent_logs = logging_manager.query_logs(
        start_time=datetime.now() - timedelta(minutes=5),
        limit=10
    )
    
    if not recent_logs.empty:
        logger.info(f"Found {len(recent_logs)} recent log entries")
    
    # Stop monitoring
    logging_manager.stop_monitoring()
    
    print("Logging system test completed!")
