"""
Quantitative Trading Platform
============================

A comprehensive platform for quantitative trading strategy development,
backtesting, and live trading execution.

Modules:
--------
- core: Core backtesting and live trading engines
- data: Data collection, processing, and loading utilities
- strategies: Trading strategy implementations
- indicators: Technical indicators and signal generators
- models: Machine learning and statistical models
- portfolio: Portfolio management and optimization
- risk_management: Risk management tools and metrics
- brokers: Broker interface implementations
- visualization: Charting and analysis visualization tools
- common: Common utilities and helper functions
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading Team"
__email__ = "team@quanttrading.com"

# Core imports for easy access
from .core import *
from .common import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__"
]
