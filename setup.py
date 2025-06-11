"""
Setup configuration for Quantitative Trading Platform

Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.
Developer: Malavath Hanmanth Nayak
Contact: hanmanthnayak.95@gmail.com
GitHub: https://github.com/hanu-mhn
LinkedIn: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/
"""

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quant-trading-platform",
    version="1.0.0",
    author="Malavath Hanmanth Nayak",
    author_email="hanmanthnayak.95@gmail.com",
    description="A comprehensive quantitative trading platform with advanced analytics and backtesting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanu-mhn/quant-trading-platform",
    project_urls={
        "Bug Tracker": "https://github.com/hanu-mhn/quant-trading-platform/issues",
        "Documentation": "https://github.com/hanu-mhn/quant-trading-platform/blob/main/README.md",
        "Source Code": "https://github.com/hanu-mhn/quant-trading-platform",
        "LinkedIn": "https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "interactive_brokers": [
            "ib_insync>=0.9.70",
        ],
        "ml": [
            "scikit-learn>=1.2.0",
            "tensorflow>=2.12.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-platform=src.main:main",
            "quant-dashboard=src.dashboard.trading_dashboard:main",
            "quant-api=src.api.api_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.sql", "*.md"],
    },
    keywords=[
        "quantitative finance",
        "algorithmic trading",
        "backtesting",
        "portfolio management",
        "financial analysis",
        "trading strategies",
        "market data",
        "risk management",
    ],
    license="MIT",
    zip_safe=False,
)
