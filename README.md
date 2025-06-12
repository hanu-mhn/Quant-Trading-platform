# Quantitative Trading Platform

A comprehensive, production-ready quantitative trading platform built with Python, featuring advanced analytics, real-time monitoring, paper trading, backtesting, and broker integration capabilities.

## 👨‍💻 Developer Information

**Developer**: Malavath Hanmanth Nayak  
**Contact**: hanmanthnayak.95@gmail.com  
**GitHub**: [@hanu-mhn](https://github.com/hanu-mhn)  
**LinkedIn**: [Hanmanth Nayak](https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/)

**Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.**

## 🚀 Quick Access

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-trading-platform.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/hanu-mhn/quant-trading-platform)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🚀 Features

### Core Trading Features
- **Paper Trading System**: Risk-free strategy testing with realistic market simulation
- **Live Trading Support**: Interactive Brokers integration with real-time execution
- **Advanced Backtesting**: Historical strategy performance analysis with comprehensive metrics
- **Portfolio Management**: Real-time position tracking, P&L calculation, and risk monitoring
- **Multiple Strategies**: Mean reversion, momentum, statistical arbitrage, and custom strategies

### Analytics & Intelligence
- **Advanced Feature Engineering**: 50+ technical and statistical features
- **Machine Learning Integration**: Scikit-learn, TensorFlow support for predictive modeling
- **Market Microstructure Analysis**: Order flow, liquidity, and market impact modeling
- **Risk Management**: VaR, Sharpe ratio, drawdown analysis, and position sizing
- **Alternative Data Integration**: Sentiment analysis and fundamental data processing

### Infrastructure & Monitoring
- **Production-Ready Architecture**: Docker containerization with orchestration
- **Real-Time Dashboard**: Streamlit-based web interface with live updates
- **Comprehensive Monitoring**: Prometheus metrics with Grafana visualization
- **REST API**: FastAPI-based API with authentication and WebSocket support
- **Database Integration**: PostgreSQL for data persistence with Redis caching
- **Load Balancing**: Nginx reverse proxy with SSL termination

### DevOps & Deployment
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment
- **Security**: JWT authentication, encrypted secrets, and security scanning
- **Automated Backups**: Database and configuration backup with retention policies
- **Health Monitoring**: System metrics, alerts, and performance tracking
- **Scalable Architecture**: Microservices design with horizontal scaling support

## 📋 Requirements

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM recommended (4GB minimum)
- **Storage**: 50GB free space recommended (20GB minimum)
- **CPU**: Multi-core processor recommended
- **OS**: Linux, macOS, or Windows with Docker support

### Software Dependencies
- **Docker**: Latest version with Docker Compose
- **Git**: For version control and deployment
- **Python Packages**: See `requirements.txt` for complete list

## 🛠 Installation

### Quick Start (Recommended)

1. **Clone the Repository**
   ```powershell
   git clone https://github.com/your-username/quant-trading-platform.git
   cd quant-trading-platform
   ```

2. **Run Platform Validation**
   ```powershell
   python scripts\validate_platform.py
   ```

3. **Deploy with Docker**
   ```powershell
   python scripts\deploy.py --environment production
   ```

4. **Access the Platform**
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000
   - Monitoring: http://localhost:3000

### Manual Installation

1. **Install Python Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**
   ```powershell
   Copy-Item .env.example .env.production
   # Edit .env.production with your configuration
   ```

3. **Initialize Database**
   ```powershell
   docker-compose up -d database
   docker-compose exec database psql -U trader -d trading_platform -f /docker-entrypoint-initdb.d/init_db.sql
   ```

4. **Start Services**
   ```powershell
   docker-compose up -d
   ```

## 🏗 Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   REST API      │    │  Data Collector │
│   (Streamlit)   │    │   (FastAPI)     │    │   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Trading App   │    │  Paper Trading  │
│  (Reverse Proxy)│    │   (Core Logic)  │    │   (Simulation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │   Monitoring    │
│   (Database)    │    │    (Cache)      │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Market Data Ingestion**: Real-time and historical data collection
2. **Feature Engineering**: Advanced technical and statistical feature calculation
3. **Strategy Execution**: Signal generation and order placement
4. **Risk Management**: Position sizing and risk monitoring
5. **Portfolio Tracking**: Real-time P&L and performance metrics
6. **Monitoring & Alerting**: System health and trading performance alerts

## 📊 Usage

### Running Paper Trading

```python
from src.trading.paper_trading import PaperTradingEngine

# Initialize paper trading
engine = PaperTradingEngine(initial_balance=100000)

# Start trading session
session_id = engine.start_session("test_strategy")

# Place orders
engine.place_order("AAPL", "buy", 100, order_type="market")

# Check portfolio
portfolio = engine.get_portfolio_summary()
print(f"Total Value: ${portfolio['total_value']:,.2f}")
```

### Using the REST API

```python
import requests

# Get portfolio status
response = requests.get("http://localhost:8000/api/v1/portfolio/status")
portfolio = response.json()

# Place a trade
trade_data = {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "order_type": "market"
}
response = requests.post("http://localhost:8000/api/v1/trades", json=trade_data)
```

### Running Backtests

```python
from src.core.backtester import BacktestEngine
from src.strategies.momentum.moving_average_crossover import MovingAverageCrossover

# Initialize backtest
engine = BacktestEngine()
strategy = MovingAverageCrossover(short_window=10, long_window=50)

# Run backtest
results = engine.run_backtest(
    strategy=strategy,
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100000
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## 🧪 Testing

### Run All Tests
```powershell
python -m pytest tests\ -v
```

### Run Specific Test Categories
```powershell
# Unit tests
python -m pytest tests\unit\ -v

# Integration tests
python -m pytest tests\integration\ -v

# Test specific module
python -m pytest tests\test_backtesting.py -v
```

### Coverage Report
```powershell
python -m pytest tests\ --cov=src --cov-report=html
```

## 📈 Monitoring & Analytics

### Accessing Dashboards

1. **Trading Dashboard**: http://localhost:8501
   - Portfolio overview
   - Performance metrics
   - Trading activity
   - System monitoring

2. **Grafana Monitoring**: http://localhost:3000
   - System metrics
   - Application performance
   - Trading analytics
   - Custom alerts

3. **Prometheus Metrics**: http://localhost:9090
   - Raw metrics data
   - Query interface
   - Alert management

### Key Metrics Monitored

- **Portfolio Metrics**: Total value, P&L, drawdown, Sharpe ratio
- **Trading Metrics**: Order fill rate, rejection rate, latency
- **System Metrics**: CPU, memory, disk usage, network traffic
- **Application Metrics**: API response times, error rates, throughput

## 🔒 Security

### Security Features
- **JWT Authentication**: Secure API access with token-based authentication
- **Encrypted Secrets**: All sensitive configuration encrypted at rest
- **Network Security**: Nginx reverse proxy with rate limiting
- **Database Security**: Connection encryption and user isolation
- **Container Security**: Minimal attack surface with non-root containers

### Security Best Practices
- Regular security scanning with Bandit and Safety
- Dependency vulnerability monitoring
- Secure secret management
- Network isolation between services
- Regular backup encryption

## 🚀 Deployment

### Production Deployment

1. **Prepare Environment**
   ```powershell
   python scripts\validate_platform.py
   ```

2. **Configure Production Settings**
   ```powershell
   # Edit production configuration
   notepad .env.production
   ```

3. **Deploy to Production**
   ```powershell
   python scripts\deploy.py --environment production
   ```

4. **Verify Deployment**
   ```powershell
   # Check service health
   curl http://localhost:8000/health
   curl http://localhost:8501
   ```

### Scaling and High Availability

- **Horizontal Scaling**: Multiple API and dashboard instances
- **Database Clustering**: PostgreSQL replication setup
- **Load Balancing**: Nginx with multiple backend servers
- **Monitoring**: Comprehensive alerting and health checks

## 📝 Configuration

### Environment Variables

Key configuration options in `.env` files:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_platform
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Trading Configuration
PAPER_TRADING_ENABLED=true
LIVE_TRADING_ENABLED=false
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=0.05

# Broker Configuration
IB_GATEWAY_HOST=localhost
IB_GATEWAY_PORT=4001
IB_CLIENT_ID=1

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=your-password-here
PROMETHEUS_RETENTION_TIME=30d
```

### Strategy Configuration

Strategies can be configured in `src/strategies/` with custom parameters:

```yaml
# config/strategies.yaml
strategies:
  mean_reversion:
    enabled: true
    parameters:
      rsi_period: 14
      oversold: 30
      overbought: 70
  
  momentum:
    enabled: true
    parameters:
      short_ma: 10
      long_ma: 50
```

## 🤝 Contributing

### Development Setup

1. **Fork and Clone**
   ```powershell
   git clone https://github.com/your-username/quant-trading-platform.git
   cd quant-trading-platform
   ```

2. **Create Development Environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```powershell
   python -m pytest tests\ -v
   ```

### Code Standards

- **Python Style**: Black formatting with flake8 linting
- **Type Hints**: All public functions should have type annotations
- **Documentation**: Docstrings for all classes and functions
- **Testing**: Minimum 80% test coverage for new code

### Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes with appropriate tests
3. Run the full test suite and validation
4. Submit a pull request with detailed description

## 📚 Documentation

### API Documentation
- **REST API**: http://localhost:8000/docs (Swagger UI)
- **API Schema**: http://localhost:8000/redoc

### Code Documentation
- **Python Docstrings**: Inline documentation for all modules
- **Architecture Docs**: `docs/architecture.md`
- **Deployment Guide**: `docs/deployment.md`
- **Strategy Development**: `docs/strategy_development.md`

## 🐛 Troubleshooting

### Common Issues

1. **Docker Build Fails**
   ```powershell
   # Clear Docker cache
   docker system prune -a
   docker-compose build --no-cache
   ```

2. **Database Connection Issues**
   ```powershell
   # Check database container
   docker-compose logs database
   
   # Reset database
   docker-compose down -v
   docker-compose up -d database
   ```

3. **Permission Errors**
   ```powershell
   # Fix file permissions on Windows
   icacls scripts /grant:r Users:F /T
   ```

4. **Memory Issues**
   ```powershell
   # Check system resources
   docker stats
   
   # Increase Docker memory limits
   # Edit docker-compose.yml memory limits
   ```

### Getting Help

- **Issues**: Open a GitHub issue with detailed error information
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers for security-related issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer & Contact

**Developed by**: Malavath Hanmanth Nayak  
**Email**: hanmanthnayak.95@gmail.com  
**GitHub**: [@hanu-mhn](https://github.com/hanu-mhn)  
**LinkedIn**: [Hanmanth Nayak](https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/)

For professional inquiries, collaboration opportunities, or technical support, feel free to reach out via email or GitHub.

**Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.**

## 🙏 Acknowledgments

- **Python Community**: For the excellent ecosystem of libraries
- **Docker**: For containerization and deployment simplification
- **Interactive Brokers**: For providing comprehensive trading APIs
- **Open Source Projects**: Dependencies that make this platform possible

## 📊 Project Status

- **Version**: 1.0.0
- **Status**: Production Ready
- **Last Updated**: 2025-06-11
- **Python Version**: 3.10+
- **Maintainers**: Active

### Roadmap

- [ ] Options and futures trading support
- [ ] Advanced machine learning strategies
- [ ] Mobile application development
- [ ] Cloud deployment templates
- [ ] Multi-broker support expansion
- [ ] Real-time news sentiment analysis
- [ ] Regulatory compliance reporting

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Use at your own risk and comply with all applicable regulations.


