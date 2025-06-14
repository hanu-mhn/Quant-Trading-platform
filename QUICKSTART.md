# 🚀 Quick Start Guide - Trading Platform

## **Start Trading in 5 Minutes!**

### **Prerequisites**
✅ Python 3.10+ installed  
✅ Virtual environment activated  
✅ All dependencies installed (`pip install -r requirements.txt`)

### **Step 1: Generate Secrets**
```bash
python scripts/secrets_manager.py generate
python scripts/secrets_manager.py create-env --environment development
```

### **Step 2: Start Core Services**
```bash
# Start API Server (Backend)
python scripts/production_manager.py start --service api_server

# Start Dashboard (Frontend)
python scripts/production_manager.py start --service dashboard
```

### **Step 3: Access the Platform**
- **Trading Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### **Step 4: Configure Your First Strategy**
1. Open the dashboard at http://localhost:8501
2. Navigate to "Paper Trading" section
3. Set up your first trading strategy
4. Configure risk parameters
5. Start paper trading!

## **Available Commands**

### **Service Management**
```bash
# Check all services status
python scripts/production_manager.py status

# Start specific service
python scripts/production_manager.py start --service api_server

# Stop specific service
python scripts/production_manager.py stop --service api_server

# Restart service
python scripts/production_manager.py restart --service api_server

# View service logs
python scripts/production_manager.py logs --service api_server
```

### **Backup & Restore**
```bash
# Create backup
python scripts/backup_restore.py backup --type manual

# List backups
python scripts/backup_restore.py list

# Restore from backup
python scripts/backup_restore.py restore --backup-path backups/backup_manual_YYYYMMDD_HHMMSS.tar.gz
```

### **Platform Validation**
```bash
# Check platform health
python scripts/validate_platform.py

# Run specific tests
python -m pytest tests/ -v
```

## **Trading Features Available**

### **📊 Market Data**
- Real-time NSE/BSE data
- Yahoo Finance integration
- Technical indicators (50+)
- Fundamental analysis

### **💰 Trading Capabilities**
- Paper trading (virtual money)
- Multiple brokers support
- Risk management rules
- Portfolio tracking
- Order management

### **📈 Analysis Tools**
- Backtesting engine
- Performance analytics
- Risk metrics
- Profit/Loss tracking

### **🔧 Management Tools**
- Real-time monitoring
- Automated alerts
- Log analysis
- System health checks

## **Default Credentials**

### **Development Environment**
- **API**: No authentication required
- **Dashboard**: Direct access
- **Database**: `postgres:dev_password_123`

### **Production Environment**
- **Secrets**: Auto-generated (check `.env.production`)
- **API**: JWT token required
- **Database**: Secure random passwords

## **Troubleshooting**

### **Common Issues**

**1. Port Already in Use**
```bash
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill process on port 8501
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

**2. Import Errors**
```bash
# Ensure in project root
cd d:\QUANT\QT_python\quant-trading-platform

# Check Python path
python -c "import sys; print(sys.path)"

# Restart with clean environment
python scripts/production_manager.py stop-all
python scripts/production_manager.py start-all
```

**3. Configuration Issues**
```bash
# Regenerate configuration
python scripts/secrets_manager.py create-env --environment development --force

# Validate configuration
python scripts/validate_platform.py
```

## **Next Steps**

1. **Explore the Dashboard** - Try different features and settings
2. **Configure Brokers** - Add your real broker API keys (for live trading)
3. **Create Strategies** - Implement your trading algorithms
4. **Set Risk Rules** - Configure position sizes and stop losses
5. **Monitor Performance** - Use the analytics tools

## **Support**

- **Logs**: Check `logs/` directory for detailed error messages
- **Validation**: Run validation script for comprehensive health check
- **Documentation**: See README.md for detailed information

---

**🎉 Happy Trading!** 

Your quantitative trading platform is ready for action. Start with paper trading to test your strategies safely before moving to live trading.
