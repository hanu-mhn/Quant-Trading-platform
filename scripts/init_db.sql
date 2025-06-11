-- Database initialization script for the quantitative trading platform
-- This script creates the necessary tables and indexes for production use

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Switch to trading schema
SET search_path = trading, public;

-- Users and authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    role VARCHAR(20) DEFAULT 'user'
);

-- Accounts and portfolios
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(20) NOT NULL, -- 'live', 'paper', 'backtest'
    broker VARCHAR(50) NOT NULL,
    initial_balance DECIMAL(15,2) NOT NULL DEFAULT 0,
    current_balance DECIMAL(15,2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Securities master data
CREATE TABLE IF NOT EXISTS securities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    security_type VARCHAR(20) NOT NULL, -- 'stock', 'option', 'future', 'forex'
    name VARCHAR(200),
    sector VARCHAR(50),
    industry VARCHAR(100),
    currency VARCHAR(3) DEFAULT 'USD',
    multiplier DECIMAL(10,4) DEFAULT 1.0,
    tick_size DECIMAL(10,8) DEFAULT 0.01,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, exchange)
);

-- Market data storage
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    security_id UUID NOT NULL REFERENCES securities(id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    vwap DECIMAL(12,4),
    bid_price DECIMAL(12,4),
    ask_price DECIMAL(12,4),
    bid_size INTEGER,
    ask_size INTEGER,
    data_source VARCHAR(50) NOT NULL
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id),
    security_id UUID NOT NULL REFERENCES securities(id),
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', 'stop_limit'
    side VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4),
    stop_price DECIMAL(12,4),
    time_in_force VARCHAR(10) DEFAULT 'DAY', -- 'DAY', 'GTC', 'IOC', 'FOK'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'partial', 'cancelled', 'rejected'
    filled_quantity INTEGER DEFAULT 0,
    avg_fill_price DECIMAL(12,4),
    commission DECIMAL(10,4) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP WITH TIME ZONE,
    broker_order_id VARCHAR(100),
    strategy_id VARCHAR(100),
    notes TEXT
);

-- Executions/fills table
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    liquidity_flag VARCHAR(10) -- 'add', 'remove', 'routing'
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id),
    security_id UUID NOT NULL REFERENCES securities(id),
    quantity INTEGER NOT NULL DEFAULT 0,
    avg_cost DECIMAL(12,4) NOT NULL DEFAULT 0,
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, security_id)
);

-- Portfolio snapshots for performance tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id),
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15,2) NOT NULL,
    cash_balance DECIMAL(15,2) NOT NULL,
    equity_value DECIMAL(15,2) NOT NULL,
    day_pnl DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, snapshot_date)
);

-- Strategy definitions
CREATE TABLE IF NOT EXISTS strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    parameters JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES strategies(id),
    account_id UUID NOT NULL REFERENCES accounts(id),
    date DATE NOT NULL,
    pnl DECIMAL(15,2),
    trades_count INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(strategy_id, account_id, date)
);

-- Switch to monitoring schema for system monitoring tables
SET search_path = monitoring, public;

-- System performance logs
CREATE TABLE IF NOT EXISTS system_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    network_in BIGINT,
    network_out BIGINT,
    active_connections INTEGER,
    response_time_ms INTEGER
);

-- Application logs
CREATE TABLE IF NOT EXISTS application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(10) NOT NULL,
    logger VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    module VARCHAR(100),
    function VARCHAR(100),
    line_number INTEGER,
    exception_info TEXT,
    extra_data JSONB
);

-- Trading alerts
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    account_id UUID,
    strategy_id UUID,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID
);

-- Switch to analytics schema for advanced analytics
SET search_path = analytics, public;

-- Feature store for ML features
CREATE TABLE IF NOT EXISTS features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    security_id UUID NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,8),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions and performance
CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    security_id UUID NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- 'price', 'direction', 'volatility'
    predicted_value DECIMAL(15,8),
    confidence DECIMAL(5,4),
    actual_value DECIMAL(15,8),
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    target_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reset search path
SET search_path = trading, monitoring, analytics, public;

-- Create indexes for better performance
-- Trading schema indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON trading.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON trading.users(email);
CREATE INDEX IF NOT EXISTS idx_accounts_user_id ON trading.accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_accounts_type ON trading.accounts(account_type);
CREATE INDEX IF NOT EXISTS idx_securities_symbol ON trading.securities(symbol);
CREATE INDEX IF NOT EXISTS idx_securities_exchange ON trading.securities(exchange);
CREATE INDEX IF NOT EXISTS idx_market_data_security_timestamp ON trading.market_data(security_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON trading.market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_account_id ON trading.orders(account_id);
CREATE INDEX IF NOT EXISTS idx_orders_security_id ON trading.orders(security_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading.orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON trading.orders(created_at);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON trading.executions(order_id);
CREATE INDEX IF NOT EXISTS idx_positions_account_id ON trading.positions(account_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account_date ON trading.portfolio_snapshots(account_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_date ON trading.strategy_performance(strategy_id, date);

-- Monitoring schema indexes
CREATE INDEX IF NOT EXISTS idx_system_performance_timestamp ON monitoring.system_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_timestamp ON monitoring.application_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_level ON monitoring.application_logs(level);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON monitoring.alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON monitoring.alerts(severity);

-- Analytics schema indexes
CREATE INDEX IF NOT EXISTS idx_features_security_timestamp ON analytics.features(security_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_name_timestamp ON analytics.features(feature_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_date ON analytics.model_predictions(model_name, prediction_date);
CREATE INDEX IF NOT EXISTS idx_model_predictions_security_date ON analytics.model_predictions(security_id, prediction_date);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON trading.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON trading.accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON trading.strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default data
INSERT INTO trading.users (username, email, password_hash, role) VALUES 
('admin', 'admin@tradingplatform.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyoXKMxXKF8Faq', 'admin'),
('demo_user', 'demo@tradingplatform.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyoXKMxXKF8Faq', 'user')
ON CONFLICT (username) DO NOTHING;

-- Insert some sample securities
INSERT INTO trading.securities (symbol, exchange, security_type, name, sector, industry) VALUES 
('AAPL', 'NASDAQ', 'stock', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
('GOOGL', 'NASDAQ', 'stock', 'Alphabet Inc.', 'Technology', 'Internet Services'),
('MSFT', 'NASDAQ', 'stock', 'Microsoft Corporation', 'Technology', 'Software'),
('TSLA', 'NASDAQ', 'stock', 'Tesla Inc.', 'Consumer Cyclical', 'Auto Manufacturers'),
('SPY', 'ARCA', 'etf', 'SPDR S&P 500 ETF Trust', 'Financial', 'Exchange Traded Fund'),
('QQQ', 'NASDAQ', 'etf', 'Invesco QQQ Trust', 'Financial', 'Exchange Traded Fund')
ON CONFLICT (symbol, exchange) DO NOTHING;

-- Create sample strategies
INSERT INTO trading.strategies (name, description, strategy_type, parameters) VALUES 
('Mean Reversion', 'Basic mean reversion strategy using RSI', 'mean_reversion', '{"rsi_period": 14, "oversold": 30, "overbought": 70}'),
('Momentum', 'Momentum strategy using moving averages', 'momentum', '{"short_ma": 10, "long_ma": 50, "volume_threshold": 1000000}'),
('Pairs Trading', 'Statistical arbitrage using correlated pairs', 'pairs_trading', '{"lookback_period": 20, "entry_threshold": 2.0, "exit_threshold": 0.5}')
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trader;

-- Create views for common queries
CREATE OR REPLACE VIEW trading.portfolio_summary AS
SELECT 
    a.id as account_id,
    a.account_name,
    a.current_balance,
    COUNT(p.id) as total_positions,
    SUM(p.market_value) as total_equity,
    SUM(p.unrealized_pnl) as total_unrealized_pnl,
    SUM(p.realized_pnl) as total_realized_pnl
FROM trading.accounts a
LEFT JOIN trading.positions p ON a.id = p.account_id
WHERE a.is_active = true
GROUP BY a.id, a.account_name, a.current_balance;

CREATE OR REPLACE VIEW trading.daily_performance AS
SELECT 
    ps.account_id,
    ps.snapshot_date,
    ps.total_value,
    ps.day_pnl,
    ps.total_pnl,
    LAG(ps.total_value) OVER (PARTITION BY ps.account_id ORDER BY ps.snapshot_date) as prev_value,
    CASE WHEN LAG(ps.total_value) OVER (PARTITION BY ps.account_id ORDER BY ps.snapshot_date) > 0
         THEN (ps.total_value - LAG(ps.total_value) OVER (PARTITION BY ps.account_id ORDER BY ps.snapshot_date)) / 
              LAG(ps.total_value) OVER (PARTITION BY ps.account_id ORDER BY ps.snapshot_date) * 100
         ELSE 0 END as daily_return_pct
FROM trading.portfolio_snapshots ps
ORDER BY ps.account_id, ps.snapshot_date;

-- Performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Log completion
INSERT INTO monitoring.application_logs (level, logger, message, module) 
VALUES ('INFO', 'database', 'Database initialization completed successfully', 'init_db');

COMMIT;
