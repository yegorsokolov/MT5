from prometheus_client import Counter, Gauge

QUEUE_DEPTH = Gauge('queue_depth', 'Number of pending messages in signal queue')
OFFSET_GAUGE = Gauge('queue_offset', 'Last processed offset in durable signal queue')
TRADE_COUNT = Counter('trade_count', 'Total trades executed')
ERROR_COUNT = Counter('error_count', 'Total errors logged')
RECONNECT_COUNT = Counter('reconnect_attempts', 'Total MT5 reconnection attempts')
DRIFT_EVENTS = Counter('drift_events', 'Detected data drift events')
TARGET_RISK = Gauge('target_risk', 'Target risk allocation per trade')
REALIZED_RISK = Gauge('realized_risk', 'Realized risk from executed trade')
ADJ_TARGET_RISK = Gauge('adj_target_risk', 'Confidence-adjusted target risk per trade')
ADJ_REALIZED_RISK = Gauge('adj_realized_risk', 'Confidence-adjusted realized risk per trade')
PORTFOLIO_DRAWDOWN = Gauge('portfolio_drawdown', 'Current portfolio drawdown')
DIVERSIFICATION_RATIO = Gauge('diversification_ratio', 'Portfolio diversification ratio')
