from prometheus_client import Counter, Gauge

QUEUE_DEPTH = Gauge('queue_depth', 'Number of pending messages in signal queue')
TRADE_COUNT = Counter('trade_count', 'Total trades executed')
ERROR_COUNT = Counter('error_count', 'Total errors logged')
RECONNECT_COUNT = Counter('reconnect_attempts', 'Total MT5 reconnection attempts')
DRIFT_EVENTS = Counter('drift_events', 'Detected data drift events')
