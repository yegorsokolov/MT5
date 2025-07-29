from prometheus_client import Counter, Gauge

QUEUE_DEPTH = Gauge('queue_depth', 'Number of pending messages in signal queue')
TRADE_COUNT = Counter('trade_count', 'Total trades executed')
ERROR_COUNT = Counter('error_count', 'Total errors logged')
