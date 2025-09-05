from prometheus_client import Counter, Gauge

BATCH_LATENCY = Gauge('batch_latency', 'Processing latency for tick batches')
QUEUE_DEPTH = Gauge('queue_depth', 'Number of pending messages in signal queue')
OFFSET_GAUGE = Gauge('queue_offset', 'Last processed offset in durable signal queue')
TRADE_COUNT = Counter('trade_count', 'Total trades executed')
ERROR_COUNT = Counter('error_count', 'Total errors logged')
RECONNECT_COUNT = Counter('reconnect_attempts', 'Total MT5 reconnection attempts')
TICK_ANOMALIES = Counter('tick_anomalies', 'Counts of tick data anomalies', ['type'])
DRIFT_EVENTS = Counter('drift_events', 'Detected data drift events')
PIPELINE_ANOMALY_TOTAL = Counter('pipeline_anomaly_total', 'Total pipeline anomalies detected')
PIPELINE_ANOMALY_RATE = Gauge('pipeline_anomaly_rate', 'Rate of pipeline anomalies')
TARGET_RISK = Gauge('target_risk', 'Target risk allocation per trade')
REALIZED_RISK = Gauge('realized_risk', 'Realized risk from executed trade')
ADJ_TARGET_RISK = Gauge('adj_target_risk', 'Confidence-adjusted target risk per trade')
ADJ_REALIZED_RISK = Gauge('adj_realized_risk', 'Confidence-adjusted realized risk per trade')
PORTFOLIO_DRAWDOWN = Gauge('portfolio_drawdown', 'Current portfolio drawdown')
DIVERSIFICATION_RATIO = Gauge('diversification_ratio', 'Portfolio diversification ratio')
SLIPPAGE_BPS = Gauge('slippage_bps', 'Configured slippage in basis points')
REALIZED_SLIPPAGE_BPS = Gauge('realized_slippage_bps', 'Realized slippage in basis points')
PARTIAL_FILL_COUNT = Counter('partial_fill_count', 'Trades partially filled due to insufficient liquidity')
SKIPPED_TRADE_COUNT = Counter('skipped_trade_count', 'Trades skipped due to insufficient liquidity')
FEATURE_ANOMALIES = Counter('feature_anomalies', 'Detected feature anomalies')
RESOURCE_RESTARTS = Counter('resource_restarts', 'Graceful restarts triggered by resource watchdog')
# Broker connection metrics
BROKER_LATENCY_MS = Gauge('broker_latency_ms', 'Heartbeat latency to broker', ['broker'])
BROKER_FAILURES = Counter('broker_failures', 'Broker heartbeat failures', ['broker'])
# Plugin reload metrics
PLUGIN_RELOADS = Counter('plugin_reloads', 'Total plugin reloads')
# Prediction cache metrics
PRED_CACHE_HIT = Counter('pred_cache_hit', 'Predictions served from cache')
# Resource utilization metrics
CPU_USAGE = Gauge('cpu_usage_pct', 'Process CPU utilization percentage')
RSS_USAGE = Gauge('rss_usage_mb', 'Process resident set size in megabytes')
