# Monitoring and Alerts

The bot exposes Prometheus metrics on `/metrics`. Key metrics include:

* `cpu_usage_pct` – process CPU utilisation
* `rss_usage_mb` – resident memory usage
* `queue_depth` – pending messages in the signal queue
* `trade_count` – total trades executed
* `drift_events` – detected model drift events

## Prometheus

A sample configuration is provided in `deploy/prometheus.yml`:

```bash
prometheus --config.file=deploy/prometheus.yml
```

## Alerting

Optional alerting rules are defined in `deploy/alert.rules.yml`. Enable them by
referencing the file from your Prometheus configuration and configuring
Alertmanager receivers:

```yaml
rule_files:
  - alert.rules.yml
```

Example alerts cover sustained CPU usage over 90% and broker reconnection
attempts. Configure Alertmanager as usual to deliver notifications.
