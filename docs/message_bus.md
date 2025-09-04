# Message Bus Deployment

The project communicates through a small message bus abstraction which can
use **NATS**, **Kafka** or a lightweight in-memory fallback.  Three topics are
used throughout the system:

| Topic    | Purpose                      | Retention/Backpressure |
|----------|-----------------------------|------------------------|
| `ticks`  | Raw market tick data         | keep ~1 minute / 10k msg|
| `features` | Engineered feature frames | keep ~10 minutes / 5k msg|
| `signals`  | Trading signals            | keep ~1 hour / 1k msg|

The in-memory backend enforces the limits above by dropping the oldest message
when a topic exceeds its `max_msgs` threshold.  External brokers should be
configured with equivalent retention and backpressure policies.

## Deploying NATS

NATS with JetStream can be launched via Docker:

```bash
docker run -p 4222:4222 -p 8222:8222 -ti nats:latest -js
```

Create streams for each topic with matching limits:

```bash
nats stream add ticks    --subjects ticks    --retention limits --max-msgs 10000 --max-age 1m
nats stream add features --subjects features --retention limits --max-msgs 5000  --max-age 10m
nats stream add signals  --subjects signals  --retention limits --max-msgs 1000  --max-age 1h
```

## Deploying Kafka

A single-node Kafka broker suitable for testing can also be launched using
Docker:

```bash
docker run -p 2181:2181 -p 9092:9092 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -ti wurstmeister/kafka
```

Create the topics with retention policies:

```bash
kafka-topics --create --topic ticks    --bootstrap-server localhost:9092 --config retention.ms=60000 --partitions 1 --replication-factor 1
kafka-topics --create --topic features --bootstrap-server localhost:9092 --config retention.ms=600000 --partitions 1 --replication-factor 1
kafka-topics --create --topic signals  --bootstrap-server localhost:9092 --config retention.ms=3600000 --partitions 1 --replication-factor 1
```

Both brokers can then be used by setting `MESSAGE_BUS_BACKEND` to `nats` or
`kafka` respectively.
