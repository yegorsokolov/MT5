
import pandas as pd
from signal_queue import KafkaSignalQueue, RedisSignalQueue
from testcontainers.kafka import KafkaContainer
from testcontainers.redis import RedisContainer


def sample_df():
    return pd.DataFrame({
        "Timestamp": pd.date_range("2024", periods=3, freq="s"),
        "Symbol": ["EURUSD"] * 3,
        "prob": [0.1, 0.2, 0.3],
    })


def test_kafka_signal_queue_persistence():
    df = sample_df()
    with KafkaContainer() as kafka:
        queue = KafkaSignalQueue(topic="signals", bootstrap_servers=kafka.get_bootstrap_servers())
        queue.publish_dataframe(df)
        queue.close()

        consumer = KafkaSignalQueue(
            topic="signals",
            bootstrap_servers=kafka.get_bootstrap_servers(),
            group_id="replay",
        )
        gen = consumer.iter_messages()
        messages = [next(gen) for _ in range(len(df))]
        consumer.close()

    assert len(messages) == len(df)
    assert messages[0]["prob"] == df.iloc[0]["prob"]


def test_redis_signal_queue_persistence():
    df = sample_df()
    with RedisContainer() as redis:
        url = redis.get_connection_url()
        queue = RedisSignalQueue(stream="signals", url=url)
        queue.publish_dataframe(df)

        consumer = RedisSignalQueue(stream="signals", url=url, start_id="0-0")
        gen = consumer.iter_messages()
        messages = [next(gen) for _ in range(len(df))]

    assert len(messages) == len(df)
    assert messages[-1]["prob"] == df.iloc[-1]["prob"]

