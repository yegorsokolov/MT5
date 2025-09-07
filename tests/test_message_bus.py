import time
import pytest

from services.message_bus import MessageBus, TOPIC_CONFIG


@pytest.mark.asyncio
async def test_message_expiration(monkeypatch):
    topic = "test_topic"
    TOPIC_CONFIG[topic] = {"retention": 1, "max_msgs": 100}
    bus = MessageBus()

    t = 1000.0
    monkeypatch.setattr(time, "time", lambda: t)
    await bus.publish(topic, "m1")

    t += 0.5
    monkeypatch.setattr(time, "time", lambda: t)
    await bus.publish(topic, "m2")
    assert bus.get_history(topic) == ["m1", "m2"]

    t += 1.0
    monkeypatch.setattr(time, "time", lambda: t)
    await bus.publish(topic, "m3")
    assert bus.get_history(topic) == ["m2", "m3"]
