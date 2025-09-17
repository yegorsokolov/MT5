import asyncio
import time

from services.message_bus import MessageBus, TOPIC_CONFIG, Topics


def test_message_expiration(monkeypatch):
    async def runner() -> None:
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

    asyncio.run(runner())


def test_publish_bytes_message():
    async def runner() -> None:
        bus = MessageBus()
        payload = b"{\"key\": \"value\"}"
        await bus.publish(Topics.SIGNALS, payload)

        history = bus.get_history(Topics.SIGNALS)
        assert history[-1] == payload

        subscriber = bus.subscribe(Topics.SIGNALS)
        msg = await subscriber.__anext__()
        assert msg == payload

    asyncio.run(runner())
