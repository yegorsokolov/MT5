import os
import pandas as pd
import zmq

_CTX = zmq.Context.instance()


def get_publisher(bind_address: str | None = None) -> zmq.Socket:
    """Return a PUB socket bound to the given address."""
    addr = bind_address or os.getenv("SIGNAL_QUEUE_BIND", "tcp://*:5555")
    sock = _CTX.socket(zmq.PUB)
    sock.bind(addr)
    return sock


def get_subscriber(connect_address: str | None = None, topic: str = "") -> zmq.Socket:
    """Return a SUB socket connected to the given address."""
    addr = connect_address or os.getenv("SIGNAL_QUEUE_URL", "tcp://localhost:5555")
    sock = _CTX.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock


def publish_dataframe(sock: zmq.Socket, df: pd.DataFrame) -> None:
    """Publish rows of a dataframe as JSON messages."""
    for _, row in df.iterrows():
        sock.send_json(row.to_dict())
