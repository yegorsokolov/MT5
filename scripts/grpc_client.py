import asyncio
from pathlib import Path

import grpc
from google.protobuf import empty_pb2
from proto import management_pb2, management_pb2_grpc


async def main():
    cert = Path(__file__).resolve().parents[1] / "certs/ca.crt"
    creds = grpc.ssl_channel_credentials(root_certificates=cert.read_bytes())
    async with grpc.aio.secure_channel("localhost:50051", creds) as channel:
        stub = management_pb2_grpc.ManagementServiceStub(channel)
        metadata = [("x-api-key", "your_key")]
        resp = await stub.StartBot(
            management_pb2.StartRequest(bot_id="bot1"), metadata=metadata
        )
        print("start:", resp.status)
        bots = await stub.ListBots(empty_pb2.Empty(), metadata=metadata)
        print("bots:", bots.bots)
        status = await stub.BotStatus(
            management_pb2.BotStatusRequest(bot_id="bot1", lines=10),
            metadata=metadata,
        )
        print("running", status.running, "pid", status.pid)
        logs = await stub.GetLogs(
            management_pb2.LogRequest(lines=5), metadata=metadata
        )
        print("logs:\n", logs.logs)
        cfg_resp = await stub.UpdateConfig(
            management_pb2.ConfigChange(key="threshold", value="0.6", reason="test"),
            metadata=metadata,
        )
        print("config:", cfg_resp.status)
        await stub.StopBot(
            management_pb2.BotIdRequest(bot_id="bot1"), metadata=metadata
        )


if __name__ == "__main__":
    asyncio.run(main())
