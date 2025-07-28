import asyncio
import grpc
from proto import management_pb2, management_pb2_grpc


async def main():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = management_pb2_grpc.ManagementServiceStub(channel)
        metadata = [("x-api-key", "your_key")]
        resp = await stub.StartBot(management_pb2.StartRequest(bot_id="bot1"), metadata=metadata)
        print("start:", resp.status)
        status = await stub.BotStatus(management_pb2.BotStatusRequest(bot_id="bot1", lines=10), metadata=metadata)
        print("running", status.running, "pid", status.pid)
        logs = await stub.GetLogs(management_pb2.LogRequest(lines=5), metadata=metadata)
        print("logs:\n", logs.logs)
        await stub.StopBot(management_pb2.BotIdRequest(bot_id="bot1"), metadata=metadata)


if __name__ == "__main__":
    asyncio.run(main())
