import asyncio
from pathlib import Path

import grpc
from fastapi import HTTPException
from google.protobuf import empty_pb2

import remote_api as ra
from proto import management_pb2, management_pb2_grpc
from risk_manager import risk_manager


class ManagementServicer(management_pb2_grpc.ManagementServiceServicer):
    async def _authorize(self, context: grpc.aio.ServicerContext) -> None:
        api_key = None
        for key, value in context.invocation_metadata():
            if key == "x-api-key":
                api_key = value
                break
        if ra.API_KEY and api_key != ra.API_KEY:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")

    async def StartBot(self, request: management_pb2.StartRequest, context):
        await self._authorize(context)
        try:
            data = await ra.start_bot(request.bot_id)
        except HTTPException as exc:
            await context.abort(grpc.StatusCode(exc.status_code), exc.detail)
        return management_pb2.StatusResponse(status=data["status"])

    async def StopBot(self, request: management_pb2.BotIdRequest, context):
        await self._authorize(context)
        try:
            data = await ra.stop_bot(request.bot_id)
        except HTTPException as exc:
            code = grpc.StatusCode.NOT_FOUND if exc.status_code == 404 else grpc.StatusCode(exc.status_code)
            await context.abort(code, exc.detail)
        return management_pb2.StatusResponse(status=data["status"])

    async def BotStatus(self, request: management_pb2.BotStatusRequest, context):
        await self._authorize(context)
        try:
            data = await ra.bot_status(request.bot_id, lines=request.lines)
        except HTTPException as exc:
            code = grpc.StatusCode.NOT_FOUND if exc.status_code == 404 else grpc.StatusCode(exc.status_code)
            await context.abort(code, exc.detail)
        return management_pb2.BotStatusResponse(
            bot_id=request.bot_id,
            running=data["running"],
            pid=data["pid"],
            returncode=data["returncode"],
            logs=data["logs"],
        )

    async def GetLogs(self, request: management_pb2.LogRequest, context):
        await self._authorize(context)
        try:
            data = await ra.get_logs(lines=request.lines)
        except HTTPException as exc:
            code = grpc.StatusCode.NOT_FOUND if exc.status_code == 404 else grpc.StatusCode(exc.status_code)
            await context.abort(code, exc.detail)
        return management_pb2.LogResponse(logs=data["logs"])

    async def ListBots(self, request: empty_pb2.Empty, context):
        await self._authorize(context)
        data = await ra.list_bots()
        return management_pb2.BotList(bots=data)

    async def UpdateConfig(self, request: management_pb2.ConfigChange, context):
        await self._authorize(context)
        change = ra.ConfigUpdate(key=request.key, value=request.value, reason=request.reason)
        try:
            data = await ra.update_configuration(change)
        except HTTPException as exc:
            await context.abort(grpc.StatusCode(exc.status_code), exc.detail)
        return management_pb2.StatusResponse(status=data["status"])

    async def GetRiskStatus(self, request, context):
        await self._authorize(context)
        data = risk_manager.status()
        return management_pb2.RiskStatus(
            exposure=data["exposure"],
            daily_loss=data["daily_loss"],
            var=data["var"],
            trading_halted=data["trading_halted"],
        )


BASE_DIR = Path(__file__).resolve().parent


async def serve(address: str = "[::]:50051") -> None:
    server = grpc.aio.server()
    management_pb2_grpc.add_ManagementServiceServicer_to_server(
        ManagementServicer(), server
    )

    cert_dir = BASE_DIR / "certs"
    private_key = (cert_dir / "server.key").read_bytes()
    certificate_chain = (cert_dir / "server.crt").read_bytes()
    server_credentials = grpc.ssl_server_credentials(
        ((private_key, certificate_chain),)
    )
    server.add_secure_port(address, server_credentials)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
