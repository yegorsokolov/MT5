import sys
import types
import importlib
import asyncio
from pathlib import Path
import os

import pytest
import grpc

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'proto'))


class DummyProc:
    def __init__(self):
        self.terminated = False
        self.pid = 123
        self.returncode = None

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True


def load_grpc(tmp_log):
    os.environ['API_KEY'] = 'token'
    sys.modules['mlflow'] = types.SimpleNamespace()
    sys.modules['log_utils'] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: None,
        log_exceptions=lambda f: f,
    )
    ra = importlib.reload(importlib.import_module('remote_api'))
    ra.bots.clear()
    Path(ra.LOG_FILE).write_text('line1\nline2\n')
    updates = {}
    ra.update_config = lambda k, v, r: updates.update({k: (v, r)})
    ra.Popen = lambda *a, **k: DummyProc()
    grpc_mod = importlib.reload(importlib.import_module('grpc_api'))
    return ra, grpc_mod, updates


async def start_server(grpc_mod):
    server = grpc.aio.server()
    grpc_mod.management_pb2_grpc.add_ManagementServiceServicer_to_server(
        grpc_mod.ManagementServicer(), server
    )
    cert_dir = Path(__file__).resolve().parents[1] / 'certs'
    private_key = (cert_dir / 'server.key').read_bytes()
    certificate_chain = (cert_dir / 'server.crt').read_bytes()
    server_credentials = grpc.ssl_server_credentials(
        ((private_key, certificate_chain),)
    )
    port = server.add_secure_port('localhost:0', server_credentials)
    await server.start()
    return server, port


@pytest.mark.asyncio
async def test_list_and_update(tmp_path):
    ra, grpc_mod, updates = load_grpc(tmp_path / 'app.log')
    server, port = await start_server(grpc_mod)

    cert_dir = Path(__file__).resolve().parents[1] / 'certs'
    creds = grpc.ssl_channel_credentials(
        root_certificates=(cert_dir / 'ca.crt').read_bytes()
    )
    async with grpc.aio.secure_channel(f'localhost:{port}', creds) as channel:
        stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await stub.ListBots(grpc_mod.empty_pb2.Empty())
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

        metadata = (('x-api-key', 'token'),)
        resp = await stub.ListBots(grpc_mod.empty_pb2.Empty(), metadata=metadata)
        assert resp.bots == {}

        cfg = grpc_mod.management_pb2.ConfigChange(
            key='threshold', value='0.7', reason='test'
        )
        res = await stub.UpdateConfig(cfg, metadata=metadata)
        assert res.status == 'updated'
        assert updates['threshold'] == ('0.7', 'test')

    await server.stop(None)


@pytest.mark.asyncio
async def test_start_stop(tmp_path):
    ra, grpc_mod, _ = load_grpc(tmp_path / 'app.log')
    server, port = await start_server(grpc_mod)

    cert_dir = Path(__file__).resolve().parents[1] / 'certs'
    creds = grpc.ssl_channel_credentials(
        root_certificates=(cert_dir / 'ca.crt').read_bytes()
    )
    async with grpc.aio.secure_channel(f'localhost:{port}', creds) as channel:
        stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
        md = (('x-api-key', 'token'),)
        resp = await stub.StartBot(grpc_mod.management_pb2.StartRequest(bot_id='b1'), metadata=md)
        assert resp.status == 'started'
        assert 'b1' in ra.bots
        resp2 = await stub.StopBot(grpc_mod.management_pb2.BotIdRequest(bot_id='b1'), metadata=md)
        assert resp2.status == 'stopped'
        assert 'b1' not in ra.bots

    await server.stop(None)


@pytest.mark.asyncio
async def test_status_and_logs(tmp_path):
    ra, grpc_mod, _ = load_grpc(tmp_path / 'app.log')
    server, port = await start_server(grpc_mod)

    cert_dir = Path(__file__).resolve().parents[1] / 'certs'
    creds = grpc.ssl_channel_credentials(
        root_certificates=(cert_dir / 'ca.crt').read_bytes()
    )
    async with grpc.aio.secure_channel(f'localhost:{port}', creds) as channel:
        stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
        md = (("x-api-key", "token"),)
        await stub.StartBot(grpc_mod.management_pb2.StartRequest(bot_id="b1"), metadata=md)

        status = await stub.BotStatus(
            grpc_mod.management_pb2.BotStatusRequest(bot_id="b1", lines=5),
            metadata=md,
        )
        assert status.bot_id == "b1"
        assert status.running is True
        assert status.pid == 123
        assert status.returncode == 0
        assert "line1" in status.logs

        logs = await stub.GetLogs(
            grpc_mod.management_pb2.LogRequest(lines=1), metadata=md
        )
        assert "line2" in logs.logs

        await stub.StopBot(
            grpc_mod.management_pb2.BotIdRequest(bot_id="b1"), metadata=md
        )

    await server.stop(None)
