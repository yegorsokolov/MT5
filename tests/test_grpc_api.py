import sys
import types
import importlib
import asyncio
from pathlib import Path

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
    sys.modules['log_utils'] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: None,
        log_exceptions=lambda f: f,
    )
    ra = importlib.reload(importlib.import_module('remote_api'))
    ra.API_KEY = 'token'
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
    port = server.add_insecure_port('localhost:0')
    await server.start()
    return server, port


@pytest.mark.asyncio
async def test_list_and_update(tmp_path):
    ra, grpc_mod, updates = load_grpc(tmp_path / 'app.log')
    server, port = await start_server(grpc_mod)

    async with grpc.aio.insecure_channel(f'localhost:{port}') as channel:
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

    async with grpc.aio.insecure_channel(f'localhost:{port}') as channel:
        stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
        md = (('x-api-key', 'token'),)
        resp = await stub.StartBot(grpc_mod.management_pb2.StartRequest(bot_id='b1'), metadata=md)
        assert resp.status == 'started'
        assert 'b1' in ra.bots
        resp2 = await stub.StopBot(grpc_mod.management_pb2.BotIdRequest(bot_id='b1'), metadata=md)
        assert resp2.status == 'stopped'
        assert 'b1' not in ra.bots

    await server.stop(None)
