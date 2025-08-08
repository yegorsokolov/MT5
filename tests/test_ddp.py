import sys
import pathlib
import importlib.util
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "resource_monitor", ROOT / "utils" / "resource_monitor.py"
)
resource_monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_monitor)  # type: ignore
ResourceMonitor = resource_monitor.ResourceMonitor


def _worker(rank, world_size, return_dict):
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method="tcp://127.0.0.1:29500"
    )
    model = torch.nn.Linear(4, 2)
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.1)
    data = torch.ones(2, 4)
    target = torch.zeros(2, 2)
    opt.zero_grad()
    loss = ((ddp(data) - target) ** 2).mean()
    loss.backward()
    opt.step()
    return_dict[rank] = {k: v.detach().clone() for k, v in ddp.module.state_dict().items()}
    dist.destroy_process_group()


def test_ddp_gradient_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    manager = mp.Manager()
    ret = manager.dict()
    mp.spawn(_worker, args=(2, ret), nprocs=2)
    w0, b0 = ret[0]["weight"], ret[0]["bias"]
    w1, b1 = ret[1]["weight"], ret[1]["bias"]
    assert torch.allclose(w0, w1)
    assert torch.allclose(b0, b1)


def test_resource_monitor_ddp(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    rm = ResourceMonitor()
    assert rm.capabilities.ddp()
