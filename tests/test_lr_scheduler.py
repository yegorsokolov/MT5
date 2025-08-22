import importlib.util
import os
import torch

spec = importlib.util.spec_from_file_location(
    "lr_scheduler", os.path.join(os.path.dirname(__file__), "..", "utils", "lr_scheduler.py")
)
lr_scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr_scheduler)
LookaheadAdamW = lr_scheduler.LookaheadAdamW


def test_scheduler_decreases_lr_on_variance_increase():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = LookaheadAdamW([param], lr=0.005)
    param.grad = torch.tensor([0.1, 0.1])
    opt.step()
    opt.zero_grad()
    lr1 = opt.get_lr()
    param.grad = torch.tensor([1.0, -1.0])
    opt.step()
    opt.zero_grad()
    lr2 = opt.get_lr()
    assert lr2 < lr1


def test_scheduler_increases_lr_on_variance_drop():
    param = torch.nn.Parameter(torch.zeros(2))
    opt = LookaheadAdamW([param], lr=0.005)
    param.grad = torch.tensor([1.0, -1.0])
    opt.step()
    opt.zero_grad()
    lr1 = opt.get_lr()
    param.grad = torch.tensor([0.1, 0.1])
    opt.step()
    opt.zero_grad()
    lr2 = opt.get_lr()
    assert lr2 > lr1
