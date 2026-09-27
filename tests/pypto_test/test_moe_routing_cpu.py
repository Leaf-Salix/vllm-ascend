"""CPU regression for arming buffers created by worker inference/graph capture."""

from types import SimpleNamespace

import torch
from offline_pd.moe_routing import RoutingCapture


def test_arm_inference_buffers_without_replacing_graph_addresses(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    capture = RoutingCapture(tokens=2, layers=1)
    with torch.inference_mode():
        capture.save(0, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                     torch.tensor([2, 2], dtype=torch.int64), 1)
    buffers = capture.buffers[0]
    addresses = {key: buffers[key].data_ptr() for key in ("topk_ids", "group_list")}
    assert torch.is_inference(buffers["topk_ids"])
    assert not torch.is_inference_mode_enabled()
    capture.arm()
    for key, address in addresses.items():
        assert buffers[key].data_ptr() == address
        assert bool((buffers[key] == -1).all())
