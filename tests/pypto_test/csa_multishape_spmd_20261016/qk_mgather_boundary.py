# SPDX-License-Identifier: Apache-2.0
"""MGATHER候选的64x512 BF16设备边界检查，计时窗口之外执行。"""

import pypto.language as pl
import torch


@pl.jit.incore
def _gather(mem: pl.Tensor[[96, 512], pl.BF16],
            raw: pl.Tensor[[1, 64], pl.INT32],
            physical: pl.Tensor[[1, 64], pl.INT32],
            out: pl.Out[pl.Tensor[[64, 512], pl.BF16]]):
    raw_tile = pl.tile.load(raw, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
    physical_tile = pl.tile.load(physical, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
    negative = pl.tile.minimums(raw_tile, 0)
    rows = pl.tile.or_(physical_tile, negative)
    value = pl.tile.mgather(mem, rows, coalesce="row", gather_oob="zero")
    out = pl.tile.store(value, [0, 0], out)
    return out


@pl.jit
def _entry(mem: pl.Tensor[[96, 512], pl.BF16],
           raw: pl.Tensor[[1, 64], pl.INT32],
           physical: pl.Tensor[[1, 64], pl.INT32],
           out: pl.Out[pl.Tensor[[64, 512], pl.BF16]]):
    with pl.spmd(1):
        out = _gather(mem, raw, physical, out)
    return out


def validate(device, options):
    # 重复、逆序、首尾地址及三种负数哨兵；设备输入地址始终不变。
    mem_cpu = (torch.arange(96 * 512).reshape(96, 512).remainder(251) / 16).to(torch.bfloat16)
    physical_cpu = ((torch.arange(64) * 37 + 95) % 96).to(torch.int32).reshape(1, 64)
    raw_cpu = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    mixed = raw_cpu.clone()
    mixed[0, ::3] = -1
    mixed[0, 1::7] = -2
    mixed[0, 2::11] = -2147483648
    cases = {"all_valid": raw_cpu, "mixed_negative": mixed,
             "all_negative": torch.full_like(raw_cpu, -1)}
    mem, physical = mem_cpu.to(device), physical_cpu.to(device)
    raw = raw_cpu.to(device)
    allocation = torch.full((66, 512), -777, dtype=torch.bfloat16, device=device)
    out = allocation[1:65]
    _entry(mem, raw, physical, out, config=options)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _entry(mem, raw, physical, out, config=options)
    checks = []
    for name, indexes in cases.items():
        raw.copy_(indexes.to(device))
        out.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        expected = mem_cpu[physical_cpu.flatten().long()].clone()
        expected[indexes.flatten() < 0] = 0
        actual = allocation.cpu()
        exact = torch.equal(actual[1:65], expected)
        guard = bool((actual[[0, 65]] == torch.tensor(-777, dtype=torch.bfloat16)).all())
        if not exact or not guard:
            raise AssertionError(f"QK MGATHER边界失败：{name} exact={exact} guard={guard}")
        checks.append({"case": name, "exact": exact, "guard": guard})
    if not torch.equal(mem.cpu(), mem_cpu) or not torch.equal(physical.cpu(), physical_cpu):
        raise AssertionError("QK MGATHER改变只读输入")
    return {"passed": True, "shape": [64, 512], "graph_replay_device_metadata_changed": True,
            "read_only_inputs_preserved": True, "checks": checks}
