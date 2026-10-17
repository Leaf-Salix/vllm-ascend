# SPDX-License-Identifier: Apache-2.0
"""复用实际 QK 候选片段，检查全有效与负索引的图重放边界。"""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import torch

from qk_valid_gather import boundary_body


def prepare_source(body_bundle, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    fragment = ast.unparse(ast.Module(
        body=boundary_body({"body_bundle": str(body_bundle)}), type_ignores=[]))
    source = '''import pypto.language as pl

@pl.jit.incore
def _gather(cmp_kv_flat_inline4388__ssa_v0: pl.Tensor[[96, 512], pl.BF16],
            raw: pl.Tensor[[1, 64], pl.INT32], physical: pl.Tensor[[1, 64], pl.INT32],
            out: pl.Out[pl.Tensor[[64, 512], pl.BF16]]):
    qk_meta_raw = pl.tile.load(raw, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
    qk_meta_physical_rows = pl.tile.load(physical, [0, 0], [1, 64], target_memory=pl.Mem.Vec)
    qk_sb_inline4367__ssa_v1 = 1
    qk_lane_kv_inline4438__ssa_v0 = 0
'''
    source += textwrap.indent(fragment, "    ") + '''
    out = pl.tile.store(qk_kv_half_inline4272__rv_v17, [0, 0], out)
    return out

@pl.jit
def entry(mem: pl.Tensor[[96, 512], pl.BF16], raw: pl.Tensor[[1, 64], pl.INT32],
          physical: pl.Tensor[[1, 64], pl.INT32], out: pl.Out[pl.Tensor[[64, 512], pl.BF16]]):
    with pl.spmd(1):
        out = _gather(mem, raw, physical, out)
    return out
'''
    path = directory / "entry.py"
    path.write_text(source)
    return path


def validate(device, options, body_bundle, output):
    path = prepare_source(body_bundle, output)
    spec = importlib.util.spec_from_file_location("qk_valid_gather_boundary_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mem_cpu = (torch.arange(96 * 512).reshape(96, 512).remainder(251) / 16).to(torch.bfloat16)
    physical_cpu = ((torch.arange(64) * 37 + 95) % 96).to(torch.int32).reshape(1, 64)
    raw_cpu = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    mixed = raw_cpu.clone()
    mixed[0, ::3], mixed[0, 1::7], mixed[0, 2::11] = -1, -2, -2147483648
    # 同一张图在不清零/清零路径之间往返，避免只检查首次 UB 内容。
    cases = [("all_valid", raw_cpu), ("mixed_negative", mixed),
             ("all_negative", torch.full_like(raw_cpu, -1)), ("valid_again", raw_cpu)]
    mem, physical, raw = mem_cpu.to(device), physical_cpu.to(device), raw_cpu.to(device)
    allocation = torch.full((66, 512), -777, dtype=torch.bfloat16, device=device)
    out = allocation[1:65]
    module.entry(mem, raw, physical, out, config=options)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        module.entry(mem, raw, physical, out, config=options)
    checks = []
    for name, indexes in cases:
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
            raise AssertionError(f"QK全有效块边界失败：{name} exact={exact} guard={guard}")
        checks.append({"case": name, "exact": exact, "guard": guard})
    if not torch.equal(mem.cpu(), mem_cpu) or not torch.equal(physical.cpu(), physical_cpu):
        raise AssertionError("QK全有效块搬运改变只读输入")
    return {"passed": True, "actual_candidate_fragment_reused": True,
            "graph_replay_device_metadata_changed": True,
            "read_only_inputs_preserved": True, "checks": checks, "source": str(path)}
