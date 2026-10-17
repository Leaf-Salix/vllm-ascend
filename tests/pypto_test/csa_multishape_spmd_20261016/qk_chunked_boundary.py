# SPDX-License-Identifier: Apache-2.0
"""从实际32行KV候选提取原生产片段，测试SWA/压缩KV的同图切换。"""

import ast
import importlib.util
import json
import sys
import textwrap
from pathlib import Path


def prepare_source(body_bundle, output):
    tree = ast.parse(Path(body_bundle).read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "qk_pv_aiv")
    candidates = [node for node in ast.walk(function) if isinstance(node, ast.If)
                  and any(isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                          and n.target.id == "qk_slot_inline4410__ssa_v1" for n in node.body)]
    if len(candidates) != 1 or len(candidates[0].body) != 3:
        raise ValueError("没有找到实际32行KV准备片段")
    producer = ast.Module(body=candidates[0].body, type_ignores=[])
    table_load = next(n for n in ast.walk(producer) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == "pl.tile.load"
                      and ast.unparse(n.args[0]) == "cmp_block_table__ssa_v0")
    columns = ast.literal_eval(table_load.args[2].elts[1])
    valid_columns = ast.literal_eval(table_load.args[3].elts[1])
    source = f'''import pypto.language as pl

@pl.jit.inline
def _gather(ori_kv_flat_inline4326__ssa_v1: pl.Tensor[[96, 512], pl.BF16],
            cmp_kv_flat_inline4388__ssa_v0: pl.Tensor[[96, 512], pl.BF16],
            ori_block_table__ssa_v0: pl.Tensor[[1, 8192], pl.INT32],
            cmp_block_table__ssa_v0: pl.Tensor[[1, {columns}], pl.INT32],
            cmp_sparse_indices_inline4329__rv_v2: pl.Tensor[[1, 512], pl.INT32],
            position_ids_t1__ssa_v0: pl.Tensor[[1, 1], pl.INT64],
            selector: pl.Tensor[[1], pl.INT32],
            kv_transfer_inline4342__ssa_v0: pl.Out[pl.Tensor[[128, 512], pl.BF16]],
            worker: pl.Scalar[pl.INDEX]):
    qk_lane_kv_inline4438__ssa_v0 = worker * 64
    qk_core_inline4347__ssa_v0 = 0
    qk_tick_inline4279__idx_v0 = 0
    qk_t_inline4408__ssa_v1 = 0
    qk_b_inline4274__ssa_v0 = 0
    qk_sb_inline4367__ssa_v1 = pl.cast(pl.tensor.read(selector, [0]), pl.INDEX)
'''
    source += textwrap.indent(ast.unparse(producer), "    ") + f'''
    return kv_transfer_inline4342__ssa_v0

@pl.jit
def entry(ori: pl.Tensor[[96, 512], pl.BF16], cmp: pl.Tensor[[96, 512], pl.BF16],
          ori_table: pl.Tensor[[1, 8192], pl.INT32], cmp_table: pl.Tensor[[1, {columns}], pl.INT32],
          indices: pl.Tensor[[1, 512], pl.INT32], position: pl.Tensor[[1, 1], pl.INT64],
          selector: pl.Tensor[[1], pl.INT32], out: pl.Out[pl.Tensor[[128, 512], pl.BF16]]):
    for worker in pl.spmd(2):
        out = _gather(ori, cmp, ori_table, cmp_table, indices, position, selector, out, worker)
    return out
'''
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    # 原片段来自打印后的 IR，__tile/__ssa 等属于编译器保留后缀。
    # 只重命名局部变量和形参，计算、分支、访存与候选保持一致。
    source_tree = ast.parse(source)
    reserved = sorted({node.id for node in ast.walk(source_tree)
                       if isinstance(node, ast.Name) and "__" in node.id}
                      | {node.arg for node in ast.walk(source_tree)
                         if isinstance(node, ast.arg) and "__" in node.arg})
    names = {name: f"fragment_value_{index}" for index, name in enumerate(reserved)}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = names.get(node.arg, node.arg)
            return self.generic_visit(node)

    source_tree = ast.fix_missing_locations(Rename().visit(source_tree))
    path = directory / "entry.py"
    path.write_text(ast.unparse(source_tree) + "\n")
    return path, columns, valid_columns


def validate(device, options, body_bundle, output):
    import torch

    path, columns, valid_columns = prepare_source(body_bundle, output)
    spec = importlib.util.spec_from_file_location("qk_chunked_boundary_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ori_cpu = (torch.arange(96 * 512).reshape(96, 512).remainder(251) / 16).to(torch.bfloat16)
    cmp_cpu = -ori_cpu - 0.125
    ori_table_cpu = torch.arange(8192, dtype=torch.int32).remainder(3).reshape(1, 8192)
    cmp_table_cpu = (2 - torch.arange(columns, dtype=torch.int32).remainder(3)).reshape(1, columns)
    raw_cpu = (torch.arange(512, dtype=torch.int32) * 37).remainder(valid_columns * 32).reshape(1, 512)
    mixed = raw_cpu.clone()
    mixed[0, ::3], mixed[0, 1::7], mixed[0, 2::11] = -1, -2, -2147483648
    inputs = [ori_cpu.to(device), cmp_cpu.to(device), ori_table_cpu.to(device), cmp_table_cpu.to(device),
              raw_cpu.to(device), torch.tensor([[0]], dtype=torch.int64, device=device),
              torch.tensor([0], dtype=torch.int32, device=device)]
    allocation = torch.full((130, 512), -777, dtype=torch.bfloat16, device=device)
    out = allocation[1:129]
    module.entry(*inputs, out, config=options)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        module.entry(*inputs, out, config=options)
    cases = [("swa_one_row", 0, 0, False, raw_cpu), ("swa_page_boundary", 0, 32, False, raw_cpu),
             ("swa_partial_negative_page", 0, 128, True, raw_cpu),
             ("swa_long_position", 0, 131072, False, raw_cpu),
             ("cmp_valid", 1, 0, False, raw_cpu), ("cmp_last_block_mixed", 4, 0, False, mixed),
             ("cmp_all_negative", 1, 0, False, torch.full_like(raw_cpu, -1)),
             ("swa_return", 0, 0, False, raw_cpu)]
    checks = []
    for name, sb, pos, negative_page, raw in cases:
        table = ori_table_cpu.clone()
        if negative_page:
            table[0, 1::3] = -1
        inputs[2].copy_(table)
        inputs[4].copy_(raw)
        inputs[5].fill_(pos)
        inputs[6].fill_(sb)
        out.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros((128, 512), dtype=torch.bfloat16)
        if sb == 0:
            count = min(pos + 1, 128)
            for row in range(count):
                absolute = pos - count + 1 + row
                page = int(table[0, absolute // 32])
                if page >= 0:
                    expected[row] = ori_cpu[page * 32 + absolute % 32]
        else:
            for row in range(128):
                index = int(raw[0, (sb - 1) * 128 + row])
                if index >= 0:
                    expected[row] = cmp_cpu[int(cmp_table_cpu[0, index // 32]) * 32 + index % 32]
        actual = allocation.cpu()
        exact = torch.equal(actual[1:129], expected)
        guard = bool((actual[[0, 129]] == torch.tensor(-777, dtype=torch.bfloat16)).all())
        read_only = all(torch.equal(a.cpu(), b) for a, b in zip(inputs[:5],
                        (ori_cpu, cmp_cpu, table, cmp_table_cpu, raw)))
        if not (exact and guard and read_only):
            raise AssertionError(f"32行KV边界失败：{name}, exact={exact}, guard={guard}, readonly={read_only}")
        checks.append({"case": name, "exact": exact, "guard": guard, "readonly": read_only})
    report = {"passed": True, "actual_candidate_fragment_reused": True,
              "same_graph_all_cases": True, "checks": checks, "source": str(path)}
    (Path(output) / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
