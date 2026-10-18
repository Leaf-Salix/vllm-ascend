# SPDX-License-Identifier: Apache-2.0
"""解释正式页计划/链表提交源码，校验跨worker页互斥、边界和全部未写字节。

仅为CPU地址/所有权oracle，不模拟DCache或设备同步；真机共享页probe另行执行。
"""

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "scale_page_owners_v1/oracle.json")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "group_cpu_oracle", HERE.parent / "dynamic_indexer_group_20261017/check_oracle.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ns, reads, stores = mod.helpers()
    ns["pl"].INT8 = np.int8
    ns["pl"].tile.load = ns["pl"].load
    ns["pl"].tile.store = ns["pl"].store
    ns["pl"].tile.write = ns["pl"].write
    ns["pl"].tile.reinterpret_view = lambda tensor, dtype: tensor.view(dtype)

    class PlainExplicitWhile(mod.Plain):
        """把显式DSL单值回边等价还原为Python while，仅用于CPU算法oracle。"""

        def visit_For(self, node):
            if not isinstance(node.iter, ast.Call) or ast.unparse(node.iter.func) != "pl.while_":
                return self.generic_visit(node)
            assert len(node.target.elts) == 1
            name = node.target.elts[0].id
            initial = next(k.value.elts[0] for k in node.iter.keywords if k.arg == "init_values")
            condition = node.body[0].value
            assert ast.unparse(condition.func) == "pl.cond"
            final = node.body[-1].value
            assert ast.unparse(final.func) == "pl.yield_" and len(final.args) == 1
            update = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(final.args[0]))
            return [
                ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(initial)),
                ast.While(test=condition.args[0], body=[*node.body[1:-1], update], orelse=[]),
            ]

    tree = ast.fix_missing_locations(
        PlainExplicitWhile().visit(ast.parse((PKG / "indexer_scale_plan_stages.py").read_text()))
    )
    exec(compile(tree, "formal_scale_page_plan", "exec"), ns)
    rng = np.random.default_rng(20261017)
    reports = []
    for case in range(36):
        widths = rng.integers(1, 18, size=5)
        q = np.array([1, *np.cumsum(widths) + 1], np.int32)
        tokens = int(q[-1]) + 2
        histories = rng.integers(0, 131072, size=5)
        lengths = (histories + widths).astype(np.int32)
        if case % 3 == 0:
            lengths[2] = 0
        if case % 9 == 0:
            lengths.fill(0)
        positions = np.full(tokens, -1, np.int64)
        compact_offsets = np.full(5, -99, np.int32)
        closed = []
        for request, (begin, end) in enumerate(zip(q[:-1], q[1:])):
            if lengths[request] <= 0:
                continue
            history = int(lengths[request]) - int(end - begin)
            compact_offsets[request] = len(closed) - history // 4 - 1
            for token in range(int(begin), int(end)):
                position = history + token - begin
                positions[token] = position
                if (position + 1) % 4 == 0:
                    closed.append(token)
        if case % 2 and closed:
            positions[closed[0]] = -17
        scales = rng.uniform(1e-6, 0.04, (tokens, 1)).astype(np.float32)
        slots = np.full((tokens, 2), -1, np.int32)
        for row in range(len(closed)):
            slots[row] = (row // 32 + 3, row % 32) if case % 2 else (row % 64, row // 64)
        if case % 4 == 1 and len(closed) >= 4:
            slots[1] = (-1, 0)
            slots[2] = (64, 0)
            slots[3] = (0, 32)
        initial = rng.integers(-128, 127, (64, 4160), dtype=np.int8)
        expected = initial.copy()
        for row, token in enumerate(closed):
            page, offset = map(int, slots[row])
            if positions[token] >= 0 and 0 <= page < 64 and 0 <= offset < 32:
                expected[page, 4096 + offset * 2 : 4098 + offset * 2] = scales[row].astype(np.float16).view(np.int8)
        for workers in (1, 8, 48):
            group_end = 16 + 4 * tokens
            base = (group_end + 15) // 16 * 16
            guard = np.full(base + workers + 4 * tokens + 32, -71, np.int32)
            workspace = guard[16:-16]
            cache = initial.copy()
            ns["indexer_scale_page_plan_stage"](
                q, lengths, positions, compact_offsets, slots, cache, workspace, len(scales), workers
            )
            page_owners = {}
            for worker in reversed(range(workers)):
                stores.clear()
                ns["indexer_scale_page_commit_stage"](scales, workspace, cache, tokens, len(lengths), worker, workers)
                for identity, address, _ in stores:
                    if identity == id(cache):
                        page = address[0]
                        assert page % workers == worker
                        assert page_owners.setdefault(page, worker) == worker
            assert np.array_equal(cache, expected), (case, workers)
            assert np.all(guard[:16] == -71) and np.all(guard[-16:] == -71)
            assert np.all(workspace[:base] == -71), "不得覆盖已有组描述符区域"
            reports.append({"case": case, "workers": workers, "passed": True})
    report = {"passed": True, "cases": reports, "device_execution": False, "cache_coherence_proven": False}
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(reports), "device_execution": False}))


if __name__ == "__main__":
    main()
