# SPDX-License-Identifier: Apache-2.0
"""CPU枚举主Q投影/后处理分片；尚未生成新依赖，不用于设备执行。"""

import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root.parent / "csa_static_spmd_feasibility_20261015/generated/core_bodies_ir.py"
    tree = ast.parse(source.read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    checks = {
        "qproj_matmul": [
            "qproj_worker_inline8514__ssa_v0 + qproj_round_inline8515__idx_v0 * 20",
            "qproj_n_idx_inline8522__ssa_v0 * 256",
            "pl.range(0, qproj_t_matmul_inline8513__ssa_v0, 128, init_values=(q_proj_i32_inline7_inline3585__iter_v1,))",
        ],
        "qproj_dequant_rms_nope_rope": [
            "96 <= tile_rows_inline11_inline3587__ssa_v0 and tile_rows_inline11_inline3587__ssa_v0 % 96 == 0",
            "pl.range(dq_worker_inline8595__ssa_v0, (tile_rows_inline11_inline3587__ssa_v0 + 7) // 8 * (64 // dq_head_tile_inline8559__phi_v4), 48)",
            "dq_work_inline8592__idx_v0 % (64 // dq_head_tile_inline8559__phi_v4) * dq_head_tile_inline8559__phi_v4",
            "dq_work_inline8592__idx_v0 // (64 // dq_head_tile_inline8559__phi_v4) * 8",
            "h_inline8564__ssa_v0 * 512",
        ],
    }
    for name, expressions in checks.items():
        available = {ast.unparse(n) for n in ast.walk(functions[name])}
        if not set(expressions) <= available:
            raise ValueError(f"原分片公式改变：{name}")
    alternatives = []
    for tile in (1, 2, 4, 8, 16):
        writers, consumers, counts = {}, [], []
        for worker in range(48):
            needed, units, count = set(), [], 0
            for unit in range(worker, 12 * (64 // tile), 48):
                row = unit // (64 // tile)
                first_head = unit % (64 // tile) * tile
                for head in range(first_head, first_head + tile):
                    if (row, head) in writers:
                        raise ValueError("重复写同一行/head")
                    writers[row, head] = worker
                    count += 1
                    needed.update((head * 2 + part) % 20 for part in range(2))
                units.append({"row_block": row, "head_begin": first_head})
            consumers.append({"worker": worker, "producers": sorted(needed), "head_row_units": units})
            counts.append(count)
        if len(writers) != 12 * 64:
            raise ValueError("没有覆盖全部行/head")
        qk_parents = [sorted({writers[query // 8, head]
                             for query in range(core, 96, 24) for head in range(64)})
                      for core in range(24)]
        alternatives.append({"head_tile": tile,
                             "producer_counts": sorted({len(x["producers"]) for x in consumers}),
                             "head_iterations_per_worker": sorted(set(counts)),
                             "qk_dequant_parent_counts": sorted({len(x) for x in qk_parents}),
                             "consumer_ownership": consumers, "qk_dequant_parents": qk_parents})
    result = {"source": str(source), "source_contracts": checks, "tokens": 96,
              "existing_head_tile": 16, "projection_column_tiles": 128, "projection_workers": 20,
              "dequant_workers": 48, "alternatives": alternatives,
              "math_or_dependencies_changed": False, "device_executed": False,
              "scope": "仅源码分片公式与行/head覆盖枚举；不是完成的同步合同、调度候选或性能预测"}
    (root / "qproj_dequant_ownership_cpu_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items()
                       if k not in ("consumer_ownership", "qk_dequant_parents")}
                      for row in alternatives], ensure_ascii=False))


if __name__ == "__main__":
    main()
