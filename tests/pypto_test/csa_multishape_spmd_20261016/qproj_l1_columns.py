# SPDX-License-Identifier: Apache-2.0
"""主Q双N128列块：整列权重驻L1，显式交错两个INT32累加链。

仍由原20个worker负责原128组N256列。每组分为两个N128，QR只在
worker入口加载一次，两份1024×128权重一次加载后分四个K256消费。
INT8点积按升序K累加，INT32不溢出；不会改变输出、根参数或列归属。
期望同时容纳两个Right和两个Acc，必须审核实际物理分配后再测设备。
"""

import ast
import copy
import json
from pathlib import Path


def body():
    lines = [
        "qlc_worker = pl.tile.get_block_idx()",
        "qlc_query = pl.tile.load(qr_i8_matmul_inline3588__ssa_v0, [0, 0], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)",
        "for qlc_round in pl.range((19 - qlc_worker + 128) // 20):",
        "    qlc_col = (qlc_worker + qlc_round * 20) * 256",
        "    qlc_w0 = pl.tile.load(wq_b__ssa_v0, [0, qlc_col], [1024, 128], [1024, 128], target_memory=pl.Mem.Mat)",
        "    qlc_w1 = pl.tile.load(wq_b__ssa_v0, [0, qlc_col + 128], [1024, 128], [1024, 128], target_memory=pl.Mem.Mat)",
        "    qlc_left_0 = pl.tile.extract(qlc_query, 0, 0, [96, 256], target_memory=pl.Mem.Left)",
        "    qlc_r0_0 = pl.tile.extract(qlc_w0, 0, 0, [256, 128], target_memory=pl.Mem.Right)",
    ]
    for k in range(4):
        lines.append(f"    qlc_r1_{k} = pl.tile.extract(qlc_w1, {k * 256}, 0, [256, 128], target_memory=pl.Mem.Right)")
        if k == 0:
            lines.append("    qlc_acc0 = pl.tile.matmul(qlc_left_0, qlc_r0_0)")
        else:
            lines.append(f"    qlc_acc0 = pl.tile.matmul_acc(qlc_acc0, qlc_left_{k}, qlc_r0_{k})")
        if k < 3:
            lines.extend([
                f"    qlc_left_{k+1} = pl.tile.extract(qlc_query, 0, {(k+1)*256}, [96, 256], target_memory=pl.Mem.Left)",
                f"    qlc_r0_{k+1} = pl.tile.extract(qlc_w0, {(k+1)*256}, 0, [256, 128], target_memory=pl.Mem.Right)",
            ])
        if k == 0:
            lines.append("    qlc_acc1 = pl.tile.matmul(qlc_left_0, qlc_r1_0)")
        else:
            lines.append(f"    qlc_acc1 = pl.tile.matmul_acc(qlc_acc1, qlc_left_{k}, qlc_r1_{k})")
    lines.extend([
        "    pl.tile.store(qlc_acc0, [0, qlc_col], q_proj_i32_inline7_inline3585__ssa_v0)",
        "    pl.tile.store(qlc_acc1, [0, qlc_col + 128], q_proj_i32_inline7_inline3585__ssa_v0)",
    ])
    return ast.parse("\n".join(lines)).body


def use_l1_columns(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("双N128主Q先验证T96，其他档应独立选择M与L1容量")
    tree = ast.parse(Path(catalog["logical_source_ir"]).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qproj_matmul")
    text = ast.unparse(fn)
    for contract in ["(19 - qproj_worker_inline8514__ssa_v0 + 128) // 20",
                     "qproj_n_idx_inline8522__ssa_v0 * 256", "pl.pipeline(256, 1024, 256",
                     "qr_i8_matmul_inline3588__ssa_v0", "pl.TensorLayout.NZ"]:
        if contract not in text:
            raise ValueError("冻结主Q合同改变：" + contract)
    tasks = [t for t in schedule["tasks"] if t["kernel_names"][0] == "qproj_matmul"]
    if len(tasks) != 1 or tasks[0]["block_num"] != 20:
        raise ValueError("主Q原逻辑worker数改变")
    fn.body = [fn.body[0], *body(), fn.body[-1]]
    ast.fix_missing_locations(tree)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "logical_qproj_l1_columns.py"
    path.write_text("# Copyright (c) PyPTO Contributors.\n"
                    "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                    + ast.unparse(tree) + "\n")
    coverage = []
    for worker in range(20):
        for round_ in range((19 - worker + 128) // 20):
            col = (worker + round_ * 20) * 256
            coverage.extend(range(col, col + 256))
    assert sorted(coverage) == list(range(32768))
    audit = {"m": 96, "n_per_chain": 128, "k": 256, "accumulator_chains": 2,
             "original_column_ownership_preserved": True, "workers": 20,
             "query_l1_loads_per_worker": 1, "weight_l1_loads_per_original_column_group": 2,
             "k_offsets_per_chain": [0, 256, 512, 768],
             "int32_dot_max_abs_bound": 128 * 128 * 1024,
             "source_live_capacity_bytes": {"mat": 360448, "left": 49152, "right": 65536, "acc": 98304},
             "actual_allocation_checked": False, "device_validated": False}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["logical_source_ir"] = str(path)
    return result, audit
