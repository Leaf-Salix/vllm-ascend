# SPDX-License-Identifier: Apache-2.0
"""主Q显式L0预取：N256/K128，QR在L1保留到本worker所有列完成。

先构造下一块Left/Right，再提交当前块matmul，使两块有重叠活跃期。
T96专用M96；两个Right各32KiB，两个Left各12KiB，Acc96KiB。
K1024仍按升序累加INT8点积，INT32不会溢出；列分工仍128块/20worker。
"""

import ast
import copy
import json
from pathlib import Path


def explicit_body():
    lines = [
        "qpf_worker = pl.tile.get_block_idx()",
        "qpf_query = pl.tile.load(qr_i8_matmul_inline3588__ssa_v0, [0, 0], [96, 1024], [96, 1024], target_memory=pl.Mem.Mat)",
        "for qpf_round in pl.range((19 - qpf_worker + 128) // 20):",
        "    qpf_col = (qpf_worker + qpf_round * 20) * 256",
    ]

    def load(k):
        lines.extend([
            f"    qpf_left_{k} = pl.tile.extract(qpf_query, 0, {k * 128}, [96, 128], target_memory=pl.Mem.Left)",
            f"    qpf_weight_{k} = pl.tile.load(wq_b__ssa_v0, [{k * 128}, qpf_col], "
            "[128, 256], [128, 256], target_memory=pl.Mem.Mat)",
            f"    qpf_right_{k} = pl.tile.move(qpf_weight_{k}, target_memory=pl.Mem.Right)",
        ])

    load(0)
    for k in range(8):
        if k != 7:
            load(k + 1)
        if k == 0:
            lines.append("    qpf_acc = pl.tile.matmul(qpf_left_0, qpf_right_0)")
        else:
            lines.append(f"    qpf_acc = pl.tile.matmul_acc(qpf_acc, qpf_left_{k}, qpf_right_{k})")
    lines.append("    pl.tile.store(qpf_acc, [0, qpf_col], q_proj_i32_inline7_inline3585__ssa_v0)")
    return ast.parse("\n".join(lines)).body


def use_prefetch(schedule, catalog, output):
    if schedule["root_arguments"]["x_hc"]["shape"][0] != 96:
        raise ValueError("显式主Q L0预取先限T96，不把固定M96套到其他档")
    tree = ast.parse(Path(catalog["logical_source_ir"]).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "qproj_matmul")
    text = ast.unparse(fn)
    for contract in ["(19 - qproj_worker_inline8514__ssa_v0 + 128) // 20",
                     "qproj_n_idx_inline8522__ssa_v0 * 256", "pl.pipeline(256, 1024, 256",
                     "qr_i8_matmul_inline3588__ssa_v0", "pl.TensorLayout.NZ"]:
        if contract not in text:
            raise ValueError("冻结Q投影契约改变：" + contract)
    tasks = [t for t in schedule["tasks"] if t["kernel_names"][0] == "qproj_matmul"]
    if len(tasks) != 1 or tasks[0]["block_num"] != 20:
        raise ValueError("本候选保留原20个逻辑worker")
    fn.body = [fn.body[0], *explicit_body(), fn.body[-1]]
    ast.fix_missing_locations(tree)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "logical_qproj_l0_prefetch.py"
    source.write_text("# Copyright (c) PyPTO Contributors.\n"
                      "# CANN Open Software License Agreement Version 2.0; see repository LICENSE.\n"
                      + ast.unparse(tree) + "\n")
    audit = {"m": 96, "n": 256, "old_k_tile": 256, "new_k_tile": 128,
             "k_order": list(range(0, 1024, 128)), "workers": 20, "column_blocks": 128,
             "root_and_gm_layout_unchanged": True, "mathematical_dependencies_unchanged": True,
             "int8_dot_product_max_abs_bound": 128 * 128 * 1024,
             "int32_overflow_possible": False,
             "query_l1_loaded_once_per_worker": True,
             "theoretical_live_bytes": {"left": 24576, "right": 65536, "acc": 98304},
             "actual_double_buffer_verified": False, "device_exact_passed": False}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    result = copy.deepcopy(catalog)
    result["logical_source_ir"] = str(source)
    return result, audit
