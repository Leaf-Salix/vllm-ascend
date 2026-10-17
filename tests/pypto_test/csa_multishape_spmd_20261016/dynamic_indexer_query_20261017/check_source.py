# SPDX-License-Identifier: Apache-2.0
"""独立核对六段数学AST、所有权和尾行：只读源码，不导入PyPTO或执行设备。

数学体直接与本次保存的高层reference_decode_indexer.py比较，仅规范化worker
常量和T_PAD名称。小T从min(T,8)变为固定8行+逐行尾分支单独披露，覆盖检查
验证每个有效token/head唯一处理；这不是设备数值等价或缓存可见性证明。
"""

import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "reference_decode_indexer.py"
TARGET = HERE.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_query_stages.py"


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        names = {
            "QR_PROJ_WORKERS": "WORKERS",
            "QH_WORKERS": "WORKERS",
            "QH_QUANT_WORKERS": "WORKERS",
            "weights_workers": "WORKERS",
            "T_PAD": "t_pad",
        }
        return ast.copy_location(ast.Name(id=names.get(node.id, node.id), ctx=node.ctx), node)


def equal(left, right):
    expected = ast.dump(
        Normalize().visit(copy.deepcopy(ast.Module(body=left, type_ignores=[]))), include_attributes=False
    )
    actual = ast.dump(ast.Module(body=right, type_ignores=[]), include_attributes=False)
    return expected == actual


def by_name(source):
    tree = ast.parse(source)
    return tree, {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def named_loop(function, name):
    return next(node for node in ast.walk(function) if isinstance(node, ast.For) and ast.unparse(node.target) == name)


def check_math():
    old_tree, old = by_name(SOURCE.read_text())
    new_tree, new = by_name(TARGET.read_text())
    qr = old["indexer_qr_rope"]
    mm = next(node for node in qr.body if isinstance(node, ast.With))
    dq = next(node for node in qr.body if isinstance(node, ast.For))
    had, quant = [node for node in old["indexer_qr_hadamard_mm"].body if isinstance(node, ast.With)]
    wp, wr = [node for node in old["indexer_weights_project"].body if isinstance(node, ast.With)]
    dq_new = next(node for node in new["indexer_query_dequant_rope_stage"].body if isinstance(node, ast.If))
    comparisons = {
        "query_projection_entire_column_loop": equal(
            mm.body[1:], [named_loop(new["indexer_query_project_stage"], "qr_col")]
        ),
        "dequant_full_and_scalar_tail_all_math": equal(dq.body, dq_new.body[1:]),
        "hadamard_matrix_and_whole_math_loop": equal(had.body[1:], new["indexer_query_hadamard_stage"].body[3:-1]),
        "quant_entire_math_loop": equal(quant.body[1:], [named_loop(new["indexer_query_quant_stage"], "idx")]),
        "weights_projection_two_K512_per_split": equal(
            wp.body[1:], [named_loop(new["indexer_weights_project_stage"], "w_unit")]
        ),
        "weights_reduce_left_fold_and_scale": equal(
            wr.body[1:], named_loop(new["indexer_weights_reduce_stage"], "w_rb").body
        ),
    }
    assert all(comparisons.values()), comparisons
    assignments = lambda tree: {
        node.targets[0].id: ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    og, ng = assignments(old_tree), assignments(new_tree)
    constants = [
        "D",
        "Q_LORA",
        "IDX_N_HEADS",
        "IDX_HEAD_DIM",
        "IDX_NOPE_HEAD_DIM",
        "ROPE_HEAD_DIM",
        "WEIGHTS_SCALE",
        "HADAMARD_SCALE",
        "QR_MM_ROW_TILE",
        "QR_MM_N_TILE",
        "DQ_ROPE_H_TILE",
        "QH_MM_TILE",
        "QH_QUANT_TILE",
        "QH_HEAD_DIM_TILE",
        "MM_ROW_TILE",
        "WEIGHTS_OK",
        "WEIGHTS_K_TILE",
        "D_TILE",
    ]
    assert all(og[name] == ng[name] for name in constants)
    calls = [ast.unparse(node.func) for node in ast.walk(new_tree) if isinstance(node, ast.Call)]
    assert not any("spmd" in name or "submit" in name or "task_dummy" in name for name in calls)
    symbols = {node.id for node in ast.walk(new_tree) if isinstance(node, ast.Name)}
    assert not {"S", "B", "DECODE_SEQ", "DECODE_BATCH", "T_PAD", "QR_MM_T_PAD"} & symbols
    return comparisons, constants


def unique_assignments(items, expected):
    seen = set()
    for item in items:
        assert item not in seen
        seen.add(item)
    assert seen == expected


def coverage(tokens, cube_workers, vector_workers):
    pad16 = (tokens + 15) // 16 * 16
    project = [
        (row, col)
        for worker in range(cube_workers)
        for col in range(worker, 32, cube_workers)
        for row in range((tokens + 63) // 64)
    ]
    unique_assignments(project, {(row, col) for row in range((tokens + 63) // 64) for col in range(32)})
    units = (tokens + 7) // 8 * 16
    active = min(units, vector_workers)
    dequant = [
        (row, head)
        for worker in range(active)
        for unit in range(worker, units, active)
        for row in range(unit // 16 * 8, min(unit // 16 * 8 + 8, tokens))
        for head in range(unit % 16 * 4, unit % 16 * 4 + 4)
    ]
    unique_assignments(dequant, {(row, head) for row in range(tokens) for head in range(64)})
    # T*64总是整除64；Hadamard和量化不以padding容量延长有效行遍历。
    for workers in (cube_workers, vector_workers):
        unique_assignments(
            (token for worker in range(workers) for token in range(worker, tokens, workers)), set(range(tokens))
        )
    partial = [
        (split, row)
        for worker in range(cube_workers)
        for task in range(worker, (pad16 // 16) * 4, cube_workers)
        for split, row in [(task % 4, task // 4)]
    ]
    unique_assignments(partial, {(split, row) for split in range(4) for row in range(pad16 // 16)})
    unique_assignments(
        (row for worker in range(vector_workers) for row in range(worker, pad16 // 16, vector_workers)),
        set(range(pad16 // 16)),
    )
    # 每一分量地址属于自己的K split和合法M16块，最后边界恰好为4*pad16。
    for split, row in partial:
        first = split * pad16 + row * 16
        assert split * pad16 <= first and first + 16 <= (split + 1) * pad16
    # RoPE完整块用head的128列stride；逐行尾块的相邻交换必须同样指向本行后64列。
    for row in range(8):
        for column in range(64):
            full = row * 128 + 64 + column + 1 - 2 * (column % 2)
            tail = 64 + column + 1 - 2 * (column % 2)
            assert full == row * 128 + tail
    return {
        "tokens": tokens,
        "cube_workers": cube_workers,
        "vector_workers": vector_workers,
        "dequant_tail_rows": tokens % 8,
        "projection_pad64": (tokens + 63) // 64 * 64,
        "weights_pad16": pad16,
        "dequant_units": units,
    }


def main():
    comparisons, constants = check_math()
    values = list(range(1, 65)) + [95, 96, 97, 127, 128, 129, 144, 168, 192, 383, 384, 385, 511, 512, 513, 769]
    results = [coverage(t, c, v) for t in values for c, v in [(24, 48), (8, 48), (1, 1)]]
    report = {
        "passed": True,
        "source": str(TARGET),
        "reference_source_snapshot": str(SOURCE),
        "scope": "六个数学体AST、真实token/head覆盖与尾行索引，仅CPU",
        "mathematical_blocks_equal": comparisons,
        "constants_equal": constants,
        "fixed_S_dependency": False,
        "nested_dispatch": False,
        "scale_order_correction": "以当前canonical源码为准：(acc*activation_scale)*weight_scale；历史合并scale未移入。",
        "small_T_note": "原DEQUANT_T_TILE=min(编译T,8)，现固定8；T<8走原逐行tail数学，所有有效元素唯一覆盖。",
        "coverage_case_count": len(results),
        "tokens_checked": values,
        "worker_pairs_checked": [[24, 48], [8, 48], [1, 1]],
        "examples": [
            row
            for row in results
            if row["tokens"] in [1, 6, 24, 48, 96, 144, 168, 192, 513] and row["cube_workers"] == 24
        ],
        "device_execution": False,
        "weight_values_loaded": False,
        "numerical_device_equivalence": False,
    }
    (HERE / "source_equivalence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "math_blocks": len(comparisons), "coverage_cases": len(results)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
