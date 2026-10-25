# SPDX-License-Identifier: Apache-2.0
"""借用已合入HCA的M192/N128双层流水，独立验证CSA大T的Q_B权重复用。"""

import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

from prepare import replace_once

HELPER = '''# SPDX-License-Identifier: Apache-2.0
"""CSA大T Q_B候选：沿用原20核所有权，仅合并M轮次并使用两级K流水。

参考同仓deepseek_v4_flash_hca_tnd/q_projection.py的已保留M192/N128实现。
实际行数来自device Tensor描述符，不使用B*S；当前调用者限制128<T<=192。
K1024的INT8点积完整累加至INT32，绝对值上界2**24，无INT32溢出。
反量化、RMS、RoPE及生产者完成通知由原调用者保持。
"""

import pypto.language as pl

from .query_projection_stages import (
    H, HEAD_DIM, PAD_ROWS_DYN, QPROJ_ROWS_DYN, Q_LORA, QUANT_WEIGHT_LAYOUT,
)

QB_TALL_M = 192
QB_TALL_N = 128
QB_TALL_L1_K = 512
QB_TALL_L0_K = 128


@pl.jit.inline(auto_scope=False)
def q_b_tall_stage(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[PAD_ROWS_DYN, Q_LORA], pl.INT8],
    q_proj_i32: pl.InOut[pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    worker = pl.max(worker_id, 0)
    n_blocks = H * HEAD_DIM // QB_TALL_N
    for step in pl.range((n_blocks - worker + WORKERS - 1) // WORKERS):
        col = (worker + step * WORKERS) * QB_TALL_N
        for t0 in pl.range(0, tokens, QB_TALL_M):
            count = pl.min(QB_TALL_M, tokens - t0)
            seed_storage = pl.create_tile(
                [QB_TALL_M, QB_TALL_N], dtype=pl.INT32,
                target_memory=pl.MemorySpace.Acc, compact=True,
            )
            seed = pl.tile.set_validshape(seed_storage, count, QB_TALL_N)
            for outer, (outer_acc,) in pl.pipeline(
                0, Q_LORA // QB_TALL_L1_K, stage=2, init_values=(seed,),
            ):
                k_base = outer * QB_TALL_L1_K
                a_l1 = pl.load(
                    qr_i8_matmul, [t0, k_base], [QB_TALL_M, QB_TALL_L1_K],
                    valid_shape=[count, QB_TALL_L1_K], target_memory=pl.MemorySpace.Mat,
                )
                b_l1 = pl.load(
                    wq_b, [k_base, col], [QB_TALL_L1_K, QB_TALL_N],
                    target_memory=pl.MemorySpace.Mat,
                )
                for inner, (inner_acc,) in pl.pipeline(
                    0, QB_TALL_L1_K // QB_TALL_L0_K, stage=2, init_values=(outer_acc,),
                ):
                    k_inner = inner * QB_TALL_L0_K
                    a_l0 = pl.tile.extract(
                        a_l1, 0, k_inner, [QB_TALL_M, QB_TALL_L0_K],
                        target_memory=pl.MemorySpace.Left,
                    )
                    b_l0 = pl.tile.extract(
                        b_l1, k_inner, 0, [QB_TALL_L0_K, QB_TALL_N],
                        target_memory=pl.MemorySpace.Right,
                    )
                    updated = pl.tile.matmul_acc(
                        inner_acc, a_l0, b_l0, init_cond=(outer == 0 and inner == 0),
                    )
                    inner_done = pl.yield_(updated)
                outer_done = pl.yield_(inner_done)
            q_proj_i32 = pl.store(outer_done, [t0, col], q_proj_i32)
    return q_proj_i32
'''


def main():
    here = Path(__file__).resolve().parent
    parent = here / "frozen/pypto/deepseek_v4_flash_csa_spmd_tnd"
    target = here / "candidates/qb_tall_v1"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "qb_tall.py").write_text(HELPER)
    path = target / "projection_overlap_stages.py"
    before = path.read_text()
    source = replace_once(
        before, "import pypto.language as pl\n", "import pypto.language as pl\n\nfrom .qb_tall import q_b_tall_stage\n"
    )
    old = "        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)\n"
    new = """        if JOINT_BF16_PREFIX and tokens > 128 and tokens <= 192:
            q_proj = q_b_tall_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS)
        else:
            q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS, QB_WIDE_N)
"""
    source = replace_once(source, old, new)
    ast.parse(HELPER)
    ast.parse(source)
    path.write_text(source)
    (here / "qb_tall_v1.patch").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                source.splitlines(True),
                fromfile="a/projection_overlap_stages.py",
                tofile="b/projection_overlap_stages.py",
            )
        )
    )
    report = {
        "source": str(target),
        "parent": "frozen_current_internal_workspace",
        "reference": "vllm_ascend/ops/pypto/deepseek_v4_flash_hca_tnd/q_projection.py",
        "mechanism": "在长G6既有joint前段区间，M192覆盖B24实际144行，Wq_B遍历由两轮降为一轮",
        "tiles": {"M": 192, "N": 128, "L1_K": 512, "L0_K": 128},
        "preserved": [
            "原20个Q_B producer及完成通知",
            "原Q_R输入与INT32输出",
            "完整INT8 K1024点积",
            "反量化/RMS/RoPE",
            "动态有效行及其余分支",
        ],
        "tradeoff": "N256变N128，列任务数翻倍；减少权重重复读取不保证完整L1更快",
        "mathematical_int32_abs_bound": 1024 * 128 * 128,
        "independent_of_pv_candidate": True,
        "status": "CPU_gate_pending",
        "device_execution": False,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.glob("*.py"))},
    }
    (here / "qb_tall_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
