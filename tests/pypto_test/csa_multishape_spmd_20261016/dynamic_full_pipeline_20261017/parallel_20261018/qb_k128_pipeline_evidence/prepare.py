# SPDX-License-Identifier: Apache-2.0
"""准备动态 Q_B 的 K128 双 L0 槽候选，保留完整正式动态入口。

原 NZ N256/K256 的单个 Right 已占满64KiB，stage2只能重叠部分L1供数。
本候选维持N256和原owner，K分成八个128片，显式在消费当前片前准备下一片。
每个Right32KiB、Left16KiB，两个槽分别合计64/32KiB；INT8整数累加无
浮点重排边界，最大绝对和1024*128*128远小于INT32范围。实际物理双槽和
局部事件仍须检查生成物，不能仅凭容量计算认为流水已成立。
"""

import ast
import difflib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SOURCE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"

BODY = '''    """NZ Q_B：K128双Left/Right；真实T尾行与N256 owner保持。"""
    qproj_worker = pl.max(worker_id, 0)
    qproj_t_matmul = (tokens + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE * QPROJ_PIPE_M_TILE
    for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
        qproj_n_idx = qproj_worker + qproj_round * WORKERS
        w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
        for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
            m_rows = pl.min(QPROJ_PIPE_M_TILE, tokens - t0)
            # 首片与下一片同时存活，强制分配不同L0地址；实际地址仍须审阅。
            qr0_mat = pl.load(qr_i8_matmul, [t0, 0],
                [QPROJ_PIPE_M_TILE, QPROJ_PREFETCH_K_TILE],
                valid_shape=[m_rows, QPROJ_PREFETCH_K_TILE], target_memory=pl.MemorySpace.Mat)
            b0_mat = pl.load(wq_b, [0, w_col0],
                [QPROJ_PREFETCH_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat)
            qr0 = pl.tile.move(qr0_mat, target_memory=pl.MemorySpace.Left)
            b0 = pl.tile.move(b0_mat, target_memory=pl.MemorySpace.Right)
            qr1_mat = pl.load(qr_i8_matmul, [t0, QPROJ_PREFETCH_K_TILE],
                [QPROJ_PIPE_M_TILE, QPROJ_PREFETCH_K_TILE],
                valid_shape=[m_rows, QPROJ_PREFETCH_K_TILE], target_memory=pl.MemorySpace.Mat)
            b1_mat = pl.load(wq_b, [QPROJ_PREFETCH_K_TILE, w_col0],
                [QPROJ_PREFETCH_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat)
            qr_current = pl.tile.move(qr1_mat, target_memory=pl.MemorySpace.Left)
            b_current = pl.tile.move(b1_mat, target_memory=pl.MemorySpace.Right)
            col_acc = pl.tile.matmul(qr0, b0)
            # 只展开模型固定的K=1024，不展开T/B；每轮的下一片先发射，
            # 当前矩阵仅依赖当前槽。重用旧槽的M→MTE1依赖交给编译器保留。
            # 刻意不套pipeline，避免自动IO前移让三个Right同时存活。
            for part in pl.unroll(1, Q_LORA // QPROJ_PREFETCH_K_TILE - 1):
                next_k = (part + 1) * QPROJ_PREFETCH_K_TILE
                qr_next_mat = pl.load(qr_i8_matmul, [t0, next_k],
                    [QPROJ_PIPE_M_TILE, QPROJ_PREFETCH_K_TILE],
                    valid_shape=[m_rows, QPROJ_PREFETCH_K_TILE], target_memory=pl.MemorySpace.Mat)
                b_next_mat = pl.load(wq_b, [next_k, w_col0],
                    [QPROJ_PREFETCH_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat)
                qr_next = pl.tile.move(qr_next_mat, target_memory=pl.MemorySpace.Left)
                b_next = pl.tile.move(b_next_mat, target_memory=pl.MemorySpace.Right)
                col_acc = pl.tile.matmul_acc(col_acc, qr_current, b_current)
                qr_current = qr_next
                b_current = b_next
            # 最后一个K128只消费，不再预取；避免两个独立if之间传递Tile定义。
            col_acc = pl.tile.matmul_acc(col_acc, qr_current, b_current)
            # 动态valid行由Left传递至Acc；仅发布本轮真实T，不能写padding。
            pl.store(col_acc, [t0, w_col0], q_proj_i32)
    return q_proj_i32
'''


def main():
    reference, candidate = HERE / "source_reference", HERE / "source_candidate"
    reference.mkdir(exist_ok=False)
    candidate.mkdir(exist_ok=False)
    for path in SOURCE.glob("*.py"):
        (reference / path.name).write_bytes(path.read_bytes())
        (candidate / path.name).write_bytes(path.read_bytes())
    file = candidate / "query_projection_stages.py"
    original = file.read_text()
    tree = ast.parse(original)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_q_b_nz_stage")
    lines = original.splitlines(keepends=True)
    # 参数签名、其它函数均保留；只替换这个动态helper的函数体。
    modified = ''.join(lines[:function.body[0].lineno - 1]) + BODY + ''.join(lines[function.end_lineno:])
    modified = modified.replace('QPROJ_PIPE_K_TILE = 256\n', 'QPROJ_PIPE_K_TILE = 256\nQPROJ_PREFETCH_K_TILE = 128\n')
    modified = modified.replace('NZ 保留 M128/N256/K256，ND 保留 M64/N512/K128',
                                'NZ 候选使用 M128/N256/K128双槽，ND 保留 M64/N512/K128')
    ast.parse(modified)
    file.write_text(modified)
    (HERE / "candidate.diff").write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile='formal/query_projection_stages.py', tofile='candidate/query_projection_stages.py')))
    report = dict(source=str(SOURCE), changed=['query_projection_stages.py'],
                  inputs_dynamic=True, n_owner_unchanged=True, integer_accumulation_bound=1024*128*128,
                  mathematical_k_tiles=8, expected_live_bytes=dict(Left=32768,Right=65536,Acc=131072),
                  actual_addresses_unchecked=True, device_validated=False)
    (HERE / "PREPARATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
