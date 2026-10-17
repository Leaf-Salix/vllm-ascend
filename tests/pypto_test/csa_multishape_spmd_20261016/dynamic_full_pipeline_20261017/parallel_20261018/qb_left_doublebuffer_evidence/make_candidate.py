#!/usr/bin/env python3
"""从 d56 已采用33文件冻结包生成仅主QB Left显式双槽候选。

只使用标准库；不导入设备库、不编译、不修改输入源。完整副本保存全部33个
生产Python文件，只替换 _q_b_nz_stage 的函数体；其余32文件逐字比较。
保留动态T/m_rows、Native NZ、原四个K256点积、128列块轮转与GM输出。
候选输出目录必须不存在，避免覆盖已被任务使用的源码。
"""

import argparse
import ast
import copy
import difflib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BODY = '''    """NZ Q_B：仅后续K256 pair显式保留两份Left；跨核协议仍由caller负责。

    初始K0与最后K768使用单块；中间K256/K512先形成两份Left再提交MMA，
    用重叠活跃期表达双槽需求，不使用固定地址或全局reserve。Right仍单槽：
    后一Right只在前一MMA提交后才构造，硬件覆盖等待由既有依赖lowering生成。
    """
    qproj_worker = pl.max(worker_id, 0)
    qproj_t_matmul = (tokens + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE * QPROJ_PIPE_M_TILE
    # 原列块owner及M128动态尾行不变；本候选不按当前batch冻结循环。
    for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
        qproj_n_idx = qproj_worker + qproj_round * WORKERS
        w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
        for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
            m_rows = pl.min(QPROJ_PIPE_M_TILE, tokens - t0)
            qr_first_mat = pl.load(
                qr_i8_matmul, [t0, 0], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                target_memory=pl.MemorySpace.Mat, valid_shape=[m_rows, QPROJ_PIPE_K_TILE],
            )
            qr_first_left = pl.tile.move(qr_first_mat, target_memory=pl.MemorySpace.Left)
            wq_first_mat = pl.load(
                wq_b, [0, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                target_memory=pl.MemorySpace.Mat,
            )
            wq_first_right = pl.tile.move(wq_first_mat, target_memory=pl.MemorySpace.Right)
            # INT8×INT8的tile.matmul建立INT32 Acc，和原首次高层matmul同一K0点积。
            col_acc = pl.tile.matmul(qr_first_left, wq_first_right)
            # 原后续pipeline的完整pair是K256/K512；先造两份Left，使二者在第一
            # 次MMA之前同时存活。两份物理[128,256]INT8恰为64KiB，尾M仍读m_rows。
            # 此range只描述模型K范围，实际T仍动态；不改变INT32累加的四段顺序。
            for k0 in pl.range(QPROJ_PIPE_K_TILE, Q_LORA - QPROJ_PIPE_K_TILE, 2 * QPROJ_PIPE_K_TILE):
                qr_pair0_mat = pl.load(
                    qr_i8_matmul, [t0, k0], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat, valid_shape=[m_rows, QPROJ_PIPE_K_TILE],
                )
                qr_pair0_left = pl.tile.move(qr_pair0_mat, target_memory=pl.MemorySpace.Left)
                qr_pair1_mat = pl.load(
                    qr_i8_matmul, [t0, k0 + QPROJ_PIPE_K_TILE], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat, valid_shape=[m_rows, QPROJ_PIPE_K_TILE],
                )
                qr_pair1_left = pl.tile.move(qr_pair1_mat, target_memory=pl.MemorySpace.Left)
                wq_pair0_mat = pl.load(
                    wq_b, [k0, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                wq_pair0_right = pl.tile.move(wq_pair0_mat, target_memory=pl.MemorySpace.Right)
                col_acc = pl.tile.matmul_acc(col_acc, qr_pair0_left, wq_pair0_right)
                # 单份Right物理64KiB，不能在消费上一份前请求第二份；这里不把
                # 恢复Left双槽伪装成整套操作数双缓冲，也不取消任何覆盖依赖。
                wq_pair1_mat = pl.load(
                    wq_b, [k0 + QPROJ_PIPE_K_TILE, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                wq_pair1_right = pl.tile.move(wq_pair1_mat, target_memory=pl.MemorySpace.Right)
                col_acc = pl.tile.matmul_acc(col_acc, qr_pair1_left, wq_pair1_right)
            # 剩余K768单块不借机拆K128，不额外驻留整条QR或预取相邻N列。
            k_tail = Q_LORA - QPROJ_PIPE_K_TILE
            qr_tail_mat = pl.load(
                qr_i8_matmul, [t0, k_tail], [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                target_memory=pl.MemorySpace.Mat, valid_shape=[m_rows, QPROJ_PIPE_K_TILE],
            )
            qr_tail_left = pl.tile.move(qr_tail_mat, target_memory=pl.MemorySpace.Left)
            wq_tail_mat = pl.load(
                wq_b, [k_tail, w_col0], [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE],
                target_memory=pl.MemorySpace.Mat,
            )
            wq_tail_right = pl.tile.move(wq_tail_mat, target_memory=pl.MemorySpace.Right)
            col_acc = pl.tile.matmul_acc(col_acc, qr_tail_left, wq_tail_right)
            # Tile的动态有效M来自原QR valid_shape；外部仍绑定同一pad128 GM容量。
            pl.store(col_acc, [t0, w_col0], q_proj_i32)
    return q_proj_i32
'''


def without_target(tree):
    tree = copy.deepcopy(tree)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')]
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--formal', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    files = sorted(args.baseline.glob('*.py'))
    assert len(files) == 33
    assert all((args.formal / f.name).read_bytes() == f.read_bytes() for f in files)
    before, candidate = args.output / 'source_before', args.output / 'source_candidate'
    before.mkdir(exist_ok=False)
    candidate.mkdir(exist_ok=False)
    for f in files:
        shutil.copy2(f, before / f.name)
        shutil.copy2(f, candidate / f.name)
    name = 'query_projection_stages.py'
    original = (before / name).read_text()
    old_tree = ast.parse(original)
    function = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')
    lines = original.splitlines(keepends=True)
    start = function.body[0].lineno - 1
    replacement = ''.join(lines[:start]) + BODY + ''.join(lines[function.end_lineno:])
    new_tree = ast.parse(replacement)
    new_function = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == function.name)
    assert ast.dump(function.args) == ast.dump(new_function.args)
    assert without_target(old_tree) == without_target(new_tree)
    (candidate / name).write_text(replacement)
    unchanged = [f.name for f in files if f.name != name and (candidate / f.name).read_bytes() == (before / f.name).read_bytes()]
    assert len(unchanged) == 32
    # 数学K顺序与动态尾行覆盖只做有限整数审计；不代替真实Tile valid_shape lowering。
    k_chain = [0] + [k for begin in range(256, 1024-256, 512) for k in (begin, begin+256)] + [768]
    assert k_chain == [0,256,512,768]
    tail_cases = []
    for t in (1,24,48,96,128,129,144,168,192,193,256):
        rows=[(base,min(128,t-base)) for base in range(0,((t+127)//128)*128,128)]
        assert sum(count for _,count in rows)==t
        tail_cases.append(dict(tokens=t,valid_rows=rows))
    stamp = datetime.now(timezone.utc).isoformat()
    audit=dict(baseline=str(args.baseline), formal=str(args.formal),baseline_commit='d56c9eb6',
               frozen_utc=stamp, files=33, changed_files=[name],unchanged_files=unchanged,
               only_target_function_ast_changed=True, signature_unchanged=True, root_abi_unchanged=True,
               original_k_order=k_chain,tail_cases=tail_cases,
               theoretical_peak_bytes=dict(Left=65536,Right=65536,Acc=131072),
               fixed_address_or_global_reserve=False,full_backend_passed=False,device_executed=False)
    (args.output/'source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (args.output/'candidate.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),replacement.splitlines(True),fromfile='d56/'+name,tofile='candidate/'+name)))
    (args.output/'controls.json').write_text(json.dumps(dict(OA_COLS=128,OB_ROWS=96),indent=2)+'\n')
    print(json.dumps(dict(source=str(candidate),frozen_utc=stamp,changed_files=[name],unchanged_files=32),ensure_ascii=False))


if __name__ == '__main__':
    main()
