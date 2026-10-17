"""从正式66882634复制33源，以默认False的constexpr选择Q_B双M区间B常驻。"""

import ast
import difflib
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/data/pyptouser/qinchuanyu/pto-eager')
MAIN = ROOT/'vllm-ascend-dsv4-pto-0251rc1'
REL = Path('vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd')
SOURCE = MAIN/REL


def main():
    reference, candidate = HERE/'source_reference', HERE/'source_candidate'
    assert not reference.exists() and not candidate.exists(), '输出源必须是全新目录'
    reference.mkdir()
    candidate.mkdir()
    files = sorted(SOURCE.glob('*.py'))
    assert len(files) == 33
    for file in files:
        committed = subprocess.check_output(['git', 'show', '66882634:'+str(REL/file.name)], cwd=MAIN)
        assert file.read_bytes() == committed, f'正式工作区与指定版本不同：{file.name}'
        shutil.copyfile(file, reference/file.name)
        shutil.copyfile(file, candidate/file.name)
    path = candidate/'query_projection_stages.py'
    before = path.read_text()
    tree = ast.parse(before)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')
    lines = before.splitlines(keepends=True)
    loop_start = next(i for i in range(fn.lineno, fn.end_lineno) if '    for qproj_round in pl.range' in lines[i])
    return_line = next(i for i in range(loop_start, fn.end_lineno) if lines[i].startswith('    return q_proj_i32'))
    old_loop = ''.join(lines[loop_start:return_line])
    new_branch = '''    # 两个真实M128块时复用整K权重：只改变GM→L1搬运，不改变M/N/K数学。
    # 256KiB权重在两个M块之间保持存活；QR仍按K256进行stage=2流水，
    # 两个QR槽最多64KiB，目标L1活跃容量320KiB。其余T沿用原流式路径。
    # Host仅按已有T描述符选择128<T<=256；实际M循环和尾valid仍读device T。
    # constexpr关闭时整支不进入IR，避免给非目标小T改变片上规划和事件。
    if QB_B_RESIDENT:
        for resident_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
            resident_n = qproj_worker + resident_round * WORKERS
            resident_col = resident_n * QPROJ_MM_N_TILE
            # 显式Mat载入放在M循环外，避免把GM的惰性view误当已驻留权重。
            # 外层wq_b的Native NZ声明与BYPASS属性继续作用于这一次载入。
            resident_weight = pl.load(
                wq_b, [0, resident_col], [Q_LORA, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat
            )
            for resident_t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
                resident_rows = pl.min(QPROJ_PIPE_M_TILE, tokens - resident_t0)
                resident_qr_first = pl.load(
                    qr_i8_matmul,
                    [resident_t0, 0],
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                # Mat slice只供矩阵消费；后端应融合成Mat→Right提取，不能
                # 生成Mat→Mat搬运，也不能将整K权重移入64KiB的Right。
                resident_w_first = pl.tile.slice(
                    resident_weight, [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], [0, 0]
                )
                # 首段创建带有效M的compact Acc；不对固定满M种子盲加尾块。
                resident_acc = pl.tile.matmul(resident_qr_first, resident_w_first)
                for resident_k in pl.pipeline(QPROJ_PIPE_K_TILE, Q_LORA, QPROJ_PIPE_K_TILE, stage=2):
                    resident_qr = pl.load(
                        qr_i8_matmul,
                        [resident_t0, resident_k],
                        [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                        valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    resident_w = pl.tile.slice(
                        resident_weight, [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], [resident_k, 0]
                    )
                    resident_acc = pl.tile.matmul_acc(resident_acc, resident_qr, resident_w)
                # Tile有效行沿原T尾部传播到FIX，只写本M块的实际query行。
                q_proj_i32 = pl.store(resident_acc, [resident_t0, resident_col], q_proj_i32)
    else:
'''
    body = new_branch + ''.join('    '+line if line.strip() else line for line in old_loop.splitlines(keepends=True))
    after = ''.join(lines[:loop_start]) + body + ''.join(lines[return_line:])
    # ND接受同一默认False参数但不改变旧数学；真正的B常驻只在NZ实现。
    assert after.count('    WORKERS: pl.constexpr,\n):') == 4
    nd_start = after.index('def _q_b_nd_stage(')
    nz_end = after.index('\n\n@pl.jit.inline(auto_scope=False)\ndef q_dequant_rope_stage(')
    qb_region = after[nd_start:nz_end].replace('    WORKERS: pl.constexpr,\n):',
        '    WORKERS: pl.constexpr,\n    QB_B_RESIDENT: pl.constexpr = False,\n):')
    after = after[:nd_start] + qb_region + after[nz_end:]
    after = after.replace('    """NZ Q_B：显式 AIC 按列块条带领取，原 K 流水不变；调用前必须等待 QR。"""',
                          '    """NZ Q_B：双M区间显式复用整K权重，原K顺序/列owner/有效尾行保持。"""')
    after = after.replace('    NZ 保留 M128/N256/K256，ND 保留 M64/N512/K128；按实际 T 计算\n',
                          '    NZ 保留 M128/N256/K256；128<T<=256时整K权重在L1跨两M复用，\n'
                          '    其它T仍按K256流式搬运。ND保留M64/N512/K128；按实际T计算\n')
    path.write_text(after)
    new_tree = ast.parse(after)
    new_fn = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')
    # 只剥去新参数核对其它数学，ND允许接收而不消费此参数。
    new_nd = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nd_stage')
    assert new_nd.args.args[-1].arg == 'QB_B_RESIDENT'
    new_nd.args.args.pop()
    new_nd.args.defaults.pop()
    old_other = [n for n in tree.body if n is not fn and not isinstance(n, ast.Expr)]
    new_other = [n for n in new_tree.body if n is not new_fn and not isinstance(n, ast.Expr)]
    assert ast.dump(ast.Module(body=old_other, type_ignores=[])) == ast.dump(ast.Module(body=new_other, type_ignores=[]))
    original_for = next(n for n in fn.body if isinstance(n, ast.For))
    conditional = next(n for n in new_fn.body if isinstance(n, ast.If))
    assert ast.dump(original_for) == ast.dump(conditional.orelse[0]), '原流式分支AST必须不变'
    # constexpr透传只进入AIC；Vector调度/等待没有新增参数或修改。
    entry = candidate/'entry.py'
    entry_text = entry.read_text()
    insertions = []
    for node in ast.walk(ast.parse(entry_text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('cube_schedule', '_csa_core_group'):
            insertions.append(node.end_lineno-1)
    assert len(insertions) == 2
    entry_lines = entry_text.splitlines(keepends=True)
    for offset in sorted(insertions, reverse=True):
        entry_lines.insert(offset, '            QB_B_RESIDENT,\n')
    entry_text = ''.join(entry_lines)
    assert entry_text.count('    QA_M16_PARALLEL: pl.constexpr = False,\n') == 2
    entry_text = entry_text.replace('    QA_M16_PARALLEL: pl.constexpr = False,\n',
        '    QA_M16_PARALLEL: pl.constexpr = False,\n    QB_B_RESIDENT: pl.constexpr = False,\n')
    entry_text = entry_text.replace('    QA_M16_PARALLEL默认False；',
        '    QB_B_RESIDENT默认False；Host按128<T<=256选择整K的B跨M复用，实际T仍动态。\n    QA_M16_PARALLEL默认False；')
    entry.write_text(entry_text)
    cube = candidate/'cube_schedule.py'
    cube_text = cube.read_text().replace('    QA_M16_PARALLEL: pl.constexpr = False,\n',
        '    QA_M16_PARALLEL: pl.constexpr = False,\n    QB_B_RESIDENT: pl.constexpr = False,\n')
    cube_text = cube_text.replace('    q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS)\n',
        '    # B常驻策略只改变本helper搬运；device T、24核列owner及QB发布保持。\n'
        '    q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS, QB_B_RESIDENT)\n')
    cube.write_text(cube_text)
    policy = candidate/'policy.py'
    policy_text = policy.read_text().replace('    qa_m16_parallel: bool = False\n',
        '    qa_m16_parallel: bool = False\n    qb_b_resident: bool = False\n')
    policy_text = policy_text.replace('            "QA_M16_PARALLEL": self.qa_m16_parallel,\n',
        '            "QA_M16_PARALLEL": self.qa_m16_parallel,\n            "QB_B_RESIDENT": self.qb_b_resident,\n')
    policy_text = policy_text.replace('        qa_m16_parallel=group == 6 and 16 < tokens < 32,\n',
        '        qa_m16_parallel=group == 6 and 16 < tokens < 32,\n'
        '        # 原双M128区间复用同一整K权重；不按B/S6白名单，实际M尾仍由device T决定。\n'
        '        qb_b_resident=128 < tokens <= 256,\n')
    policy.write_text(policy_text)
    changes = sorted(p.name for p in reference.glob('*.py') if p.read_text() != (candidate/p.name).read_text())
    assert changes == ['cube_schedule.py', 'entry.py', 'policy.py', 'query_projection_stages.py']
    patches = []
    for name in changes:
        patches.extend(difflib.unified_diff((reference/name).read_text().splitlines(keepends=True), (candidate/name).read_text().splitlines(keepends=True),
            fromfile='source_reference/'+name, tofile='source_candidate/'+name))
    (HERE/'candidate.patch').write_text(''.join(patches))
    controls = dict(QUERY_GROUP=6, QK_PANEL=64, SCORE_TILE=1024, PREFETCH=1, TO_L0=True,
                    BALANCED=True, ROOTS=1, OA_COLS=256, OB_ROWS=128, OUTPUT_GROUP_PIPELINE=True,
                    HC_ROW_DEPENDENCIES=True, ATTENTION_QUERY_METADATA=False, HC_CONSUMER_REDUCE=False,
                    QA_M16_PARALLEL=False, QB_B_RESIDENT=True)
    (HERE/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    (HERE/'source_audit.json').write_text(json.dumps(dict(base_revision='66882634', source_files=33,
        reference_matches_formal_committed_text=True, changed_files=changes, original_streaming_branch_ast_equal=True,
        other_functions_and_constants_ast_equal_except_default_parameter=True,
        activation='Host现有描述符128 < tokens <= 256选择constexpr；实际T/M循环/尾valid保持device动态',
        physical_candidate_targets=dict(Mat=327680, Left=65536, Right=65536, Acc=131072),
        no_compilation_in_prepare=True), ensure_ascii=False, indent=2)+'\n')
    print(str(candidate))


if __name__ == '__main__':
    main()
