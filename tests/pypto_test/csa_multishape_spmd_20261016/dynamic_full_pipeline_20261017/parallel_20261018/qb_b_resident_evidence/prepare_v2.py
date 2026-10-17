"""从冻结v1复制，修正Right生存期；不修改v1、不运行编译、不做形状实值化。"""

import ast
import difflib
import json
import shutil
from pathlib import Path

from audit_source import FoldDisabled


HERE = Path(__file__).resolve().parent


def main():
    reference = HERE / 'source_reference'
    first = HERE / 'source_candidate'
    candidate = HERE / 'source_candidate_v2'
    assert not candidate.exists(), 'v2必须写入全新目录'
    candidate.mkdir()
    files = sorted(first.glob('*.py'))
    assert len(files) == 33
    for file in files:
        shutil.copyfile(file, candidate / file.name)
    path = candidate / 'query_projection_stages.py'
    before = path.read_text()
    begin = before.index('                resident_qr_first = pl.load(')
    end = before.index('                # Tile有效行沿原T尾部传播到FIX', begin)
    # K=1024为算法常量，固定四段不是对动态T/M循环展开。
    body = '''                # 四段K256保持原INT32顺序。这里故意使用普通直线语句：
                # CanonicalizeIOOrder会把pipeline中的Mat→Right提取一起提前，
                # 两份Right各64KiB会超过A3的64KiB。普通直线不做该重排。
                # 先准备QR0/QR1，消耗QR0后才载QR2，消耗QR1后再载QR3；
                # QR同时最多两份32KiB，Right每次消费后才准备下一份。
                resident_qr0 = pl.load(
                    qr_i8_matmul,
                    [resident_t0, 0],
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                resident_qr1 = pl.load(
                    qr_i8_matmul,
                    [resident_t0, QPROJ_PIPE_K_TILE],
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                resident_w0 = pl.tile.extract(
                    resident_weight, 0, 0,
                    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                )
                # 第一段matmul据QR真实valid M创建compact Acc，尾部不取满M种子。
                resident_acc = pl.tile.matmul(resident_qr0, resident_w0)
                resident_qr2 = pl.load(
                    qr_i8_matmul,
                    [resident_t0, 2 * QPROJ_PIPE_K_TILE],
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                resident_w1 = pl.tile.extract(
                    resident_weight, QPROJ_PIPE_K_TILE, 0,
                    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                )
                resident_acc = pl.tile.matmul_acc(resident_acc, resident_qr1, resident_w1)
                resident_qr3 = pl.load(
                    qr_i8_matmul,
                    [resident_t0, 3 * QPROJ_PIPE_K_TILE],
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    valid_shape=[resident_rows, QPROJ_PIPE_K_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                resident_w2 = pl.tile.extract(
                    resident_weight, 2 * QPROJ_PIPE_K_TILE, 0,
                    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                )
                resident_acc = pl.tile.matmul_acc(resident_acc, resident_qr2, resident_w2)
                resident_w3 = pl.tile.extract(
                    resident_weight, 3 * QPROJ_PIPE_K_TILE, 0,
                    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                )
                resident_acc = pl.tile.matmul_acc(resident_acc, resident_qr3, resident_w3)
'''
    after = before[:begin] + body + before[end:]
    after = after.replace(
        '    # 256KiB权重在两个M块之间保持存活；QR仍按K256进行stage=2流水，\n',
        '    # 256KiB权重在两个M块之间保持存活；QR按K256手工交错供数，\n',
    )
    path.write_text(after)
    changes_from_first = [p.name for p in files if p.read_text() != (candidate / p.name).read_text()]
    assert changes_from_first == ['query_projection_stages.py']
    disabled = {}
    for name in ('query_projection_stages.py', 'entry.py', 'cube_schedule.py'):
        left, right = [FoldDisabled().visit(ast.parse((p / name).read_text())) for p in (reference, candidate)]
        disabled[name] = ast.dump(left) == ast.dump(right)
        assert disabled[name], name
    tree = ast.parse(after)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')
    branch = next(n for n in function.body if isinstance(n, ast.If))
    calls = [n for n in ast.walk(ast.Module(body=branch.body, type_ignores=[])) if isinstance(n, ast.Call)]
    call_names = [ast.unparse(n.func) for n in calls]
    assert 'pl.pipeline' not in call_names
    assert call_names.count('pl.tile.extract') == 4
    assert call_names.count('pl.tile.matmul') == 1
    assert call_names.count('pl.tile.matmul_acc') == 3
    assert call_names.count('pl.load') == 5  # 整B一次、QR四次；B在动态M循环外。
    assert call_names.count('pl.store') == 1
    changes = sorted(p.name for p in reference.glob('*.py') if p.read_text() != (candidate / p.name).read_text())
    assert changes == ['cube_schedule.py', 'entry.py', 'policy.py', 'query_projection_stages.py']
    patch = []
    for name in changes:
        patch.extend(difflib.unified_diff(
            (reference / name).read_text().splitlines(keepends=True),
            (candidate / name).read_text().splitlines(keepends=True),
            fromfile='source_reference/' + name, tofile='source_candidate_v2/' + name,
        ))
    (HERE / 'candidate_v2.patch').write_text(''.join(patch))
    summary = dict(
        passed=True, base_revision='66882634', source_files=33,
        changed_files_vs_reference=changes, changed_files_vs_v1=changes_from_first,
        disabled_ast_equal=disabled, policy_and_controls_unchanged_from_v1=True,
        mathematical_order='K0→K256→K512→K768；INT32 matmul初始化，三次matmul_acc',
        scheduled_order='QR0,QR1,R0/MMA0,QR2,R1/MMA1,QR3,R2/MMA2,R3/MMA3',
        no_pipeline_in_resident_branch=True,
        scope='只检查源AST与预期生存期；实际分配和事件须看本轮CPU产物',
    )
    (HERE / 'source_audit_v2.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
