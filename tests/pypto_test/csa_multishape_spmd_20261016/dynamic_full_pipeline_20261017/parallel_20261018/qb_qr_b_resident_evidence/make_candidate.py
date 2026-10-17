"""从冻结 B-only 候选派生 QR+B 常驻；只改 NZ 的 constexpr 开启分支。"""

import ast
import difflib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
BASE = Path('/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/ultra_qb_weight_reuse_design_20261018/b_resident_cpu')
SOURCE = BASE / 'source_candidate'
DEST = HERE / 'source_candidate'
shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copyfile(BASE / 'controls.json', HERE / 'controls.json')
path = DEST / 'query_projection_stages.py'
before = path.read_text()
start = before.index('    # 两个真实M128块时复用整K权重：')
stop = before.index('    else:\n        for qproj_round', start)
replacement = '''    # 仅Host选出的128<T<=256区间：QR的固定容量为256×1024 INT8，
    # validT仍来自设备动态维度；每个AIC在N循环外只载一次，不能用256作有效T。
    # 与每N常驻的B1024×256合计恰512KiB Mat，不另建Mat切片或stage2副本。
    # 两个M128按原次序完成，各自K0初始化，再依次累加K256/K512/K768。
    # Right一份64KiB已满L0B：后一extract只能排在前一MMA之后；不手工删覆盖事件。
    # constexpr关闭支逐字保留B-only的原流式代码，全部跨核协议仍由外层负责。
    if QB_B_RESIDENT:
        resident_qr_full = pl.load(
            qr_i8_matmul,
            [0, 0],
            [2 * QPROJ_PIPE_M_TILE, Q_LORA],
            valid_shape=[tokens, Q_LORA],
            target_memory=pl.MemorySpace.Mat,
        )
        for resident_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
            resident_n = qproj_worker + resident_round * WORKERS
            resident_col = resident_n * QPROJ_MM_N_TILE
            resident_weight = pl.load(
                wq_b, [0, resident_col], [Q_LORA, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat
            )
            for resident_t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
                # extract从QR源validT减去动态行偏移并裁至128，生成compact Left；
                # 不把动态validT改成物理Rows，Mat布局始终以capacity256为步长。
                resident_left_first = pl.tile.extract(
                    resident_qr_full, resident_t0, 0,
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Left,
                )
                resident_right_first = pl.tile.extract(
                    resident_weight, 0, 0,
                    [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                )
                resident_acc = pl.tile.matmul(resident_left_first, resident_right_first)
                # 使用普通K循环，避免stage2先构造两份64KiB Right再做首MMA。
                # 无新增数学、fence或同步；后端必须保留M→MTE1的单槽覆盖依赖。
                for resident_k in pl.range(QPROJ_PIPE_K_TILE, Q_LORA, QPROJ_PIPE_K_TILE):
                    resident_left = pl.tile.extract(
                        resident_qr_full, resident_t0, resident_k,
                        [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], target_memory=pl.MemorySpace.Left,
                    )
                    resident_right = pl.tile.extract(
                        resident_weight, resident_k, 0,
                        [QPROJ_PIPE_K_TILE, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Right,
                    )
                    resident_acc = pl.tile.matmul_acc(resident_acc, resident_left, resident_right)
                q_proj_i32 = pl.store(resident_acc, [resident_t0, resident_col], q_proj_i32)
'''
after = before[:start] + replacement + before[stop:]
after = after.replace(
    '    NZ 保留 M128/N256/K256；128<T<=256时整K权重在L1跨两M复用，\n',
    '    NZ 保留 M128/N256/K256；128<T<=256时QR全行与整K权重各占256KiB L1，\n'
    '    QR跨本核全部N复用、权重跨两M复用；四K256仍按原顺序完成。\n',
)
after = after.replace(
    '    """NZ Q_B：双M区间显式复用整K权重，原K顺序/列owner/有效尾行保持。"""',
    '    """NZ Q_B：双M区间QR/B恰512KiB常驻；直接提取到L0，动态尾行与K顺序保持。"""',
)
path.write_text(after)
(HERE / 'candidate_vs_b_only.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(keepends=True), after.splitlines(keepends=True),
    fromfile=str(SOURCE / path.name), tofile=str(path),
)))

def q_b_tree(text):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_q_b_nz_stage')

old_tree, new_tree = q_b_tree(before), q_b_tree(after)
old_branch = next(n for n in old_tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'QB_B_RESIDENT')
new_branch = next(n for n in new_tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'QB_B_RESIDENT')
unchanged_false = ast.dump(ast.Module(body=old_branch.orelse, type_ignores=[])) == ast.dump(ast.Module(body=new_branch.orelse, type_ignores=[]))
changed = [p.name for p in sorted(SOURCE.glob('*.py')) if p.read_bytes() != (DEST / p.name).read_bytes()]
source_count = len(list(DEST.glob('*.py')))
assert unchanged_false and changed == ['query_projection_stages.py'] and source_count == 33
old_tree.body = [n for n in old_tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant) or not isinstance(n.value.value, str)]
new_tree.body = [n for n in new_tree.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant) or not isinstance(n.value.value, str)]
old_branch.body, new_branch.body = [], []
unchanged_outside_true = ast.dump(old_tree) == ast.dump(new_tree)
assert unchanged_outside_true
(HERE / 'source_audit.json').write_text(json.dumps({
    'baseline_formal_revision': '66882634',
    'b_only_source': str(SOURCE),
    'candidate_source': str(DEST),
    'files': source_count,
    'changed_files_vs_b_only': changed,
    'false_branch_ast_identical': unchanged_false,
    'nz_function_ast_except_true_and_docstring_identical': unchanged_outside_true,
    'controls_identical_to_b_only': (BASE / 'controls.json').read_bytes() == (HERE / 'controls.json').read_bytes(),
    'tail_cases': [{'tokens': t, 'm_starts': [0, 128], 'valid_rows': [128, t - 128]} for t in [129, 255, 256]],
    'source_math': '每N每M：K0 matmul初始化INT32；普通动态K循环256/512/768依次matmul_acc；输出valid由extract继承QR源validT。',
    'cpu_runs_authorized': 1,
    'device_runs_authorized': 0,
}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'source': str(DEST), 'files': source_count, 'changed': changed, 'false_ast_equal': unchanged_false}, ensure_ascii=False))
