"""Keep Native-Cube scaled Score in UB through the existing Top-K sort."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
P = 'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py'
BASE = 'ec12e0e9'
base = subprocess.check_output(['git', 'show', BASE + ':' + P], cwd=REPO, text=True)
s = base.replace('BUFFERED_SCORE_LANE_TILE = BUFFERED_SCORE_TILE // 2', '''BUFFERED_SCORE_LANE_TILE = BUFFERED_SCORE_TILE // 2
# 为最后一个不足384列的panel也预留完整物理空间。
BUFFERED_SCORE_UB_COLS = (
    (TOPK_CANDIDATES_PER_LEAF // 2 + BUFFERED_SCORE_LANE_TILE - 1)
    // BUFFERED_SCORE_LANE_TILE * BUFFERED_SCORE_LANE_TILE
)''')
start = s.index('@pl.jit.inline\ndef indexer_topk_half_leaf(')
end = s.index('@pl.jit.inline\ndef indexer_topk_query_merge_one(', start)
helper = s[start:end].replace('def indexer_topk_half_leaf(', 'def indexer_topk_half_leaf_ub(').replace(
    'score_arena: pl.Tensor[[SCORE_ARENA_ROWS, TOPK_CANDIDATES_PER_LEAF], pl.FP32]',
    'score_ub: pl.Tile[[2, BUFFERED_SCORE_UB_COLS], pl.FP32]')
for cols in (512, 1024, 2048, 4096):
    helper = helper.replace(
        f'pl.load(score_arena, [score_row, 0], [1, {cols}], valid_shape=[1, valid_count])',
        f'pl.set_validshape(pl.tile.gather(score_ub, '
        f'pl.tile.adds(pl.tile.arange(0, [1, {cols}], dtype=pl.INT32), '
        f'pl.cast(score_row * BUFFERED_SCORE_UB_COLS, pl.INT32)), '
        f'pl.create_tile([1, {cols}], dtype=pl.INT32)), 1, valid_count)')
helper = helper.replace('@pl.jit.inline', '@pl.inline')
assert 'pl.load(score_arena' not in helper
# Insert the tile body at its sole call site; current jit specialization
# classifies Tensor/Scalar parameters but drops Tile annotations.
start = s.index('        for buf_score_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):')
end = s.index('    return buffered_leaf_tid', start)
aiv = s[start:end]
old = '                    for buf_score_step in pl.range(buf_score_iters):'
assert aiv.count(old) == 1
aiv = aiv.replace(old, '''                    # 有效分数全部写完后才排序；padding沿用GM路径的规则。
                    buf_score_ub = pl.create_tile([2, BUFFERED_SCORE_UB_COLS], dtype=pl.FP32)
''' + old)
old = '''                                pl.store(
                                    pl.set_validshape(buf_score_row, 1, buf_lane_valid_rows),
                                    [buf_worker * 4 + buf_query_lane * 2 + buf_score_lane, buf_score_begin],
                                    score_arena,
                                )'''
new = '''                                buf_score_ub = pl.tile.assemble(
                                    buf_score_ub, pl.tile.slice(buf_score_row, [1, BUFFERED_SCORE_LANE_TILE], [0, 0]),
                                    [buf_query_lane, buf_score_begin]
                                )'''
assert aiv.count(old) == 1
aiv = aiv.replace(old, new)
# PTOAS non-Mat TMOV requires equal physical shapes. Loads still transfer
# only one 384-element row; the larger physical tile matches the destination.
aiv = aiv.replace('                            [1, BUFFERED_SCORE_LANE_TILE],\n',
                  '                            [2, BUFFERED_SCORE_UB_COLS], valid_shape=[1, BUFFERED_SCORE_LANE_TILE],\n')

aiv = aiv.replace('''                            indexer_topk_half_leaf(
                                score_arena,
                                pair_arena,
                                buf_worker * 4 + buf_query_lane * 2 + buf_score_lane,''', '''                            indexer_topk_half_leaf_ub(
                                buf_score_ub,
                                pair_arena,
                                buf_query_lane,''')
call = """                            indexer_topk_half_leaf_ub(
                                buf_score_ub,
                                pair_arena,
                                buf_query_lane,
                                buf_half_begin,
                                buf_half_valid,
                                buf_half_slot,
                            )"""
body = helper[helper.index('    logical_begin_i32 ='):].rstrip()
import re
mapping = {'score_ub': 'buf_score_ub', 'score_row': 'buf_query_lane',
           'logical_begin': 'buf_half_begin', 'valid_count': 'buf_half_valid', 'output_slot': 'buf_half_slot'}
for old_name, new_name in mapping.items():
    body = re.sub(r'\b' + old_name + r'\b', new_name, body)
body = '\n'.join(' ' * 24 + line if line else '' for line in body.splitlines())
assert aiv.count(call) == 1
aiv = aiv.replace(call, body)
s = s[:start] + aiv + s[end:]
(ROOT / 'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True), s.splitlines(True), fromfile='a/' + P, tofile='b/' + P)))
(ROOT / 'decode_indexer.py').write_text(s)
sys.path.insert(0, str(REPO / 'tests/pypto_test'))
from dsv4_csa_env import activate
activate()
# Freeze the sparse module too: another independent candidate may be on device.
sp = 'vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py'
(ROOT / 'decode_sparse_attn_csa.py').write_text(subprocess.check_output(['git', 'show', BASE + ':' + sp], cwd=REPO, text=True))
for p in (sp, P):
    name = p.removesuffix('.py').replace('/', '.')
    spec = importlib.util.spec_from_file_location(name, ROOT / Path(p).name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
sys.argv = ['compile_contiguous.py', str(ROOT / 'compile')]
runpy.run_path(str(REPO / 'tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py'), run_name='__main__')
