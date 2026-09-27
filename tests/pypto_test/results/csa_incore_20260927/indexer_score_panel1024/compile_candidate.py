"""Native-inspired larger Score chunks, with 768 retained for short histories."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
BASE='ec12e0e9'
PREFIX='vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/'
p=PREFIX+'decode_indexer.py'
base=subprocess.check_output(['git','show',BASE+':'+p],cwd=REPO,text=True)
s=base.replace('BUFFERED_SCORE_TILE = 768', 'BUFFERED_SCORE_TILE = 768\nBUFFERED_LONG_SCORE_TILE = 1024')
start=s.index('@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_native_cube(')
end=s.index('@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_forest(',start)
func=s[start:end]
func=func.replace('    cache_write_tid: pl.Scalar[pl.TASK_ID],\n', '    cache_write_tid: pl.Scalar[pl.TASK_ID],\n    score_tile: pl.constexpr,\n',1)
func=func.replace('BUFFERED_SCORE_LANE_TILE','(score_tile // 2)').replace('BUFFERED_SCORE_TILE','score_tile')
s=s[:start]+func+s[end:]
start=s.index('        score_tid = indexer_score_topk_native_cube(',s.index('def indexer_score_topk_forest'))
end=s.index('    else:\n        with pl.spmd(',start)
call=s[start:end]
assert call.count('cache_write_tid,')==1
large=call.replace('            cache_write_tid,\n','            cache_write_tid,\n            BUFFERED_LONG_SCORE_TILE,\n')
small=call.replace('            cache_write_tid,\n','            cache_write_tid,\n            BUFFERED_SCORE_TILE,\n')
s=s[:start]+'''        # 长历史的完整leaf每半区4096候选：512一轮，11轮降至8轮。
        # 短历史保留384半区，避免扩大最后一轮的padding计算。
        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:
'''+''.join('    '+line if line.strip() else line for line in large.splitlines(True))+'''        else:
'''+''.join('    '+line if line.strip() else line for line in small.splitlines(True))+s[end:]
patch=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(ROOT/'decode_indexer.py').write_text(s)
p=PREFIX+'indexer_cache.py'
base=subprocess.check_output(['git','show',BASE+':'+p],cwd=REPO,text=True)
s=base.replace('SCORE_BUFFERED_TILE_ROWS = 768','SCORE_BUFFERED_TILE_ROWS = 1024  # 覆盖长历史Score的最大物理读取范围。')
patch+=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(ROOT/'indexer_cache.py').write_text(s)
(ROOT/'candidate.patch').write_text(patch)
sys.path.insert(0,str(REPO/'tests/pypto_test'))
from dsv4_csa_env import activate
activate()
for filename in ('decode_indexer.py','indexer_cache.py'):
    name=(PREFIX+filename).removesuffix('.py').replace('/','.')
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
sys.argv=['compile_contiguous.py',str(ROOT/'compile')]
runpy.run_path(str(REPO/'tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py'),run_name='__main__')
