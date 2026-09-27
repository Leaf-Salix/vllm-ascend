"""Compile skipping compressed KV clears only for entirely valid blocks."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
P='vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py'
base=subprocess.check_output(['git','show','ec12e0e9:'+P],cwd=REPO,text=True)
old="""                for c_dt in pl.range(bias_rows):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), pl.INT32)
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)"""
new="""                c_blk_full = pl.row_min(pl.slice(
                    c_mask_tile, [BIAS_T_TILE, ATTN_K_TILE], [0, c_s0],
                    valid_shape=[bias_rows, ATTN_K_TILE],
                ))
                for c_dt in pl.range(bias_rows):
                    c_valid = pl.cast(pl.read(c_blk_valid, [c_dt, 0]), pl.INT32)
                    c_full = pl.cast(pl.read(c_blk_full, [c_dt, 0]), pl.INT32)
                    # 0: empty, 1: partial, 2: every row will be overwritten.
                    pl.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid + c_full)"""
assert base.count(old)==1
candidate=base.replace(old,new)
old="""                        qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        if qk_sb == 0:"""
new="""                        qk_kv_storage = pl.create_tile(
                            [ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Vec
                        )
                        if qk_sb == 0 or pl.read(valid_block_mask, [qk_t, qk_sb]) != 2:
                            # Clear the existing buffer only for SWA/partial blocks.
                            # A fully valid compressed block overwrites every byte.
                            qk_empty = pl.set_validshape(qk_kv_storage, 0, HEAD_DIM)
                            qk_kv_half = pl.tile.fillpad_inplace(qk_empty, pad_value=pl.PadValue.zero)
                        else:
                            # Full valid shape: no padding writes; keep the same
                            # buffer/result type in both branches.
                            qk_kv_half = pl.tile.fillpad_inplace(qk_kv_storage, pad_value=pl.PadValue.zero)
                        if qk_sb == 0:"""
assert candidate.count(old)==1
candidate=candidate.replace(old,new)
(ROOT/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+P,tofile='b/'+P)))
(ROOT/'decode_sparse_attn_csa.py').write_text(candidate)
sys.path.insert(0,str(REPO/'tests/pypto_test'))
from dsv4_csa_env import activate
activate()
name=P.removesuffix('.py').replace('/','.')
spec=importlib.util.spec_from_file_location(name,ROOT/'decode_sparse_attn_csa.py')
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
sys.argv=['compile_contiguous.py',str(ROOT/'compile')]
runpy.run_path(str(REPO/'tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py'),run_name='__main__')
