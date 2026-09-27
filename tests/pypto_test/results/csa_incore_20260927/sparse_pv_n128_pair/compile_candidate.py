"""Compile an isolated PV-only candidate; no device is initialized."""
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
# Reconstruct the candidate from its recorded base without editing the checkout.
source_path='vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_sparse_attn_csa.py'
base=subprocess.check_output(['git','show','07365e52:'+source_path],cwd=REPO,text=True)
old='                    pv_kv = pl.tile.slice(qk_l1, [ATTN_K_TILE, HEAD_DIM], [pv_l1_row, 0])\n                    pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)\n                    pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)\n'
new="                    # Match Native's N128/K128 PV shape while preserving each\n                    # 128-candidate softmax and the existing query pipeline.\n                    pv_probability_left = pl.tile.move(pv_probability, target_memory=pl.MemorySpace.Left)\n                    pv_first_right = pl.tile.extract(\n                        qk_l1, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right\n                    )\n                    pv_previous = pl.tile.matmul(pv_probability_left, pv_first_right)\n                    for pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):\n                        pv_next_right = pl.tile.extract(\n                            qk_l1, pv_l1_row, pv_n * PV_N_TILE,\n                            [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right,\n                        )\n                        # Keep two accumulators live so the preceding result's\n                        # FIX write can overlap the following Cube operation.\n                        pv_current = pl.tile.matmul(pv_probability_left, pv_next_right)\n                        pl.store(pv_previous, [pv_transfer_row, (pv_n - 1) * PV_N_TILE], pv_transfer)\n                        pv_previous = pv_current\n                    pl.store(pv_previous, [pv_transfer_row, HEAD_DIM - PV_N_TILE], pv_transfer)\n"
assert base.count(old)==1
candidate=base.replace(old,new).replace('NUM_QK_CORES = 24  # qk_pv dispatch lanes', 'PV_N_TILE = 128  # Native PV uses N128/K128 and alternating accumulators.\n\nNUM_QK_CORES = 24  # qk_pv dispatch lanes')
(ROOT/'decode_sparse_attn_csa.py').write_text(candidate)
sys.path.insert(0,str(REPO/'tests/pypto_test'))
from dsv4_csa_env import activate
activate()
name='vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_sparse_attn_csa'
spec=importlib.util.spec_from_file_location(name,ROOT/'decode_sparse_attn_csa.py')
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
sys.argv=['compile_contiguous.py',str(ROOT/'compile')]
runpy.run_path(str(REPO/'tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py'),run_name='__main__')
