"""Isolated performance-only FP16 FIXPIPE output pilot; shared precision stays intact."""

import difflib
import importlib.util
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PACKAGE = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf"
RELATIVE = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf"


def main():
    shared = (REPO / RELATIVE.replace("_perf", "") / "q_projection.py").read_text()
    function = shared[shared.index("@pl.jit.inline(auto_scope=False)\ndef _q_proj_q_matmul_nz"):]
    function = function[:function.index("\n\nq_proj_q_matmul =")]
    function = function.replace("q_proj_i32", "q_proj_compact").replace("pl.INT32]", "pl.FP16]")
    function = function.replace('"""Use upstream\'s full-K resident weight and compact tail matmul on Native NZ."""',
                                '"""Keep NZ tiling, drain INT32 Acc through scalar FIXPIPE into FP16."""')
    function = function.replace("wq_full = wq_b[0:Q_LORA, w_col0 : w_col0 + QPROJ_MM_N_TILE]",
                                "wq_full = pl.load(wq_b, [0, w_col0], [Q_LORA, QPROJ_MM_N_TILE], target_memory=pl.MemorySpace.Mat)")
    function = function.replace("qr_full = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, 0:Q_LORA]",
                                "qr_full = pl.load(qr_i8_matmul, [t0, 0], [QPROJ_M_TILE, Q_LORA], target_memory=pl.MemorySpace.Mat)")
    function = function.replace("pl.matmul(qr_full, wq_full, out_dtype=pl.INT32)", "pl.tile.matmul(qr_full, wq_full)")
    function = function.replace("q_proj_compact[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc",
                                "pl.store(col_acc, [t0, w_col0], q_proj_compact, pre_quant=1.0 / QPROJ_RESULT_RESTORE)")
    function = function.replace("pl.slice(qr_i8_matmul, [QPROJ_TAIL_M_TILE, Q_LORA], [tail_t0, 0],\n                                   valid_shape=[tail_rows, Q_LORA])",
                                "pl.load(qr_i8_matmul, [tail_t0, 0], [QPROJ_TAIL_M_TILE, Q_LORA],\n                                  valid_shape=[tail_rows, Q_LORA], target_memory=pl.MemorySpace.Mat)")
    function = function.replace("pl.matmul(qr_tail, wq_full, out_dtype=pl.INT32)", "pl.tile.matmul(qr_tail, wq_full)")
    function = function.replace("q_proj_compact[tail_t0 : tail_t0 + QPROJ_TAIL_M_TILE,\n                           w_col0 : w_col0 + QPROJ_MM_N_TILE] = tail_acc",
                                "pl.store(tail_acc, [tail_t0, w_col0], q_proj_compact, pre_quant=1.0 / QPROJ_RESULT_RESTORE)")
    projection = '''"""Performance-only compact Q_B output; ND retains the shared integer path.

Native fuses row/channel scales and BF16 conversion. The current PyPTO FIXPIPE
API carries only a constant scalar, so this pilot instead scales Acc by 2^-10
and stores FP16. Vector restores that power of two before the existing scales.
This introduces FP16 rounding before dequantization; it is not Native-equivalent.
K1024 full-range INT8 accumulation divided by 1024 fits finite FP16.
"""

import pypto.language as pl

from ..deepseek_v4_flash_dspark.q_projection import (
    H, HEAD_DIM, Q_LORA, QPROJ_MM_T_DYN, QPROJ_MM_N_TILE, QPROJ_M_TILE,
    QPROJ_WORKERS, QPROJ_TAIL_M_TILE, QPROJ_T_PAD, QPROJ_N_BLOCKS,
    _q_proj_q_matmul_nd,
)
from .nz_mode import QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ

QPROJ_RESULT_DTYPE = pl.FP16 if QUANT_WEIGHT_NZ else pl.INT32
QPROJ_RESULT_RESTORE = 1024.0 if QUANT_WEIGHT_NZ else 1.0

'''+function+"\n\nq_proj_q_matmul = _q_proj_q_matmul_nz if QUANT_WEIGHT_NZ else _q_proj_q_matmul_nd\n"
    original = (REPO / RELATIVE / "qkv_proj_rope.py").read_text()
    qkv = original.replace("    q_proj_q_matmul,\n", "")
    qkv = qkv.replace("from .nz_mode import BF16_WEIGHT_LAYOUT, BF16_WEIGHT_NZ, QUANT_WEIGHT_LAYOUT\n",
                      "from .nz_mode import BF16_WEIGHT_LAYOUT, BF16_WEIGHT_NZ, QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ\n"
                      "from .q_projection import QPROJ_RESULT_DTYPE, QPROJ_RESULT_RESTORE, q_proj_q_matmul\n")
    qkv = qkv.replace("q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32]",
                      "q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], QPROJ_RESULT_DTYPE]")
    qkv = qkv.replace("q_proj_i32 = pl.create_tensor([qproj_t_matmul, H * HEAD_DIM], dtype=pl.INT32)",
                      "q_proj_i32 = pl.create_tensor([qproj_t_matmul, H * HEAD_DIM], dtype=pl.FP16)")
    qkv = qkv.replace("q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)",
                      "q_head_acc_restored = pl.mul(q_head_acc_fp32, QPROJ_RESULT_RESTORE)\n"
                      "                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_restored, qr_scale_dq_t)")
    qkv = qkv.replace("q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)",
                      "q_head_acc_restored_tail = pl.mul(q_head_acc_fp32_tail, QPROJ_RESULT_RESTORE)\n"
                      "                    q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_restored_tail, qr_scale_dq_tail)")
    diffs = []
    for name, old, new in (("q_projection.py", "", projection), ("qkv_proj_rope.py", original, qkv)):
        (ROOT / name).write_text(new)
        diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile=f"a/{RELATIVE}/{name}" if old else "/dev/null",
                                        tofile=f"b/{RELATIVE}/{name}"))
    (ROOT / "candidate.patch").write_text("".join(diffs))
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    for name in ("q_projection", "qkv_proj_rope"):
        qualified = f"{PACKAGE}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
