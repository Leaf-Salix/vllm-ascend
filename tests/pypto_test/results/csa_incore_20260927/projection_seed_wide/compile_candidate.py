"""Compile wide contiguous seed stores without editing live NPU sources."""

import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
RELATIVE = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/qkv_proj_rope.py"
QUALIFIED = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.qkv_proj_rope"


def main():
    old = subprocess.check_output(["git", "show", f"b9b05001:{RELATIVE}"], cwd=REPO, text=True)
    qa_old = '''        for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
            for nseed0 in pl.range(0, Q_LORA, QR_N_TILE):
                qr_seed = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = qr_seed
'''
    qa_new = '''        # Reuse one full-width zero tile; keep the same single seed task
        # and padded row range, avoiding repeated narrow strided stores.
        qr_seed = pl.full([QR_M_TILE, Q_LORA], dtype=pl.FP32, value=0.0)
        for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
            qr_fp32[ts0 : ts0 + QR_M_TILE, :] = qr_seed
'''
    kv_old = '''                for kts0 in pl.range(0, t_matmul, KV_M_TILE):
                    for kvseed0 in pl.range(0, HEAD_DIM, KV_N_TILE):
                        kv_seed = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
                        kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = kv_seed
'''
    kv_new = '''                kv_seed = pl.full([KV_M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                for kts0 in pl.range(0, t_matmul, KV_M_TILE):
                    kv_fp32[kts0 : kts0 + KV_M_TILE, :] = kv_seed
'''
    assert old.count(qa_old) == 2 and old.count(kv_old) == 1
    candidate = old.replace(qa_old, qa_new).replace(kv_old, kv_new)
    target = ROOT / "qkv_proj_rope.py"
    target.write_text(candidate)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(True), candidate.splitlines(True), fromfile=f"a/{RELATIVE}", tofile=f"b/{RELATIVE}")))
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    spec = importlib.util.spec_from_file_location(QUALIFIED, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[QUALIFIED] = module
    spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
