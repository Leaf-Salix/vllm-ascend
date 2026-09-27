"""Select Compressor/Q_A overlap by runtime workload, without changing arithmetic."""

import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
RELATIVE = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py"
QUALIFIED = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa"


def main():
    original = subprocess.check_output(["git", "show", f"2dd51f15:{RELATIVE}"], cwd=REPO, text=True)
    old = "        # Keep Q_A ahead of the two background Compressor projections.\n        compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])"
    new = """        # Small workloads overlap Compressor projections with Q_A as upstream.
        # Large workloads retain the measured Q_A-first scheduling policy.
        if t_dim >= 144:
            compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])
        else:
            compressor_dep = late_dep"""
    assert original.count(old) == 1
    candidate = original.replace(old, new)
    (ROOT / "decode_csa.py").write_text(candidate)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True), fromfile=f"a/{RELATIVE}", tofile=f"b/{RELATIVE}")))
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    spec = importlib.util.spec_from_file_location(QUALIFIED, ROOT / "decode_csa.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[QUALIFIED] = module
    spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
