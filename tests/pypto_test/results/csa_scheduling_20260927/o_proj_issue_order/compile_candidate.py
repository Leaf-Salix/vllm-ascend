"""Keep O projection kernels/group dependencies; submit all O_A groups first."""

import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
RELATIVE = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_o_proj.py"
QUALIFIED = "vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_o_proj"


def main():
    original = subprocess.check_output(["git", "show", f"c7a52af5:{RELATIVE}"], cwd=REPO, text=True)
    old = "    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)\n"
    assert original.count(old) == 1
    candidate = original.replace(old, "    proj_a_tids = pl.array.create(O_GROUPS, pl.TASK_ID)\n" + old)
    old = "\n            col_g = g * O_LORA\n            # 性能版按上游把 amax"
    new = "\n            proj_a_tids[g] = pa_tid\n\n        # Register all O_A groups before their quant/O_B successors.\n        # Each successor still depends only on its own group's producer.\n        for g in pl.parallel(O_GROUPS):\n            col_g = g * O_LORA\n            # 性能版按上游把 amax"
    assert candidate.count(old) == 1
    candidate = candidate.replace(old, new)
    assert candidate.count("                deps=[pa_tid],") == 1
    candidate = candidate.replace("                deps=[pa_tid],", "                deps=[proj_a_tids[g]],")
    (ROOT / "decode_o_proj.py").write_text(candidate)
    (ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), candidate.splitlines(True), fromfile=f"a/{RELATIVE}", tofile=f"b/{RELATIVE}")))
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    spec = importlib.util.spec_from_file_location(QUALIFIED, ROOT / "decode_o_proj.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[QUALIFIED] = module
    spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
