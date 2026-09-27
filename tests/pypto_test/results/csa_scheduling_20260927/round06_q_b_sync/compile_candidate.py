"""Compile a performance-only Q_B admission policy with the shared math intact."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PREFIX = "vllm_ascend/ops/pypto/"


def main():
    paths = [PREFIX + "deepseek_v4_flash_dspark/q_projection.py",
             PREFIX + "deepseek_v4_flash_dspark_perf/qkv_proj_rope.py",
             PREFIX + "deepseek_v4_flash_dspark_perf/decode_csa.py"]
    originals = {p: subprocess.check_output(["git", "show", "2dd51f15:" + p], cwd=REPO, text=True)
                 for p in paths}
    candidate = dict(originals)
    p = paths[0]
    old = "    qproj_dep: pl.Scalar[pl.TASK_ID],\n):"
    assert candidate[p].count(old) == 2
    candidate[p] = candidate[p].replace(old, "    qproj_dep: pl.Scalar[pl.TASK_ID],\n    sync_launch: pl.constexpr = False,\n):")
    candidate[p] = candidate[p].replace("        deps=[qproj_dep],\n", "        deps=[qproj_dep],\n        sync_start=sync_launch,\n")
    candidate[p] = candidate[p].replace("name_hint=\"qproj_matmul\", deps=[qproj_dep])", "name_hint=\"qproj_matmul\", deps=[qproj_dep], sync_start=sync_launch)")
    p = paths[1]
    old = "                tile_rows,\n                qproj_dep,\n            )"
    assert candidate[p].count(old) == 1
    candidate[p] = candidate[p].replace(old, "                tile_rows,\n                qproj_dep,\n                True,\n            )")
    patch = "".join("".join(difflib.unified_diff(originals[p].splitlines(True), candidate[p].splitlines(True),
                                             fromfile="a/" + p, tofile="b/" + p)) for p in paths)
    (ROOT / "candidate.patch").write_text(patch)
    sys.path.insert(0, str(REPO / "tests/pypto_test"))
    from dsv4_csa_env import activate
    activate()
    for p in paths:
        target = ROOT / Path(p).name
        target.write_text(candidate[p])
        qualified = p.removesuffix(".py").replace("/", ".")
        spec = importlib.util.spec_from_file_location(qualified, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
    runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
