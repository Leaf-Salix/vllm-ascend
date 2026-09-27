"""Use upstream row-by-column O_A grid with Native NZ input; preserve group dependencies."""

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
    begin = original.index("@pl.jit.inline\ndef _proj_a_mm_nz(")
    middle = original.index("@pl.jit.inline\ndef _proj_a_mm_nd(", begin)
    end = original.index("\n\nproj_a_mm =", middle)
    nd = original[middle:end]
    nz = nd.replace("def _proj_a_mm_nd(", "def _proj_a_mm_nz(")
    nz = nz.replace('"""ND 版：与上游 _decode_o_proj 同形，(行块 x N 块) 二维展开、行块最外。"""',
                    '"""Parallelize row and column tiles as upstream; retain Native NZ weights."""')
    nz = nz.replace("        n0 = nf * A_COL_TILE", 
                    "        # The block index remainder is nonnegative; make the NZ bound explicit.\n"
                    "        n0 = pl.max(nf, 0) * A_COL_TILE")
    candidate = original[:begin] + nz + "\n\n" + original[middle:]
    candidate = candidate.replace('# proj_a 的 grid 有两种形状，NZ 与 ND 各一个函数，在下面按开关绑定到 `proj_a_mm`。', '# NZ 与 ND 都按行块×列块分配；NZ 单独显式约束列偏移非负，下面按开关绑定。')
    candidate = candidate.replace("PROJ_A_ROW_TILE = 128  # proj_a token block; one block covers T_PAD, 8 tasks/group",
                                 "PROJ_A_ROW_TILE = 128  # Parallel row tile; the last block may contain fewer valid rows.")
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
