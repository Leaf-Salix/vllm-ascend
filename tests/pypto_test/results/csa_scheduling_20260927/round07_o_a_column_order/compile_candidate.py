"""Compile O_A column-major block assignment without changing per-tile arithmetic."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
P = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_o_proj.py"
original = subprocess.check_output(["git", "show", "2dd51f15:" + P], cwd=REPO, text=True)
old = "        pa_rb = pa_unit // (O_LORA // A_COL_TILE)  # row block outermost\n        nf = pa_unit % (O_LORA // A_COL_TILE)"
assert original.count(old) == 2
candidate = original.replace(old, "        pa_rb = pa_unit % proj_a_rows\n        nf = pa_unit // proj_a_rows")
(ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile="a/"+P, tofile="b/"+P)))
sys.path.insert(0, str(REPO / "tests/pypto_test"))
from dsv4_csa_env import activate
activate()
for path, source in [(P, candidate), ("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py", subprocess.check_output(["git", "show", "2dd51f15:vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py"], cwd=REPO, text=True))]:
    target = ROOT / Path(path).name
    target.write_text(source)
    qualified = path.removesuffix(".py").replace("/", ".")
    spec = importlib.util.spec_from_file_location(qualified, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"), run_name="__main__")
