"""Compile 48 Score work units, preserving query/leaf arithmetic."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
P = "vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py"
original = subprocess.check_output(["git", "show", "07365e52:" + P], cwd=REPO, text=True)
old = "TOPK_SCORE_WORKERS = 24  # Top-K score workers"
assert original.count(old) == 1
candidate = original.replace(old, "TOPK_SCORE_WORKERS = 48  # Finer query/leaf work distribution")
(ROOT / "candidate.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile="a/"+P, tofile="b/"+P)))
sys.path.insert(0, str(REPO / "tests/pypto_test"))
from dsv4_csa_env import activate
activate()
for path, source in [(P, candidate), ("vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py", subprocess.check_output(["git", "show", "07365e52:vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_csa.py"], cwd=REPO, text=True))]:
    target = ROOT / Path(path).name
    target.write_text(source)
    qualified = path.removesuffix(".py").replace("/", ".")
    spec = importlib.util.spec_from_file_location(qualified, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"), run_name="__main__")
