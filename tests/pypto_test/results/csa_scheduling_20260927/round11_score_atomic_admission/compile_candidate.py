"""Compile atomic Score cohort admission with no consumer pre-staging."""
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
old = 'name_hint="indexer_score_topk_native_pair",\n        deps=[coefficients_tid, cache_write_tid],\n        allow_early_resolve=True,'
assert original.count(old) == 1
candidate = original.replace(old, 'name_hint="indexer_score_topk_native_pair",\n        deps=[coefficients_tid, cache_write_tid],\n        sync_start=True,\n        allow_early_resolve=False,')
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
