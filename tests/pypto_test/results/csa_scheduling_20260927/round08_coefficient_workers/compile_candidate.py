"""Compile fewer head-coefficient workers, preserving each query arithmetic."""
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
a = original.index("def indexer_head_coefficients(")
b = original.index("def indexer_score_topk_native_cube(", a)
block = original[a:b]
block = block.replace("    with pl.spmd(\n        TOPK_QUERY_WORKERS,", "    coefficient_count = pl.tensor.dim(position_ids, 0)\n    coefficient_workers = pl.min(24, pl.max(coefficient_count // 2, 1))\n    with pl.spmd(\n        coefficient_workers,")
block = block.replace("        coefficient_count = pl.tensor.dim(position_ids, 0)\n", "")
block = block.replace("coefficient_count // 2, TOPK_QUERY_WORKERS", "coefficient_count // 2, coefficient_workers")
assert block != original[a:b]
candidate = original[:a] + block + original[b:]
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
