"""Compile leaf-major Score item scheduling with unchanged per-leaf arithmetic."""
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
lines = original.splitlines(True)
result = []
i = 0
changes = 0
while i < len(lines):
    line = lines[i]
    if line.strip() == "if buf_query_count < 2 * TOPK_SCORE_WORKERS:":
        indent = line[:len(line)-len(line.lstrip())]
        result.append(indent + "buf_query = buf_item % (buf_query_count // 2) * 2\n")
        result.append(indent + "buf_leaf = buf_item // (buf_query_count // 2)\n")
        i += 1
        while i < len(lines) and (len(lines[i])-len(lines[i].lstrip()) > len(indent) or lines[i].strip() == "else:"):
            i += 1
        changes += 1
        continue
    result.append(line)
    i += 1
assert changes == 2
candidate = "".join(result)
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
