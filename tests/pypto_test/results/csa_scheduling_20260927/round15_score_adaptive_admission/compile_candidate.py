"""Compile one Score implementation with runtime workload-based admission."""
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
candidate = original.replace('TOPK_SCORE_WORKERS = 24  # Top-K score workers', 'TOPK_SCORE_WORKERS = 24  # Top-K score workers\nTOPK_LONG_SCORE_WORKERS = 2 * TOPK_SCORE_WORKERS')
start = candidate.index('def indexer_score_topk_native_cube(')
end = candidate.index('def indexer_score_topk_forest(', start)
section = candidate[start:end]
section = section.replace('    cache_write_tid: pl.Scalar[pl.TASK_ID],\n):', '    cache_write_tid: pl.Scalar[pl.TASK_ID],\n    score_workers: pl.constexpr,\n):', 1)
section = section.replace('TOPK_SCORE_WORKERS', 'score_workers')
candidate = candidate[:start] + section + candidate[end:]
start = candidate.index('        score_tid = indexer_score_topk_native_cube(', candidate.index('def indexer_score_topk_forest('))
end = candidate.index('\n    else:', start)
call = candidate[start:end]
long_call = call.replace('            cache_write_tid,\n', '            cache_write_tid,\n            TOPK_LONG_SCORE_WORKERS,\n')
short_call = call.replace('            cache_write_tid,\n', '            cache_write_tid,\n            TOPK_SCORE_WORKERS,\n')
indent = lambda value: '\n'.join('    ' + line for line in value.splitlines())
replacement = ('        # 多leaf细分为两波任务；单leaf保留24任务，避免额外派发成本。\n'
               '        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:\n' + indent(long_call)
               + '\n        else:\n' + indent(short_call))
candidate = candidate[:start] + replacement + candidate[end:]
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
