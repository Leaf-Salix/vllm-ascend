"""Keep the heavier Attention projection after Q_B, following upstream TP chain idea."""
import difflib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
PREFIX = "vllm_ascend/ops/pypto/"
paths = [PREFIX+"deepseek_v4_flash_dspark/q_projection.py", PREFIX+"deepseek_v4_flash_dspark_perf/qkv_proj_rope.py", PREFIX+"deepseek_v4_flash_dspark_perf/decode_csa.py"]
originals = {p: subprocess.check_output(["git", "show", "07365e52:"+p], cwd=REPO, text=True) for p in paths}
candidates = dict(originals)
p = paths[1]
s = candidates[p]
a = s.index("def q_proj_q(")
b = s.index("def q_proj_rope(", a)
block = s[a:b]
block = block.replace("    t_dim = pl.tensor.dim(x, 0)", "    qproj_tids = pl.array.create(1, pl.TASK_ID)\n    qproj_tids[0] = pl.system.task_invalid()\n    t_dim = pl.tensor.dim(x, 0)", 1)
block = block.replace("            q_proj_q_dequant(", "            qproj_tids[0] = _qproj_tid\n            q_proj_q_dequant(", 1)
block = block.replace("    return q\n", "    return q, qproj_tids[0]\n")
s = s[:a] + block + s[b:]
a = s.index("def q_proj_rope(")
b = s.index("def kv_proj_rope(", a)
block = s[a:b].replace("    q_proj_q(\n", "    q, qb_tid = q_proj_q(\n", 1).replace("    return qa_tid", "    return qa_tid, qb_tid")
s = s[:a] + block + s[b:]
s = s.replace("    qa_tid = q_proj_rope(", "    qa_tid, qb_tid = q_proj_rope(").replace("    return q, qa_tid", "    return q, qa_tid, qb_tid")
candidates[p] = s
p = paths[2]
candidates[p] = candidates[p].replace("q, qa_tid = qkv_proj_rope(", "q, qa_tid, qb_tid = qkv_proj_rope(").replace("compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])", "compressor_dep = pl.system.task_dummy(deps=[late_dep, qb_tid])")
(ROOT / "candidate.patch").write_text("".join("".join(difflib.unified_diff(originals[p].splitlines(True), candidates[p].splitlines(True), fromfile="a/"+p, tofile="b/"+p)) for p in paths))
sys.path.insert(0, str(REPO / "tests/pypto_test"))
from dsv4_csa_env import activate
activate()
for p in paths:
    target = ROOT / Path(p).name
    target.write_text(candidates[p])
    qualified = p.removesuffix(".py").replace("/", ".")
    spec = importlib.util.spec_from_file_location(qualified, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
sys.argv = ["compile_contiguous.py", str(ROOT / "compile")]
runpy.run_path(str(REPO / "tests/pypto_test/results/csa_split_optimization_20260927/compile_contiguous.py"), run_name="__main__")
