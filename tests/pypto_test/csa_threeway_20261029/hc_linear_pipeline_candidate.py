# SPDX-License-Identifier: Apache-2.0
"""独立候选：借鉴nalinaly的两级K流水，不改变Native HC累加次序。"""

import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
baseline = args.root / "cohorts/eightcard-source-v5/ours"
target = args.root / "candidates/hc-linear-pipeline2-20261102-v1/source" / package
assert not target.exists()
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_native_projection_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_linear")
loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "panel64_wide_kb")
assert ast.unparse(loop.iter) == "pl.range(1, panel64_wide_count)"
loop.iter.func.attr = "pipeline"
loop.iter.keywords.append(ast.keyword(arg="stage", value=ast.Constant(value=2)))
reverted = copy.deepcopy(tree)
f = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == "hc_linear")
check = next(n for n in ast.walk(f) if isinstance(n, ast.For) and ast.unparse(n.target) == "panel64_wide_kb")
check.iter.func.attr = "range"
check.iter.keywords = []
assert ast.dump(reverted) == ast.dump(before), "HC arithmetic or other stages changed"
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
manifest = {
    "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
    + " + HC wide-M K pipeline(stage=2)",
    "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    "reverse_ast_equal": True,
    "scope": "T>=64 HC linear K loop only; Native math unchanged; performance/precision not qualified",
}
(target.parents[4] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(target)
