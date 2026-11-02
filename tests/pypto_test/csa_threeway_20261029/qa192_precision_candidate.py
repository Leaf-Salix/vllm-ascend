# SPDX-License-Identifier: Apache-2.0
"""只在独立副本接入已对拍的T192 Q-A列次序，保留冻结生产源码。"""

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
root = args.root
source = root / "cohorts/eightcard-source-v5/ours/source"
target = root / "candidates/qa192-precision-20261102-v1/source"
package = "vllm_ascend/ops/pypto"
name = "deepseek_v4_flash_csa_spmd_tnd"
assert not target.exists()
shutil.copytree(source / package / name, target / package / name, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(
    root / "src" / package / "deepseek_v4_flash_csa",
    target / package / "deepseek_v4_flash_csa",
    ignore=shutil.ignore_patterns("__pycache__"),
)
for path in (root / "src" / package).glob("*.py"):
    shutil.copyfile(path, target / package / path.name)
tests = "tests/pypto_test/nal_native_alignment_20261027"
(target / tests).mkdir(parents=True)
for path in (root / "src" / tests).glob("*.py"):
    shutil.copyfile(path, target / tests / path.name)

path = target / package / name / "qkv_stages.py"
text = path.read_text()
tree = ast.parse(text)
original = copy.deepcopy(tree)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "q_a_stage")
branch = next(n for n in function.body if isinstance(n, ast.If))
assert ast.unparse(branch.test) == "tokens == 240"
addition = ast.parse("""if tokens == 192:
    qa_fp32 = _q_a_t192_worker(x, wq_a, qa_fp32, worker_id, WORKERS, True)
""").body[0]
addition.orelse = branch.orelse
branch.orelse = [addition]
template = ast.parse((root / "src" / tests / "qa192_kernels.py").read_text())
helpers = [n for n in template.body if isinstance(n, ast.FunctionDef) and n.name in ("hypothesis_block", "narrow192")]
assert len(helpers) == 2
renames = {"hypothesis_block": "_q_a_t192_k_block", "narrow192": "_q_a_t192_worker"}
for helper in helpers:
    helper.name = renames[helper.name]
    for node in ast.walk(helper):
        if isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]
tree.body.extend(helpers)
reverted = copy.deepcopy(tree)
reverted.body = reverted.body[:-2]
fn = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == "q_a_stage")
old_branch = next(n for n in fn.body if isinstance(n, ast.If))
old_branch.orelse = old_branch.orelse[0].orelse
assert ast.dump(reverted) == ast.dump(original), "unexpected changes outside T192 Q-A dispatch"
ast.fix_missing_locations(tree)
path.write_text(text.split('"""', 1)[0] + ast.unparse(tree) + "\n")
(target.parent / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": "65bebf7f + isolated T192 Q-A column-order candidate",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "reverse_ast_equal": True,
            "attention_optimization_included": False,
        },
        indent=2,
    )
    + "\n"
)
print(target)
