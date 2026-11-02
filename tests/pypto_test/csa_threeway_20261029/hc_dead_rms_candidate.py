# SPDX-License-Identifier: Apache-2.0
"""独立候选：删去稍后被Native-order RMS完全覆盖的旧RMS，不改加宽和同步。"""

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
target = args.root / "candidates/hc-dead-rms-20261102-v1/source" / package
assert not target.exists()
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_widen_rms_range")
obsolete = {"w_sq_sum", "w_sq", "w_sq_row", "w_sq_tail", "w_sq_row_tail", "w_mean", "w_inv", "w_valid", "w_clean"}
zero = "pl.reshape(pl.full([1, T_TILE], dtype=pl.FP32, value=0.0), [T_TILE, 1])"
removed = []
replaced = []
for node in ast.walk(fn):
    for _field, statements in ast.iter_fields(node):
        if not isinstance(statements, list):
            continue
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.Assign):
                if isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in obsolete:
                    removed.append((statements, index, statement))
                elif ast.unparse(statement.targets[0]) == "inv_rms[w_t0:w_t0 + T_TILE, 0:1]":
                    replaced.append((statement, copy.deepcopy(statement.value)))
                    statement.value = ast.parse(zero, mode="eval").body
assert len(replaced) == 1 and {n.targets[0].id for _, _, n in removed} == obsolete
# 先证明所有非目标AST完全不变，再按倒序从每个语句容器删除旧归约。
replaced[0][0].value = copy.deepcopy(replaced[0][1])
assert ast.dump(tree) == ast.dump(before)
replaced[0][0].value = ast.parse(zero, mode="eval").body
for statements, index, statement in reversed(removed):
    assert statements[index] is statement
    del statements[index]
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
            + " + remove overwritten legacy RMS",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "removed_assignments": len(removed),
            "scope": (
                "unchanged widen/tail GM writes; inv zeroed then fully overwritten by Native-order RMS; not qualified"
            ),
        },
        indent=2,
    )
    + "\n"
)
print(target)
