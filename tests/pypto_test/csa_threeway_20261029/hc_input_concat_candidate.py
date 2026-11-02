# SPDX-License-Identifier: Apache-2.0
"""独立候选：把逐8列复制改成reshape/concat；保留FP32往返和尾行清零。"""

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
target = args.root / "candidates/hc-input-concat-20261102-v1/source" / package
assert not target.exists()
# 独立标量布局证明：每行256列，每8列连续重复两次，跨行不能串位。
source_indices = list(range(16 * 256))
old = [
    index
    for row in range(16)
    for chunk in range(32)
    for _ in range(2)
    for index in source_indices[row * 256 + chunk * 8 : row * 256 + chunk * 8 + 8]
]
chunks = [source_indices[i : i + 8] for i in range(0, len(source_indices), 8)]
new = [index for chunk in chunks for index in chunk + chunk]
assert len(old) == len(new) == 8192 and old == new
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_native_projection_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_pack_input_from_bf16")
container = None
for node in ast.walk(fn):
    for _field, statements in ast.iter_fields(node):
        if isinstance(statements, list):
            for index, statement in enumerate(statements):
                if (
                    isinstance(statement, ast.Assign)
                    and ast.unparse(statement.targets[0]) == "panel64_transposed_values"
                ):
                    assert container is None
                    container = statements, index
assert container is not None
body, index = container
original = copy.deepcopy(body[index : index + 4])
assert isinstance(original[2], ast.For) and ast.unparse(original[2].target) == "panel64_chunk"
assert ast.unparse(original[3].targets[0]) == "panel64_packed_values"
replacement = ast.parse("""panel64_groups = pl.reshape(panel64_values, [16 * 32, 8])
panel64_doubled = pl.concat(panel64_groups, panel64_groups)
panel64_packed_values = pl.reshape(panel64_doubled, [16, 512])
""").body
body[index : index + 4] = replacement
reverted = copy.deepcopy(tree)
f = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
loop = next(n for n in ast.walk(f) if isinstance(n, ast.For) and ast.unparse(n.target) == "panel64_input_local")
position = next(
    i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "panel64_groups"
)
loop.body[position : position + 3] = original
assert ast.dump(reverted) == ast.dump(before), "casts, padding or other HC math changed"
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
            + " + HC input reshape/concat layout",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "reverse_ast_equal": True,
            "layout_index_exact": True,
            "scope": "FP32 pack layout only; casts/padding/Native math unchanged; not qualified",
        },
        indent=2,
    )
    + "\n"
)
print(target)
