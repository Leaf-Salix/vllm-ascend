# SPDX-License-Identifier: Apache-2.0
"""独立候选：HC输入pack只做BF16布局搬运，不改RMS和投影数学。"""

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
target = args.root / "candidates/hc-input-bf16-20261102-v2/source" / package
assert not target.exists()
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
path = target / "hc_native_projection_stages.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_pack_input_from_bf16")
original_name = fn.name
fn.name = "_hc_pack_input_bf16_legacy"
fast = copy.deepcopy(fn)
fast.name = "_hc_pack_input_bf16_aligned"
deleted = []
changed_dtype = []
changed_store = []
padding_loop = []
for node in ast.walk(fast):
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for index, statement in enumerate(value):
                if (
                    isinstance(statement, ast.Assign)
                    and ast.unparse(statement.targets[0]) == "panel64_values"
                    and ast.unparse(statement.value) == "pl.cast(panel64_values, pl.FP32)"
                ):
                    deleted.append((value, index, statement))
                if isinstance(statement, ast.For) and ast.unparse(statement.target) == "panel64_invalid_row":
                    padding_loop.append((value, statement))
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "panel64_packed_transposed":
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "pl.tile.full":
            dtype = next(k for k in node.value.keywords if k.arg == "dtype")
            assert ast.unparse(dtype.value) == "pl.FP32"
            changed_dtype.append(dtype)
            dtype.value = ast.parse("pl.BF16", mode="eval").body
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.store":
        if ast.unparse(node.args[0]) == "pl.tile.cast(panel64_packed_values, pl.BF16)":
            changed_store.append(node)
            node.args[0] = ast.Name(id="panel64_packed_values", ctx=ast.Load())
assert len(deleted) == len(changed_dtype) == len(changed_store) == len(padding_loop) == 1
body, index, cast = deleted[0]
del body[index]
# 三处局部逆变换后，整个模块必须与冻结源AST相同。
body.insert(index, cast)
changed_dtype[0].value = ast.parse("pl.FP32", mode="eval").body
changed_store[0].args[0] = ast.parse("pl.tile.cast(panel64_packed_values, pl.BF16)", mode="eval").body
check = copy.deepcopy(fast)
check.name = original_name
assert ast.dump(check) == ast.dump(
    next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == original_name)
)
del body[index]
changed_dtype[0].value = ast.parse("pl.BF16", mode="eval").body
changed_store[0].args[0] = ast.Name(id="panel64_packed_values", ctx=ast.Load())
# 仅T为16倍数时使用BF16快路，所有load均为完整16行，不需要尾行写零。
# 任意真实TND的非对齐T仍走冻结旧函数，不能删除其清零或要求请求补S6。
padding_loop[0][0].remove(padding_loop[0][1])
dispatcher = copy.deepcopy(fn)
dispatcher.name = original_name
dispatcher.body = ast.parse("""t_dim = pl.tensor.dim(x_flat, 0)
if t_dim % 16 == 0:
    result = _hc_pack_input_bf16_aligned(x_flat, panel64_packed_flat, worker)
else:
    result = _hc_pack_input_bf16_legacy(x_flat, panel64_packed_flat, worker)
return result
""").body
tree.body.extend((fast, dispatcher))
reverted = copy.deepcopy(tree)
reverted.body = reverted.body[:-2]
next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == fn.name).name = original_name
assert ast.dump(reverted) == ast.dump(before), "legacy path changed"
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
            + " + BF16-only HC input layout pack",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "reverse_ast_equal": True,
            "scope": "finite BF16; T%16==0 only; other T retains legacy padding; not qualified",
        },
        indent=2,
    )
    + "\n"
)
print(target)
