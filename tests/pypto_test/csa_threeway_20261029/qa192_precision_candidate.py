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
parser.add_argument("--with-wkv-group64", action="store_true")
parser.add_argument("--with-woa-k80", action="store_true")
args = parser.parse_args()
assert not args.with_woa_k80 or args.with_wkv_group64
root = args.root
source = root / "cohorts/eightcard-source-v5/ours/source"
variant = "qa-wkv192-precision-20261102-v1" if args.with_wkv_group64 else "qa192-precision-20261102-v1"
if args.with_woa_k80:
    variant = "qa-wkv-woa192-precision-20261102-v1"
target = root / "candidates" / variant / "source"
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
changed_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
if args.with_wkv_group64:
    # Numerical helpers are copied from the measured MODE3. Only the parent's
    # worker count replaces the literal 24; the full root passes the same 24.
    native_path = target / package / name / "kv_native_stages.py"
    native_text = native_path.read_text()
    native_tree = ast.parse(native_text)
    native_before = copy.deepcopy(native_tree)
    template = ast.parse((root / "src" / tests / "wkv192_kernels.py").read_text())
    helpers = [n for n in template.body if isinstance(n, ast.FunctionDef) and n.name in ("group64_block", "ordered192")]
    renames = {"group64_block": "_kv_t192_k_block", "ordered192": "kv_t192_stage"}
    for helper in helpers:
        helper.name = renames[helper.name]
        for node in ast.walk(helper):
            if isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]
    worker = next(n for n in helpers if n.name == "kv_t192_stage")
    worker.args.args.insert(-1, ast.arg(arg="WORKERS", annotation=ast.parse("pl.constexpr", mode="eval").body))
    worker.args.defaults = [ast.Constant(value=True)]
    loop = next(n for n in worker.body if isinstance(n, ast.For))
    assert ast.unparse(loop.iter) == "pl.range(worker, 4 * 3, 24)"
    loop.iter.args[-1] = ast.Name(id="WORKERS", ctx=ast.Load())
    native_tree.body.extend(helpers)
    assert ast.dump(ast.Module(body=native_tree.body[:-2], type_ignores=[])) == ast.dump(native_before)
    ast.fix_missing_locations(native_tree)
    native_path.write_text(native_text.split('"""', 1)[0] + ast.unparse(native_tree) + "\n")

    dispatch_path = target / package / name / "kv_precision_dispatch.py"
    dispatch = ast.parse(dispatch_path.read_text())
    dispatch_before = copy.deepcopy(dispatch)
    imported = next(n for n in dispatch.body if isinstance(n, ast.ImportFrom) and n.module == "kv_native_stages")
    imported.names.append(ast.alias(name="kv_t192_stage"))
    fn = next(n for n in dispatch.body if isinstance(n, ast.FunctionDef))
    branch = next(n for n in fn.body if isinstance(n, ast.If))
    addition = ast.parse("""if tokens == 192:
        projected = kv_t192_stage(x, weight, projected, worker, WORKERS, True)
""").body[0]
    addition.orelse = branch.orelse
    branch.orelse = [addition]
    reverted = copy.deepcopy(dispatch)
    next(n for n in reverted.body if isinstance(n, ast.ImportFrom) and n.module == "kv_native_stages").names.pop()
    fn = next(n for n in reverted.body if isinstance(n, ast.FunctionDef))
    branch = next(n for n in fn.body if isinstance(n, ast.If))
    branch.orelse = branch.orelse[0].orelse
    assert ast.dump(reverted) == ast.dump(dispatch_before)
    ast.fix_missing_locations(dispatch)
    dispatch_path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(dispatch) + "\n")
    changed_files.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (native_path, dispatch_path)})
if args.with_woa_k80:
    path = target / package / name / "output_projection_a_stages.py"
    text = path.read_text()
    tree = ast.parse(text)
    original = copy.deepcopy(tree)
    template = ast.parse((root / "src" / tests / "woa192_kernels.py").read_text())
    helper = next(n for n in template.body if isinstance(n, ast.FunctionDef) and n.name == "k80_arm")
    helper.name = "_native_t192_k80_groups_stage"
    signature = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_native_t240_o_projection_a_groups_stage_nz"
    )
    helper.args = copy.deepcopy(signature.args)
    renames = {"packed": "o_packed", "weight": "wo_a", "projected": "o_r_pad", "worker": "worker_id"}
    for node in ast.walk(helper):
        if isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]
    loop = next(n for n in helper.body if isinstance(n, ast.For))
    loop.iter.args[-1] = ast.Name(id="WORKERS", ctx=ast.Load())
    # The parent calls per group: preserve its owner filter and signal ordering.
    wrapper = ast.parse("if logical >= group_begin * 8 and logical < group_end * 8: pass").body[0]
    wrapper.body = loop.body
    loop.body = [wrapper]
    for body in (n.body for n in ast.walk(helper) if isinstance(n, ast.For)):
        for n in body:
            if isinstance(n, ast.If) and ast.unparse(n.test) == "REORDER":
                n.test = ast.Constant(value=True)
    tree.body.append(helper)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "o_projection_a_groups_stage_nz")
    branch = next(n for n in fn.body if isinstance(n, ast.If))
    addition = ast.parse("""if A_COL_TILE == 256 and A_ROW_TILE == 128 and tokens == 192:
        o_r_pad = _native_t192_k80_groups_stage(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, group_begin, group_end)
""").body[0]
    addition.orelse = branch.orelse
    branch.orelse = [addition]
    reverted = copy.deepcopy(tree)
    reverted.body.pop()
    fn = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == "o_projection_a_groups_stage_nz")
    branch = next(n for n in fn.body if isinstance(n, ast.If))
    branch.orelse = branch.orelse[0].orelse
    assert ast.dump(reverted) == ast.dump(original), "unexpected changes outside T192 WO-A dispatch"
    ast.fix_missing_locations(tree)
    path.write_text(text.split('"""', 1)[0] + ast.unparse(tree) + "\n")
    changed_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
provenance_files = [
    f"{package}/deepseek_v4_flash_csa/{module}.py"
    for module in ("config", "rmsnorm", "native_adapter", "native_storage", "service")
] + [f"{tests}/{module}.py" for module in ("probe_spmd", "bind_package", "spmd_probe_entry", "ragged_fixture")]
(target.parent / "provenance.json").write_text(
    json.dumps({name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in provenance_files}, indent=2)
    + "\n"
)
(target.parent / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": "65bebf7f + isolated T192 Q-A column order"
            + (" + WKV N64 order" if args.with_wkv_group64 else "")
            + (" + WO-A K80 Native owner" if args.with_woa_k80 else ""),
            "changed_files": changed_files,
            "reverse_ast_equal": True,
            "attention_optimization_included": False,
        },
        indent=2,
    )
    + "\n"
)
print(target)
