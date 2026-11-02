# SPDX-License-Identifier: Apache-2.0
"""从冻结精度版独立复制：仅批量搬运compressed KV，不改Native attention数学。"""

import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("--source", type=Path)
parser.add_argument("--label", default="attention-pair-mgather-20261102-v1")
args = parser.parse_args()
assert Path(args.label).name == args.label
package = "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
source_root = args.source or args.root / "cohorts/eightcard-source-v5/ours/source"
source = source_root / package
target = args.root / "candidates" / args.label / "source" / package
baseline_identity = json.loads((source_root.parent / "manifest.json").read_text())["csa_source"]
assert not target.exists()
shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
s = (source / "attention_native_stream_stages.py").read_text()
tree = ast.parse(s)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_attention_vector")
before = copy.deepcopy(fn)
fn.name = "native_attention_vector_planned"
index = next(i for i, a in enumerate(fn.args.args) if a.arg == "cmp_sparse_indices") + 1
fn.args.args.insert(
    index,
    ast.arg(
        arg="physical_indices", annotation=ast.parse("pl.Tensor[[PHYSICAL_ROWS_DYN, 512], pl.INT32]", mode="eval").body
    ),
)
for name, default in [("PLAN_PHYSICAL_ROWS", "False"), ("PLAN_TABLE_COLUMNS", "PHYSICAL_PLAN_COLUMNS")]:
    fn.args.args.append(ast.arg(arg=name, annotation=ast.parse("pl.constexpr", mode="eval").body))
    fn.args.defaults.append(ast.parse(default, mode="eval").body)
mode = ast.parse("""physical_mode = physical_plan_supported(
    pl.tensor.dim(cmp_block_table, 1), pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
    PLAN_PHYSICAL_ROWS, PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS)""").body[0]
fn.body.insert(0, mode)
query_loop = next(
    n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "qk_t"
)
position = ast.parse("""if PLAN_PHYSICAL_ROWS:
    physical_position = pl.cast(pl.read(position_ids, [qk_t]), pl.INDEX)
else:
    physical_position = pl.cast(0, pl.INDEX)""").body[0]
query_loop.body.insert(2, position)
# 只包装原64行KV块初始化和SWA/compressed读取分支，后面的store/事件均原样。
container = None
for node in ast.walk(fn):
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if (
                    isinstance(item, ast.Assign)
                    and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == "qk_kv_half"
                    and "tile.full" in ast.unparse(item.value)
                ):
                    assert container is None
                    container = value, i
assert container is not None
body, i = container
original = copy.deepcopy(body[i : i + 2])
assert isinstance(original[1], ast.If) and ast.unparse(original[1].test) == "qk_sb == 0"
fast = ast.parse("""if (physical_mode != 0 and qk_sb > 0 and physical_position >= 0
        and physical_position < 16777216):
    qk_kv_half = native_gather_physical_pairs(
        cmp_kv_flat, cmp_sparse_indices, physical_indices, qk_t,
        (qk_sb - 1) * ATTN_K_TILE + qk_part_row,
        (physical_position + 1) // COMPRESS_RATIO)
""").body[0]
fast.orelse = original
body[i : i + 2] = [fast]
# 逆变换证明：除了额外plan传参与这一个搬运分支，全部数学/同步AST保持冻结版。
reverted = copy.deepcopy(fn)
reverted.name = before.name
reverted.args.args.pop(index)
reverted.args.args = reverted.args.args[:-2]
reverted.args.defaults = reverted.args.defaults[:-2]
reverted.body.pop(0)
loop = next(
    n for n in ast.walk(reverted) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "qk_t"
)
loop.body.pop(2)
for node in ast.walk(reverted):
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for j, item in enumerate(value):
                if (
                    isinstance(item, ast.If)
                    and isinstance(item.test, ast.BoolOp)
                    and "physical_mode != 0" in ast.unparse(item.test)
                ):
                    value[j : j + 1] = copy.deepcopy(original)
assert ast.dump(reverted) == ast.dump(before)
local = {a.arg for a in fn.args.args} | {
    n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
}
extra = {"pl", "PHYSICAL_ROWS_DYN", "PHYSICAL_PLAN_COLUMNS", "physical_plan_supported", "native_gather_physical_pairs"}
free = sorted({n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - local - extra)
header = s.split('"""', 1)[0]
imports = (
    "import pypto.language as pl\nfrom .attention_native_stream_stages import (\n"
    + "".join("    " + n + ",\n" for n in free)
    + ")\n"
)
imports += "from .attention_plan_stages import PHYSICAL_PLAN_COLUMNS, physical_plan_supported\n"
imports += "from .native_pair_mgather import PHYSICAL_ROWS_DYN, native_gather_physical_pairs\n"
ast.fix_missing_locations(fn)
(target / "attention_physical_stream_stages.py").write_text(
    header + '"""Native数学不变，仅复用物理plan批量搬KV。"""\n' + imports + "\n" + ast.unparse(fn) + "\n"
)
helper = Path(__file__).with_name("native_pair_mgather_template.py").read_text()
(target / "native_pair_mgather.py").write_text(helper)
v = target / "vector_schedule.py"
old = ast.parse(v.read_text())
new = copy.deepcopy(old)
imp = next(n for n in new.body if isinstance(n, ast.ImportFrom) and n.module == "attention_native_stream_stages")
assert len(imp.names) == 1 and imp.names[0].name == "native_attention_vector"
imp.module = "attention_physical_stream_stages"
imp.names[0] = ast.alias(name="native_attention_vector_planned", asname="native_attention_vector")
call = next(
    n
    for n in ast.walk(new)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "native_attention_vector"
)
assert isinstance(call.args[index - 1], ast.Name) and call.args[index - 1].id == "topk_indices"
call.args.insert(index, ast.Name(id="attention_indices", ctx=ast.Load()))
call.keywords.extend(
    [ast.keyword(arg=n, value=ast.Name(id=n, ctx=ast.Load())) for n in ["PLAN_PHYSICAL_ROWS", "PLAN_TABLE_COLUMNS"]]
)
check = copy.deepcopy(new)
imp2 = next(n for n in check.body if isinstance(n, ast.ImportFrom) and n.module == "attention_physical_stream_stages")
imp2.module = "attention_native_stream_stages"
imp2.names[0] = ast.alias(name="native_attention_vector")
call2 = next(
    n
    for n in ast.walk(check)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "native_attention_vector"
)
call2.args.pop(index)
call2.keywords = call2.keywords[:-2]
assert ast.dump(check) == ast.dump(old)
ast.fix_missing_locations(new)
v.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(new) + "\n")
files = [v, target / "attention_physical_stream_stages.py", target / "native_pair_mgather.py"]
(target.parents[4] / "manifest.json").write_text(
    json.dumps(
        dict(
            csa_source=baseline_identity + " + physical plan / Native pair MGather only",
            changed_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        ),
        indent=2,
    )
    + "\n"
)
print(target)
