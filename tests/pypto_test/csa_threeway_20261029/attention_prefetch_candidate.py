# SPDX-License-Identifier: Apache-2.0
"""独立副本：将CMP最后一个panel搬运移到SWA PV等待前；数学和事件不变。"""

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
baseline = args.root / "candidates/attention-pair-qa-wkv-woa192-20261102-v1"
target = args.root / "candidates/attention-prefetch-panel3-20261102-v1/source"
assert not target.exists()
shutil.copytree(baseline / "source" / package, target / package, ignore=shutil.ignore_patterns("__pycache__"))
path = target / package / "attention_physical_stream_stages.py"
text = path.read_text()
tree = ast.parse(text)
original = copy.deepcopy(tree)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_attention_vector_planned")
query = next(n for n in ast.walk(function) if isinstance(n, ast.For) and ast.unparse(n.target) == "qk_t")
position = next(
    i for i, n in enumerate(query.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "PLAN_PHYSICAL_ROWS"
)
enabled = ast.parse("""prefetch_panel3 = pl.cast(0, pl.INT32)
if (physical_mode != 0 and physical_position >= 0 and physical_position < 16777216
        and pl.read(valid_block_mask, [qk_t, 0]) > 0
        and pl.max(pl.max(pl.read(valid_block_mask, [qk_t, 1]), pl.read(valid_block_mask, [qk_t, 2])),
                   pl.max(pl.read(valid_block_mask, [qk_t, 3]), pl.read(valid_block_mask, [qk_t, 4]))) > 0):
    prefetch_panel3 = pl.cast(1, pl.INT32)
""").body
query.body[position + 1 : position + 1] = enabled
part = next(n for n in ast.walk(query) if isinstance(n, ast.For) and ast.unparse(n.target) == "qk_part")
guard = part.body[0]
assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "qk_sb > 0 or qk_part == 0"
old_test = copy.deepcopy(guard.test)
guard.test = ast.parse(
    "(qk_sb > 0 or qk_part == 0) and (qk_sb != 1 or qk_part != 3 or prefetch_panel3 == 0)", mode="eval"
).body
prefetch = ast.parse("""if qk_sb == 0 and prefetch_panel3 != 0:
    prefetched_half = native_gather_physical_pairs(
        cmp_kv_flat, cmp_sparse_indices, physical_indices, qk_t,
        3 * ATTN_CUBE_KV_TILE + qk_lane_kv,
        (physical_position + 1) // COMPRESS_RATIO)
    pl.store(prefetched_half, [qk_kv_base + 3 * ATTN_CUBE_KV_TILE + qk_lane_kv, 0], kv_transfer)
""").body[0]
sites = []
for node in ast.walk(query):
    for _field, body in ast.iter_fields(node):
        if isinstance(body, list):
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Expr) and "sync_set" in ast.unparse(statement):
                    if "QK_PROB_READY_EVENT" in ast.unparse(statement):
                        sites.append((body, index))
assert len(sites) == 1
body, index = sites[0]
body.insert(index + 1, prefetch)
# Undo only the placement/skip guards. Every original math and event AST remains.
reverted = copy.deepcopy(tree)
fn = next(n for n in reverted.body if isinstance(n, ast.FunctionDef) and n.name == function.name)
loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "qk_t")
del loop.body[position + 1 : position + 3]
part = next(n for n in ast.walk(loop) if isinstance(n, ast.For) and ast.unparse(n.target) == "qk_part")
part.body[0].test = old_test
removed = 0
for node in ast.walk(loop):
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for i, statement in enumerate(value):
                if (
                    isinstance(statement, ast.If)
                    and ast.unparse(statement.test) == "qk_sb == 0 and prefetch_panel3 != 0"
                ):
                    del value[i]
                    removed += 1
                    break
assert removed == 1 and ast.dump(reverted) == ast.dump(original)
ast.fix_missing_locations(tree)
path.write_text(text.split('"""', 1)[0] + ast.unparse(tree) + "\n")
base_manifest = json.loads((baseline / "manifest.json").read_text())
(target.parent / "manifest.json").write_text(
    json.dumps(
        {
            "csa_source": base_manifest["csa_source"] + " + CMP panel3 prefetch during SWA PV",
            "changed_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
            "baseline_manifest_sha256": hashlib.sha256((baseline / "manifest.json").read_bytes()).hexdigest(),
            "reverse_ast_equal": True,
            "scope": "CPU/static candidate only; no performance or precision qualification yet",
        },
        indent=2,
    )
    + "\n"
)
print(target)
