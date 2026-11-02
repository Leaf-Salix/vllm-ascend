# SPDX-License-Identifier: Apache-2.0
"""独立候选：只将HC常量权重打包从每次解码移至图外，保留精度基准。"""

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
target = args.root / "candidates/hc-prepack-20261102-v2/source" / package
assert not target.exists()
shutil.copytree(baseline / "source" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copyfile(Path(__file__).with_name("hc_weight_prepack.py"), target / "hc_weight_prepack.py")

path = target / "entry.py"
tree = ast.parse(path.read_text())
before = copy.deepcopy(tree)
for fn in (
    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("decode_csa_spmd", "decode_csa_spmd_debug")
):
    index = next(i for i, arg in enumerate(fn.args.args) if ast.unparse(arg.annotation) == "pl.constexpr")
    fn.args.args.insert(
        index,
        ast.arg(
            arg="hc_prepacked_weight",
            annotation=ast.parse("pl.InOut[pl.Tensor[[32768, 32], pl.BF16]]", mode="eval").body,
        ),
    )
    node = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "hc_packed_weight")
    assert ast.unparse(node.value) == "pl.create_tensor([32768, 32], dtype=pl.BF16)"
    node.value = ast.Name(id="hc_prepacked_weight", ctx=ast.Load())
reverted = copy.deepcopy(tree)
for fn in (
    n
    for n in reverted.body
    if isinstance(n, ast.FunctionDef) and n.name in ("decode_csa_spmd", "decode_csa_spmd_debug")
):
    fn.args.args = [arg for arg in fn.args.args if arg.arg != "hc_prepacked_weight"]
    node = next(n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "hc_packed_weight")
    node.value = ast.parse("pl.create_tensor([32768, 32], dtype=pl.BF16)", mode="eval").body
assert ast.dump(reverted) == ast.dump(before)
ast.fix_missing_locations(tree)
path.write_text("# SPDX-License-Identifier: Apache-2.0\n" + ast.unparse(tree) + "\n")

path = target / "vector_schedule.py"
text = path.read_text()
line = "    hc_packed_weight = hc_pack_weight(hc_fn, hc_packed_weight, worker)\n"
assert text.count(line) == 1
path.write_text(text.replace(line, ""))
assert (
    path.read_text().replace(
        "    signal = publish_phase(signal, PHASE_INPUT", line + "    signal = publish_phase(signal, PHASE_INPUT", 1
    )
    == text
)

path = target / "binding.py"
text = path.read_text()
text += """

# 实验专用：图外增加2MiB只读常量；原绑定的所有格式检查保持不变。
from .hc_weight_prepack import check_static_weights, precompile_static_weights, prepare_static_weights

class PreparedDynamicCSA(PreparedDynamicCSA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prepare_static_weights(self)

    def rebind(self, native):
        check_static_weights(self, native)
        super().rebind(native)

    def __call__(self, *, config=None):
        check_static_weights(self)
        return super().__call__(config=config)
"""
path.write_text(text)
manifest = {
    "csa_source": json.loads((baseline / "manifest.json").read_text())["csa_source"]
    + " + graph-external HC constant pack",
    "extra_persistent_bytes_per_binding": 2 << 20,
    "entry_reverse_ast_equal": True,
    "changed_files": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.glob("*.py")
        if p.name in ("entry.py", "vector_schedule.py", "binding.py", "hc_weight_prepack.py")
    },
    "scope": "private candidate; performance and precision not qualified",
}
(target.parents[4] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(target)
