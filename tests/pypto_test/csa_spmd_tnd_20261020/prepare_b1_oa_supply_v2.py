# SPDX-License-Identifier: Apache-2.0
"""Keep the next K128 L0 operand live before accumulating the current one.

The first CPU artifact reused Right address0 for both K2 and K3, so those two
transfers could not overlap their neighboring MMAs. This changes expression
liveness to require alternating Right slots; numerical MMA order is unchanged.
The resulting event sequence still needs inspection before device execution.
"""

import ast
import difflib
import json
import runpy
import shutil
from pathlib import Path


def consume(label, previous, first):
    lines = [f"oa_weight_{label} = pl.tile.reshape(oa_w_{label}, [512, A_COL_TILE])"]
    for k in range(4):
        lines.extend([
            f"oa_l_{label}_{k} = pl.tile.extract(oa_x_{label}, 0, {k * 128},",
            "    [16, 128], target_memory=pl.MemorySpace.Left)",
            f"oa_r_{label}_{k} = pl.tile.extract(oa_weight_{label}, {k * 128}, 0,",
            "    [128, A_COL_TILE], target_memory=pl.MemorySpace.Right)",
        ])
        if k > 0:
            j = k - 1
            old = previous if j == 0 else f"oa_acc_{label}_{j - 1}"
            init = "(oa_pair == 0)" if first and j == 0 else "False"
            lines.extend([
                f"oa_acc_{label}_{j} = pl.tile.matmul_acc(",
                f"    {old}, oa_l_{label}_{j}, oa_r_{label}_{j}, init_cond={init})",
            ])
    lines.extend([
        f"oa_acc_{label}_3 = pl.tile.matmul_acc(",
        f"    oa_acc_{label}_2, oa_l_{label}_3, oa_r_{label}_3, init_cond=False)",
    ])
    return "\n".join("            " + line for line in lines) + "\n"


def main():
    here = Path(__file__).resolve().parent
    source = here / "runs/b1_oa_supply_native_candidate_v1"
    destination = here / "runs/b1_oa_supply_native_candidate_v2"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    generator = runpy.run_path(str(here / "prepare_b1_oa_supply.py"))
    function = generator["helper_source"]
    function.__globals__["consume"] = consume
    after = function()
    path = destination / "oa_small_supply.py"
    before = path.read_text()
    ast.parse(after)
    # Only scheduling independent extracts around the original MMA sequence.
    def mmas(text):
        return [ast.dump(node) for node in ast.walk(ast.parse(text))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "matmul_acc"]
    assert mmas(before) == mmas(after)
    path.write_text(after)
    (here / "b1_oa_supply_v2.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="v1/oa_small_supply.py", tofile="v2/oa_small_supply.py",
    )))
    record = json.loads((here / "b1_oa_supply_v1.json").read_text())
    record.update(candidate=str(destination), parent=str(source), same_ordered_mma_ast=True,
                  actual_v1_l0_issue="Right slots for K2/K3 both0; CPU only, no NPU submission")
    (here / "b1_oa_supply_v2.json").write_text(json.dumps(record, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
