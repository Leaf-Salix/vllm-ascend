# SPDX-License-Identifier: Apache-2.0
"""HC分行候选：保留8行物理Tile，只用valid_shape写本worker拥有的1/2/4/8行。"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

from prepare_priority_fusion import scope_name


def main():
    here = Path(__file__).resolve().parent
    name, parent = sys.argv[1:3]
    subprocess.run([sys.executable, str(here / "prepare_experiment.py"), name], check=True)
    target = here / "experiments" / name / "operator"
    shutil.copytree(here / "experiments" / parent / "operator", target, dirs_exist_ok=True)
    shutil.copyfile(here / "hc_row_stages_source.py", target / "hc_rows.py")
    names = ("hc_widen_rms_range", "hc_gates_range", "hc_mix_norm_range")
    entry_path = target / "entry.py"
    source = entry_path.read_text()
    root = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "decode_csa_tnd")
    gate = next(n for n in root.body if isinstance(n, ast.If) and ast.unparse(n.test) == "FUSED_DAG")
    for node in gate.body:
        if scope_name(node) in ("hc_widen_rms", "hc_gate_mix_norm"):
            node.items[0].context_expr.args[0] = ast.parse(
                "pl.min((tokens + HC_ROW_TILE - 1) // HC_ROW_TILE, VECTORS)", mode="eval"
            ).body
            for item in ast.walk(node):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in names:
                    item.func.id = "row_" + item.func.id
                    item.args.append(ast.Name(id="HC_ROW_TILE", ctx=ast.Load()))
    root.args.args.append(ast.arg(arg="HC_ROW_TILE", annotation=ast.parse("pl.constexpr", mode="eval").body))
    root.args.defaults.append(ast.Constant(8))
    start = min([root.lineno] + [n.lineno for n in root.decorator_list]) - 1
    lines = source.splitlines(keepends=True)
    lines[start : root.end_lineno] = [ast.unparse(ast.fix_missing_locations(root)) + "\n"]
    source = "".join(lines).replace(
        "import pypto.language as pl\n",
        "import pypto.language as pl\nfrom .hc_rows import " + ", ".join("row_" + n for n in names) + "\n",
        1,
    )
    entry_path.write_text(source)
    policy_path = target / "policy.py"
    source = policy_path.read_text()
    source = source.replace(
        "\n\n@dataclass", "\n\nHC_VECTOR_LANES = 48\nHC_ROW_TILES = (1, 2, 4, 8)\n\n\n@dataclass", 1
    )
    needle = "    fused_dag: bool = True\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + "    hc_row_tile: int = 8\n")
    needle = '            "FUSED_DAG": self.fused_dag,\n'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '            "HC_ROW_TILE": self.hc_row_tile,\n')
    needle = "    return CSAMPMDPolicy(\n"
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        needle + "        hc_row_tile=next((tile for tile in HC_ROW_TILES if tokens <= HC_VECTOR_LANES * tile), 8),\n",
    )
    policy_path.write_text(source)
    for path in (target / "hc_rows.py", entry_path, policy_path):
        ast.parse(path.read_text())
    print(target)


if __name__ == "__main__":
    main()
