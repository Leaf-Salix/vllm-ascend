# SPDX-License-Identifier: Apache-2.0
"""Keep A3's eight-row physical layout but reduce one live proof row.

Both helpers broadcast the same page check to eight rows and only read row zero.
No page columns, legal ranges, fallback decisions, signals or layouts change.
"""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from prepare_query_head_guard import edit_functions


def transform(source, name, row_constant):
    def edit(fn):
        changed = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in ("ones", "repeated"):
                assert ast.unparse(node.value.func) == "pl.set_validshape"
                assert ast.unparse(node.value.args[1]) == row_constant
                node.value.args[1] = ast.Constant(value=1)
                changed.append(ast.unparse(node.targets[0]))
        assert sorted(changed) == ["ones", "repeated"]

    return edit_functions(source, [name], edit)


def main():
    here = Path(__file__).resolve().parent
    parent = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.glob("*.py")} == expected
    target = here / "runs/page_proof_single_row"
    shutil.copytree(parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    modified = []
    for file, function, rows in (
        ("attention_plan_stages.py", "compressed_pages_all_valid", "ROWS"),
        ("indexer_scale_layout_stages.py", "indexer_scale_layout_plan_stage", "LAYOUT_PROOF_ROWS"),
    ):
        source = transform((parent / file).read_text(), function, rows)
        if file == "indexer_scale_layout_stages.py":
            assert "A3 的行归约使用 8 行副本" in source
            source = source.replace("A3 的行归约使用 8 行副本", "A3 的行归约保留 8 行物理布局但只计算第0行", 1)
        (target / file).write_text(source)
        modified.append(file)
    report = dict(
        source=str(target),
        reference_kind="frozen_tnd_v122",
        parent="frozen_tnd_v122",
        adopted=False,
        device_execution=False,
        modified_files=modified,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
        change="Only the valid row count for the repeated proof and its ones operand becomes one",
        retained="Physical eight-row tile, valid columns, page reads/ranges, output row zero, all fallback/sync/ABI",
        isa_basis="Pinned a2a3 TROWMIN/TROWMAX use src.GetValidRow() for TRowReduceInstr",
        limitations=(
            "Reduced logical work is not latency evidence; CPU semantics, generated layout and device checks required"
        ),
    )
    with (here / "page_proof_single_row.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
