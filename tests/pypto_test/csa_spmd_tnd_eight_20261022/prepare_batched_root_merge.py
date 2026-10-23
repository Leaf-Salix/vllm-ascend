# SPDX-License-Identifier: Apache-2.0
"""Prepare full-G6 root batching only after the device tie probe qualifies it."""

import argparse
import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path


def transform(source, layout):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_vector_lane_stage")
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "batch_sort_valid > 0")
    loop = next(n for n in branch.body if isinstance(n, ast.For))
    assert ast.unparse(loop.target) == "batch_sort_local"
    if layout == "natural":
        pack = "batch_root_input = pl.concat(batch_new_roots, single_roots)\n"
    elif layout == "swapped":
        pack = """\
batch_new_swapped = pl.concat(
    pl.slice(batch_new_roots, [query_group_size // 2, TOPK_PAIR_WIDTH // 2], [0, TOPK_PAIR_WIDTH // 2]),
    pl.slice(batch_new_roots, [query_group_size // 2, TOPK_PAIR_WIDTH // 2], [0, 0]),
)
batch_old_swapped = pl.concat(
    pl.slice(single_roots, [query_group_size // 2, TOPK_PAIR_WIDTH // 2], [0, TOPK_PAIR_WIDTH // 2]),
    pl.slice(single_roots, [query_group_size // 2, TOPK_PAIR_WIDTH // 2], [0, 0]),
)
batch_root_input = pl.concat(batch_new_swapped, batch_old_swapped)
"""
    else:
        raise ValueError(layout)
    replacement = """\
batch_new_roots = pl.tile.extract(
    batch_sort_rows, 0, 0, [query_group_size // 2, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
)
if single_step < 2:
    single_roots = batch_new_roots
else:
"""
    replacement += textwrap.indent(pack, "    ")
    replacement += """\
    batch_root_flat = pl.reshape(batch_root_input, [1, query_group_size * TOPK_PAIR_WIDTH])
    batch_root_sorted = pl.mrgsort(batch_root_flat, block_len=TOPK_PAIR_WIDTH // 2)
    batch_root_rows = pl.reshape(batch_root_sorted, [query_group_size // 2, 2 * TOPK_PAIR_WIDTH])
    single_roots = pl.tile.extract(
        batch_root_rows, 0, 0, [query_group_size // 2, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
    )
"""
    lines = source.splitlines(keepends=True)
    lines[loop.lineno - 1 : loop.end_lineno] = [textwrap.indent(replacement, " " * loop.col_offset)]
    result = "".join(lines)
    after = ast.parse(result)
    new_fn = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    new_branch = next(
        n for n in ast.walk(new_fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "batch_sort_valid > 0"
    )
    loop_index = branch.body.index(loop)
    assert len(new_branch.body) == loop_index + 2
    assert [ast.dump(n, include_attributes=False) for n in new_branch.body[:loop_index]] == [
        ast.dump(n, include_attributes=False) for n in branch.body[:loop_index]
    ]
    # Substituting just the original merge loop must recover the complete AST,
    # including all predicates, visibility proof, arithmetic and FFTS calls.
    new_branch.body = branch.body
    assert ast.dump(tree, include_attributes=False) == ast.dump(after, include_attributes=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", required=True, choices=("natural", "swapped"))
    parser.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.probe.read_text())
    assert evidence["passed"] and evidence["execution_checks_passed"] and len(evidence["cases"]) == 48
    assert args.layout in evidence["qualified_variants"], "No exact device prefix proof for this layout"
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "output_rectangle_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    source["indexer_score_stages.py"] = transform(source["indexer_score_stages.py"], args.layout)
    name = f"batched_root_{args.layout}_v1"
    target = here / "runs" / name
    target.mkdir(exist_ok=False)
    for filename, value in source.items():
        (target / filename).write_text(value)
    shutil.copy2(here / "descriptors/C04_output_rectangle_v1.json", here / f"descriptors/C04_{name}.json")
    report = dict(
        source=str(target),
        parent="output_rectangle_production_v1.json",
        adopted=False,
        device_execution=False,
        layout=args.layout,
        numerical_probe=str(args.probe.resolve()),
        mechanism="Three 2-way 512-pair root merges become one format1/block_len=512, repeat=3 merge",
        unchanged="Cube, score math, full/partial validity guards, fallback, original new-segment sort and FFTS",
        scope="Only the existing three-complete-query 2048-score path; C04 pilot",
        files={filename: hashlib.sha256(value.encode()).hexdigest() for filename, value in source.items()},
    )
    with (here / f"{name}.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
