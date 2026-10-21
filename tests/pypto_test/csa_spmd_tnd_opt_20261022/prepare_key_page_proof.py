# SPDX-License-Identifier: Apache-2.0
"""Reuse the existing exhaustive device page-layout proof for AIC key addresses."""

import ast
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

LAYOUT_INIT = """
key_page_direction = pl.cast(0, pl.INT32)
key_first_page = pl.cast(0, pl.INDEX)
if SCALE_PAGE_COALESCING:
    key_layout_base = indexer_scale_layout_base(
        pl.cast(pl.tensor.dim(kv_seq_lens, 0), pl.INDEX),
        pl.tensor.dim(position_ids, 0), TOPK_SCORE_WORKERS * 2,
    )
    key_layout_row = key_layout_base + LAYOUT_WORDS_PER_REQUEST * buf_batch_idx
    key_page_direction = pl.read(offsets, [key_layout_row + 1])
    key_first_page = pl.cast(pl.read(offsets, [key_layout_row]), pl.INDEX)
"""


def add_signature(text, name, parameter):
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    last = fn.args.args[-1]
    lines[last.end_lineno : last.end_lineno] = ["    " + parameter + ",\n"]
    return "".join(lines)


def add_call(text, name, argument):
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    assert nodes, name
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        last = node.args[-1]
        assert last.end_lineno < node.end_lineno
        lines[last.end_lineno : last.end_lineno] = [" " * last.col_offset + argument + ",\n"]
    return "".join(lines)


def address_replacement(target, safe_page):
    return f"""
{target} = pl.cast(0, pl.INDEX)
if SCALE_PAGE_COALESCING and key_page_direction != 0:
    {target} = key_first_page + pl.cast(key_page_direction, pl.INDEX) * {safe_page}
else:
    {target} = pl.cast(pl.read(idx_block_table, [buf_batch_idx, {safe_page}]), pl.INDEX)
"""


def score_source(source):
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    lines = source.splitlines(keepends=True)
    patches = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("buf_prime_physical_page", "buf_physical_page"):
            target = ast.unparse(n.targets[0])
            safe = "buf_prime_safe_page" if "prime" in target else "buf_safe_page"
            patches.append(
                (
                    n.lineno - 1,
                    n.end_lineno,
                    textwrap.indent(address_replacement(target, safe).lstrip(), " " * n.col_offset),
                )
            )
    assert len(patches) == 2
    cache = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "buf_cache_len")
    patches.append(
        (
            cache.end_lineno,
            cache.end_lineno,
            textwrap.indent(textwrap.dedent(LAYOUT_INIT).lstrip(), " " * cache.col_offset),
        )
    )
    for start, end, replacement in sorted(patches, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    for name in ("_indexer_score_cube_split_store", "indexer_score_cube_stage"):
        result = add_signature(result, name, "SCALE_PAGE_COALESCING: pl.constexpr = False")
    return add_call(result, "_indexer_score_cube_split_store", "SCALE_PAGE_COALESCING")


def main():
    here = Path(__file__).resolve().parent
    reference = here / "runs/frozen_v122"
    expected = json.loads((here / "frozen_v122.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*.py")} == expected
    revised = {"indexer_score_stages.py": score_source((reference / "indexer_score_stages.py").read_text())}
    cube = add_signature(
        (reference / "cube_schedule.py").read_text(), "cube_schedule", "SCALE_PAGE_COALESCING: pl.constexpr = False"
    )
    revised["cube_schedule.py"] = add_call(cube, "indexer_score_cube_stage", "SCALE_PAGE_COALESCING")
    revised["entry.py"] = add_call((reference / "entry.py").read_text(), "cube_schedule", "SCALE_PAGE_COALESCING")
    target = here / "runs/key_page_proof"
    shutil.copytree(reference, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, text in revised.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = {
        "source": str(target),
        "reference_kind": "frozen_tnd_v122",
        "adopted": False,
        "device_execution": False,
        "scope": "existing SCALE_PAGE_COALESCING and split-G6 path only; no extra plan or host metadata",
        "preserved": [
            "all logical and physical bounds checks",
            "tail page clamping",
            "zero-direction fallback",
            "all payload acquisition and publication",
            "key layouts and page order",
            "all arithmetic",
        ],
        "changed_files": list(revised),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.py")},
    }
    with (here / "key_page_proof.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
