# SPDX-License-Identifier: Apache-2.0
"""Create an isolated scale-gather experiment from the frozen eight-case baseline."""

import argparse
import ast
import hashlib
import json
import textwrap
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one source site: {old}")
    return source.replace(old, new, 1)


def wire_argument(source, name):
    return replace_once(
        source, f"def {name}(\n", f"def {name}(\n    idx_scale_rows: pl.Tensor[[SCALE_ROWS, 64], pl.INT8],\n"
    )


def transform_score(source):
    """Retain the complete original scalar fallback and all Score FIFO operations."""
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_vector_lane_stage"
    )
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "single_scale_page"
    )
    lines = source.splitlines(keepends=True)
    original = textwrap.dedent("".join(lines[loop.lineno - 1 : loop.end_lineno]))
    branch = """page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
cache_pages = pl.tensor.dim(idx_native_kv_cache, 0)
table_columns = pl.tensor.dim(idx_block_table, 1)
if (page_bytes >= 4160 and page_bytes % 64 == 0 and cache_pages * (page_bytes // 64) < 2147483647
        and cache_pages < 16777216 and 0 < table_columns and table_columns < 16777216
        and pl.tensor.dim(idx_block_table, 0) * table_columns < 2147483647):
    gathered_scale_half = gather_scale_pages(
        idx_scale_rows, idx_block_table, single_batch, single_score_begin,
        single_cache_len, cache_pages, page_bytes, score_tile,
    )
    single_scale_bytes = pl.tile.reinterpret_view(
        pl.fillpad(gathered_scale_half, pad_value=pl.PadValue.zero), pl.INT8
    )
else:
"""
    lines[loop.lineno - 1 : loop.end_lineno] = [
        textwrap.indent(branch + textwrap.indent(original, "    "), " " * loop.col_offset)
    ]
    revised = "".join(lines)
    return wire_argument(revised, "indexer_score_vector_lane_stage")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="scale_gather_v7")
    args = parser.parse_args()
    if not args.name.isidentifier():
        raise ValueError("Experiment name must be a simple identifier")
    baseline = here / "runs/baseline"
    expected = json.loads((here / "baseline.json").read_text())["files"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.glob("*.py")} == expected
    source = {p.name: p.read_text() for p in baseline.glob("*.py")}
    score = transform_score(source["indexer_score_stages.py"])
    source["indexer_score_stages.py"] = replace_once(
        score,
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .scale_gather_stages import SCALE_ROWS, gather_scale_pages\n",
    )
    for name, function in (("entry.py", "_csa_core_group"), ("vector_schedule.py", "vector_schedule")):
        source[name] = wire_argument(source[name], function)
        source[name] = replace_once(
            source[name],
            "import pypto.language as pl\n",
            "import pypto.language as pl\n\nfrom .scale_gather_stages import SCALE_ROWS\n",
        )
    for file, call in (
        ("entry.py", "_csa_core_group"),
        ("entry.py", "vector_schedule"),
        ("vector_schedule.py", "indexer_score_vector_lane_stage"),
    ):
        source[file] = replace_once(source[file], f" = {call}(\n", f" = {call}(\n        idx_scale_rows,\n")
    marker = "    idx_key64 = pl.reshape(shifted, [(pages * page_bytes - 64) // 128, 128])\n"
    source["entry.py"] = replace_once(
        source["entry.py"],
        marker,
        marker
        + """    # Byte-equivalent readonly view; Native cache remains the only external root.
    scale_prefix_bytes = cache_flat[0 : (pages * page_bytes // 64) * 64]
    idx_scale_rows = pl.reshape(scale_prefix_bytes, [pages * page_bytes // 64, 64])
""",
    )
    source["scale_gather_stages.py"] = (here / "scale_gather_source.py").read_text()
    for text in source.values():
        ast.parse(text)
    target = here / "runs" / args.name
    target.mkdir(parents=True, exist_ok=False)
    for name, text in source.items():
        (target / name).write_text(text)
    report = dict(
        baseline="baseline.json",
        source=str(target),
        adopted=False,
        device_execution=False,
        mechanism="Batch Native scale-page index calculation and row gathers; unchanged fragmented-page semantics",
        changes=[
            name for name, text in source.items() if hashlib.sha256(text.encode()).hexdigest() != expected.get(name)
        ],
        files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in source.items()},
        preserved=[
            "single MIX task",
            "external cache root ABI",
            "FFTS READY/ACK and drain",
            "score/TopK math and tie tree",
        ],
        hypotheses=["fewer scalar page-address checks", "larger Vec index working set may offset the saving"],
    )
    with (here / (args.name + ".json")).open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    descriptors = here / "descriptors"
    descriptors.mkdir(exist_ok=True)
    for key in ("C01", "C04", "C08"):
        evidence = json.loads((here / "baseline.json").read_text())["cases"][key]
        measured = json.loads(Path(evidence["source_report"]).read_text())
        value = dict(buffers=measured["descriptors"]["spmd"], policy=measured["policies"]["spmd"])
        path = descriptors / f"{key}.json"
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            with path.open("x") as stream:
                stream.write(json.dumps(value, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
