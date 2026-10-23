# SPDX-License-Identifier: Apache-2.0
"""Freeze a C01-only query/head partition pilot, preserving all K reductions."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def rewrite_query_expressions(source):
    """Replace every stream query index while retaining physical-core ring slots."""
    lines = source.splitlines(keepends=True)
    matches = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Name)
            and node.left.id == "qk_core"
            and isinstance(node.right, ast.BinOp)
            and isinstance(node.right.op, ast.Mult)
            and isinstance(node.right.right, ast.Name)
            and node.right.right.id == "CORE_GROUPS"
        ):
            matches.append(node)
    assert len(matches) == 6, len(matches)
    for node in sorted(matches, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        assert node.lineno == node.end_lineno
        before = lines[node.lineno - 1]
        replacement = f"qk_query_core + ({ast.unparse(node.right.left)}) * (CORE_GROUPS // ATTENTION_HEAD_PARTS)"
        lines[node.lineno - 1] = before[: node.col_offset] + replacement + before[node.end_col_offset :]
    return "".join(lines)


def main():
    here = Path(__file__).resolve().parent
    parent = json.loads((here / "physical_capacity_production_v1.json").read_text())
    source = {p.name: p.read_text() for p in Path(parent["source"]).glob("*.py")}
    stream = source["attention_stream_stages.py"]
    stream = replace_once(stream, "H = 64\n", "H = 64\nATTENTION_HEAD_PARTS = 2\nPARTITION_HEADS = 32\n")
    assert stream.count("    qk_core = core_id\n") == 2
    stream = stream.replace(
        "    qk_core = core_id\n",
        "    qk_core = core_id\n"
        "    qk_query_core = qk_core // ATTENTION_HEAD_PARTS\n"
        "    qk_head_base = (qk_core % ATTENTION_HEAD_PARTS) * PARTITION_HEADS\n",
    )
    count = "    qk_query_count = (t_dim + CORE_GROUPS - 1 - qk_core) // CORE_GROUPS\n"
    assert stream.count(count) == 2
    stream = stream.replace(
        count,
        "    qk_query_count = (t_dim + CORE_GROUPS // ATTENTION_HEAD_PARTS - 1 "
        "- qk_query_core) // (CORE_GROUPS // ATTENTION_HEAD_PARTS)\n",
    )
    stream = rewrite_query_expressions(stream)
    # H remains the external 64-head pitch and ring-slot pitch. Only per-cluster
    # computation and per-lane live accumulators use PARTITION_HEADS=32.
    stream = stream.replace("H // 2", "PARTITION_HEADS // 2")
    stream = replace_once(stream, "pl.create_tile([H, HEAD_DIM]", "pl.create_tile([PARTITION_HEADS, HEAD_DIM]")
    stream = replace_once(
        stream, "[qk_t * H, 0], [H, HEAD_DIM]", "[qk_t * H + qk_head_base, 0], [PARTITION_HEADS, HEAD_DIM]"
    )
    stream = replace_once(
        stream, "                    [H, ATTN_K_TILE],\n", "                    [PARTITION_HEADS, ATTN_K_TILE],\n"
    )
    assert stream.count("pl.load(attn_sink_col, [qk_lane_head, 0]") == 2
    stream = stream.replace(
        "pl.load(attn_sink_col, [qk_lane_head, 0]", "pl.load(attn_sink_col, [qk_head_base + qk_lane_head, 0]"
    )
    stream = replace_once(
        stream,
        "                    m_h0 = qk_lane_head + pub_local_head\n",
        "                    m_h0 = qk_head_base + qk_lane_head + pub_local_head\n",
    )
    source["attention_stream_stages.py"] = stream

    dependency = source["attention_dependencies.py"]
    dependency = replace_once(
        dependency,
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .attention_stream_stages import ATTENTION_HEAD_PARTS\n",
    )
    dependency = replace_once(
        dependency,
        "    if core_group < tokens:\n        for query in pl.range(core_group, tokens, GROUPS):\n",
        "    query_group = core_group // ATTENTION_HEAD_PARTS\n"
        "    if query_group < tokens:\n"
        "        for query in pl.range(query_group, tokens, GROUPS // ATTENTION_HEAD_PARTS):\n",
    )
    source["attention_dependencies.py"] = dependency
    cube = replace_once(
        source["cube_schedule.py"],
        "from .attention_stream_stages import attention_qk_pv_cube_stage\n",
        "from .attention_stream_stages import ATTENTION_HEAD_PARTS, attention_qk_pv_cube_stage\n",
    )
    source["cube_schedule.py"] = replace_once(
        cube,
        "    packed_workers = pl.min(tokens, GROUPS) * 2\n",
        "    packed_workers = pl.min(tokens * ATTENTION_HEAD_PARTS, GROUPS) * 2\n",
    )
    for content in source.values():
        ast.parse(content)
    target = here / "runs/head_partition2_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        (target / name).write_text(content)
    report = dict(
        source=str(target.resolve()),
        parent="physical_capacity_production_v1.json",
        adopted=False,
        device_execution=False,
        scope="C01 128K/B1, T capacity 1..6 only; not a global policy",
        mechanism="Two physical clusters per query, each computes 32 disjoint heads; each AIV computes 16 heads",
        changed_modules=["attention_stream_stages.py", "attention_dependencies.py", "cube_schedule.py"],
        unchanged="64-head external/ring pitches, K reduction order, softmax/PV block order, event pairing and drain, "
        "page guards, projection/Indexer/TopK and original native roots",
        costs="KV acquisition and KV ring traffic are duplicated twice per query; no latency prediction",
        gate="Current C01 phase diagnosis, ownership/protocol/math CPU checks, allocation/codegen and complete L1",
        files={name: hashlib.sha256(content.encode()).hexdigest() for name, content in source.items()},
    )
    with (here / "head_partition2_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
