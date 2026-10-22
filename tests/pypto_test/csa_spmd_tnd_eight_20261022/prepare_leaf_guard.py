# SPDX-License-Identifier: Apache-2.0
"""Retain the leaf proof only in the existing 64..96-token long-G6 resource tier."""

import ast
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "baseline.json").read_text())
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == baseline["files"]
    parent = here / "runs/leaf_scale_one_row_service_v1"
    for name in ("leaf_scale_layout_stages.py", "service.py", "service_config.py"):
        source[name] = (parent / name).read_text()
    for name in ("entry.py", "vector_schedule.py"):
        count = 2 if name == "entry.py" else 1
        declaration = "    SCALE_PAGE_COALESCING: pl.constexpr = False,\n"
        assert source[name].count(declaration) == count
        source[name] = source[name].replace(declaration, declaration + "    LEAF_SCALE_LAYOUT: pl.constexpr = False,\n")
    call = "            SCALE_PAGE_COALESCING,\n"
    assert source["entry.py"].count(call) == 2
    source["entry.py"] = source["entry.py"].replace(call, call + "            LEAF_SCALE_LAYOUT,\n")
    source["vector_schedule.py"] = replace_once(
        source["vector_schedule.py"],
        "if BALANCED and SCALE_PAGE_COALESCING:",
        "if BALANCED and SCALE_PAGE_COALESCING and not LEAF_SCALE_LAYOUT:",
    )
    source["vector_schedule.py"] = replace_once(
        source["vector_schedule.py"],
        "        SCALE_PAGE_COALESCING,\n",
        "        SCALE_PAGE_COALESCING,\n        LEAF_SCALE_LAYOUT,\n",
    )
    filename = "indexer_score_stages.py"
    source[filename] = replace_once(
        source[filename],
        "from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base\n",
        "from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base\n"
        "from .leaf_scale_layout_stages import indexer_leaf_scale_layout\n",
    )
    source[filename] = replace_once(
        source[filename],
        "    scale_page_coalescing: pl.constexpr,\n",
        "    scale_page_coalescing: pl.constexpr,\n    leaf_scale_layout: pl.constexpr,\n",
    )
    tree = ast.parse(source[filename])
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "scale_page_coalescing"]
    assert len(nodes) == 1
    node = nodes[0]
    lines = source[filename].splitlines(keepends=True)
    original = lines[node.body[0].lineno - 1 : node.body[-1].end_lineno]
    new = [
        " " * (node.col_offset + 4) + line + "\n"
        for line in (
            "if leaf_scale_layout:",
            "    single_first_page, single_contiguous = indexer_leaf_scale_layout(",
            "        idx_block_table, single_batch, single_page_begin, single_page_count,",
            "        pl.tensor.dim(idx_native_kv_cache, 0),",
            "    )",
            "else:",
        )
    ]
    new.extend("    " + line for line in original)
    lines[node.body[0].lineno - 1 : node.body[-1].end_lineno] = new
    source[filename] = "".join(lines)
    filename = "policy.py"
    source[filename] = replace_once(
        source[filename],
        "    scale_page_coalescing: bool = False\n",
        "    scale_page_coalescing: bool = False\n    leaf_scale_layout: bool = False\n",
    )
    source[filename] = replace_once(
        source[filename],
        '            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,\n',
        '            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,\n'
        '            "LEAF_SCALE_LAYOUT": self.leaf_scale_layout,\n',
    )
    source[filename] = replace_once(
        source[filename],
        "    return CSASPMDPolicy(\n",
        "    leaf_scale_layout = (long_history and group == 6\n"
        "                         and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)\n"
        "    return CSASPMDPolicy(\n",
    )
    source[filename] = replace_once(
        source[filename],
        "        scale_page_coalescing=(long_history and group == 6 "
        "and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS)),\n",
        "        scale_page_coalescing=(leaf_scale_layout or (long_history and group == 6\n"
        "                               and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS))),\n"
        "        leaf_scale_layout=leaf_scale_layout,\n",
    )
    target = here / "runs/leaf_scale_guard_v1"
    target.mkdir(exist_ok=False)
    for name, content in source.items():
        ast.parse(content)
        (target / name).write_text(content)
    select = runpy.run_path(str(target / "policy.py"))["select_policy"]
    decisions = {}
    for cid, record in baseline["cases"].items():
        case = record["case"]
        policy = select(
            tokens=case["token_capacity"],
            requests=case["request_capacity"],
            max_query_tokens=max(case["query_lengths"]),
            max_seq_len=max(case["seq_lens"]),
        ).constexprs()
        original = json.loads(Path(record["source_report"]).read_text())
        old_policy = original["policies"]["spmd"]
        active = cid in ("C02", "C04")
        assert policy["LEAF_SCALE_LAYOUT"] == active
        assert {k: v for k, v in policy.items() if k not in ("LEAF_SCALE_LAYOUT", "SCALE_PAGE_COALESCING")} == {
            k: v for k, v in old_policy.items() if k != "SCALE_PAGE_COALESCING"
        }
        assert policy["SCALE_PAGE_COALESCING"] == (active or old_policy["SCALE_PAGE_COALESCING"])
        decisions[cid] = policy
        if cid in ("C04", "C05"):
            buffers = original["descriptors"]["spmd"]
            if active and not old_policy["SCALE_PAGE_COALESCING"]:
                buffers["group_offsets"]["shape"][0] += 2 * case["request_capacity"]
            description = dict(buffers=buffers, policy=policy)
            (here / f"descriptors/{cid}_leaf_scale_guard_v1.json").write_text(json.dumps(description, indent=2) + "\n")
    report = dict(
        source=str(target),
        baseline="baseline.json",
        parent="leaf_scale_one_row_service_v1",
        adopted=False,
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        guard="long G6 and 64<=token_capacity<=96; actual leaf metadata still validated on device",
        reason="C04 gain; C05 no gain. Retain original META proof and scale policy outside the existing interval",
        changed_files=[
            n for n, s in source.items() if hashlib.sha256(s.encode()).hexdigest() != baseline["files"].get(n)
        ],
        host_source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        eight_case_policies=decisions,
    )
    with (here / "leaf_scale_guard_v1.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
