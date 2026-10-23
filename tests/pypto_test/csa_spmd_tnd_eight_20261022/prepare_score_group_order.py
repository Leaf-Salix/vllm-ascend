# SPDX-License-Identifier: Apache-2.0
"""Freeze dynamic heavy/light group ordering with matching TopK root owners."""

import ast
import hashlib
import json
from pathlib import Path

from prepare_scale_gather import replace_once


def main():
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "baseline.json").read_text())["files"]
    source = {p.name: p.read_text() for p in (here / "runs/baseline").glob("*.py")}
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()} == expected
    source["score_group_order_stages.py"] = (here / "score_group_order_source.py").read_text()
    name = "indexer_group_stages.py"
    source[name] = replace_once(
        source[name],
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .score_group_order_stages import plan_score_group_order\n",
    )
    source[name] = replace_once(
        source[name],
        "        pl.write(plan, [0], pl.cast(groups, pl.INT32))\n",
        "        if BALANCED and GROUP_SIZE == 6:\n"
        "            offsets = plan_score_group_order(offsets, groups, requests, tokens, leaves)\n"
        "        pl.write(plan, [0], pl.cast(groups, pl.INT32))\n",
    )
    name = "indexer_score_stages.py"
    source[name] = replace_once(
        source[name],
        "import pypto.language as pl\n",
        "import pypto.language as pl\n\nfrom .score_group_order_stages import active_score_order_base\n",
    )
    loop = "    for buf_item in pl.range(buf_worker, groups * buf_max_leaves, TOPK_SCORE_WORKERS):\n"
    assert source[name].count(loop) == 3
    source[name] = source[name].replace(
        loop,
        "    buf_order_base = pl.cast(-1, pl.INDEX)\n"
        "    if balance_leaves and query_group_size == 6:\n"
        "        buf_order_base = active_score_order_base(\n"
        "            offsets, groups, pl.tensor.dim(kv_seq_lens, 0), pl.tensor.dim(position_ids, 0)\n"
        "        )\n" + loop,
    )
    decode = "        buf_group, buf_leaf = indexer_work_item(buf_item, groups, buf_max_leaves)\n"
    assert source[name].count(decode) == 3
    source[name] = source[name].replace(
        decode,
        decode + "        if buf_order_base >= 0:\n"
        "            buf_group = pl.cast(pl.read(offsets, [buf_order_base + 1 + buf_group]), pl.INDEX)\n",
    )
    name = "attention_dependencies.py"
    source[name] = replace_once(
        source[name],
        "from .indexer_group_stages import PLAN_WORDS\n",
        "from .indexer_group_stages import PLAN_WORDS, OFFSET_ROWS_DYN\n"
        "from .score_group_order_stages import active_score_order_base\n",
    )
    source[name] = replace_once(
        source[name],
        "    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],\n",
        "    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],\n"
        "    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],\n"
        "    requests: pl.Scalar[pl.INDEX],\n"
        "    tokens: pl.Scalar[pl.INDEX],\n",
    )
    source[name] = replace_once(
        source[name],
        "    for root in pl.range(root_count):\n",
        "    schedule_group = group\n"
        "    if BALANCED and QUERY_GROUP == 6:\n"
        "        order_base = active_score_order_base(offsets, groups, requests, tokens)\n"
        "        if order_base >= 0:\n"
        "            schedule_group = pl.cast(pl.read(offsets, [order_base + 1 + groups + group]), pl.INDEX)\n"
        "    for root in pl.range(root_count):\n",
    )
    source[name] = replace_once(
        source[name],
        "            item = leaf * groups + group\n",
        "            item = leaf * groups + schedule_group\n",
    )
    name = "indexer_publish_stages.py"
    source[name] = replace_once(
        source[name],
        "            signals = wait_indexer_query_roots(\n                signals,\n                plan,\n",
        "            signals = wait_indexer_query_roots(\n                signals,\n                plan,\n"
        "                offsets,\n                pl.tensor.dim(lengths, 0),\n                tokens,\n",
    )
    target = here / "runs/score_group_order_v2"
    target.mkdir(exist_ok=False)
    for name, text in source.items():
        ast.parse(text)
        (target / name).write_text(text)
    report = dict(
        source=str(target),
        baseline="baseline.json",
        adopted=False,
        device_execution=False,
        changes=[n for n, s in source.items() if hashlib.sha256(s.encode()).hexdigest() != expected.get(n)],
        files={n: hashlib.sha256(s.encode()).hexdigest() for n, s in source.items()},
        preserved=[
            "Native ABI and allocated workspace sizes",
            "logical leaves and merge tree",
            "Score FIFO slots and event counts",
            "matching inverse root dependencies",
        ],
        scope=(
            "Existing 16-group balanced geometry; actual live halves and nearly equal visible histories; "
            "otherwise identity"
        ),
        hypothesis="Reduce late heavy/heavy work pairs without adding a task or barrier; full-chain gain unproven",
    )
    with (here / "score_group_order_v2.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
