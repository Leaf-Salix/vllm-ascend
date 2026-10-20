# SPDX-License-Identifier: Apache-2.0
"""迁移e91e9f17的四token Query Hadamard，与已有20核Q_B保留多task并行。"""

import argparse
import ast
import json
import re
import shutil
import textwrap

from prepare import HERE, section
from prepare_fused_topk import add_parameter
from prepare_scale_early import add_calls


def migrate(source, donor, long_domain=False, unified_dispatch=False):
    original_kernel = section(
        source[source.index("def indexer_qr_hadamard_mm(") :],
        "    with pl.spmd(\n        QH_WORKERS,",
        "    with pl.spmd(\n        QH_QUANT_WORKERS,",
    )
    tree = ast.parse(donor)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_query_hadamard_four_tokens")
    lines = donor.splitlines(keepends=True)
    begin = next(n.lineno for n in fn.body if isinstance(n, ast.Assign))
    body = textwrap.dedent("".join(lines[begin - 1 : fn.body[-1].lineno - 1]))
    for old, new in (
        ("tokens", "bs"),
        ("query_rope", "qr_bf16"),
        ("query_had", "qh_acc_gm"),
        ("worker", "qh_worker"),
        ("WORKERS", "QH_COOPERATIVE_WORKERS"),
    ):
        body = re.sub(r"\b" + old + r"\b", new, body)
    start = source.index("def indexer_qr_hadamard_mm(")
    finish = source.index("    with pl.spmd(\n        QH_QUANT_WORKERS,", start)
    old = source[start:finish]
    old = old.replace("        QH_WORKERS,\n", "        QH_COOPERATIVE_WORKERS,\n", 1)
    begin = old.index("        # Shared Hadamard matrix.")
    old = old[:begin] + textwrap.indent(body, "        ") + "\n"
    source = source[:start] + old + source[finish:]
    marker = "QH_WORKERS = 24  # Query Hadamard matmul workers\n"
    if source.count(marker) != 1:
        raise AssertionError("需要唯一Query Hadamard核数声明")
    source = source.replace(
        marker,
        marker + "\n# e91e9f17：M256覆盖四token；为已有20核Q_B保留并行资源。\nQH_COOPERATIVE_WORKERS = 4\n",
    )
    if long_domain:
        source = source.replace(
            "QH_COOPERATIVE_WORKERS = 4\n",
            "QH_COOPERATIVE_WORKERS = 4\n"
            "# 至少两轮四token任务才采用；短历史Score链保留原24核M64。\n"
            "QH_COOPERATIVE_MIN_ROWS = QH_COOPERATIVE_WORKERS * 4 * 2\n",
            1,
        )
        for name in ("indexer_qr_hadamard_mm", "indexer_qr_hadamard"):
            source = add_parameter(source, name, "    host_max_seq_len: pl.Scalar[pl.INT32],\n")
        source = add_calls(source, ("indexer_qr_hadamard_mm", "indexer_qr_hadamard"), ("host_max_seq_len",))
        start = source.index("def indexer_qr_hadamard_mm(")
        end = source.index("    with pl.spmd(\n        QH_QUANT_WORKERS,", start)
        body = source[start:end]
        cooperative = body[body.index("    with pl.spmd(") :]
        selection = (
            "    use_cooperative_hadamard = pl.cast(0, pl.INT32)\n"
            "    if host_max_seq_len // COMPRESS_RATIO > TOPK_CANDIDATES_PER_LEAF:\n"
            "        if bs >= QH_COOPERATIVE_MIN_ROWS:\n"
            "            use_cooperative_hadamard = pl.cast(1, pl.INT32)\n"
        )
        if unified_dispatch:
            worker_line = "        qh_worker = pl.tile.get_block_idx()\n"
            old_math = original_kernel[original_kernel.index(worker_line) + len(worker_line) :]
            new_math = cooperative[cooperative.index(worker_line) + len(worker_line) :]
            replacement = (
                selection
                + "    qh_workers = pl.cast(QH_WORKERS, pl.INDEX)\n"
                + "    if use_cooperative_hadamard != 0:\n"
                + "        qh_workers = pl.cast(QH_COOPERATIVE_WORKERS, pl.INDEX)\n"
                + '    with pl.spmd(qh_workers, name_hint="qr_hadamard_matmul",\n'
                + "                 deps=[qh_mm_dep], allow_early_resolve=True) as qh_mm_tid:\n"
                + worker_line
                + "        if use_cooperative_hadamard != 0:\n"
                + textwrap.indent(new_math, "    ")
                + "        else:\n"
                + textwrap.indent(old_math, "    ")
            )
        else:
            replacement = (
                selection
                + "    qh_branch_tids = pl.array.create(1, pl.TASK_ID)\n"
                + "    qh_branch_tids[0] = pl.system.task_invalid()\n"
                + "    if use_cooperative_hadamard != 0:\n"
                + textwrap.indent(cooperative + "    qh_branch_tids[0] = qh_mm_tid\n", "    ")
                + "    else:\n"
                + textwrap.indent(original_kernel + "    qh_branch_tids[0] = qh_mm_tid\n", "    ")
            )
        body = body.replace(
            cooperative,
            replacement,
            1,
        )
        source = source[:start] + body + source[end:]
        if not unified_dispatch:
            source = source.replace(
                "    return qh_mm_tid, qh_quant_tid\n",
                "    return qh_branch_tids[0], qh_quant_tid\n",
                1,
            )
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="query_hadamard4")
    parser.add_argument("--long-domain", action="store_true", help="只在长历史且至少两轮M256工作量采用")
    parser.add_argument("--unified-dispatch", action="store_true", help="一个SPMD入口，按策略选核数与核内M块")
    args = parser.parse_args()
    if args.unified_dispatch and not args.long_domain:
        raise ValueError("统一分派仅用于长短资源分支")
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "decode_indexer.py"
    donor = (HERE / "sources/spmd/projection_overlap_stages.py").read_text()
    path.write_text(migrate(path.read_text(), donor, args.long_domain, args.unified_dispatch))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="e91e9f17",
                long_domain=args.long_domain,
                unified_dispatch=args.unified_dispatch,
                changed_files=["decode_indexer.py"],
                scope="Query Hadamard四token M256与4核分工；实际T尾行、K128、FP32结果及量化不变",
                scheduling="保留多task依赖/early_resolve；Q_B原为20核；compact Hadamard原为独立单核，先不变",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
