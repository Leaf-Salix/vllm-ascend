# SPDX-License-Identifier: Apache-2.0
"""按用户授权，为长历史B1联动选择G6、叶计划、布局证明及单根归并。"""

import argparse
import ast
import json
import shutil

from prepare import HERE
from prepare_query_ready import replace_once


def migrate(indexer, entry):
    old = "if pl.tensor.dim(position_ids, 0) >= LONG_S6_MIN_QUERY_ROWS:"
    if indexer.count(old) != 2:
        raise ValueError("Score选择与单根归并必须同时找到")
    lines = []
    for line in indexer.splitlines(keepends=True):
        if line.strip() == old:
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(indent + "long_s6_group = indexer_use_long_s6(pl.tensor.dim(position_ids, 0))\n")
            lines.append(indent + "if long_s6_group == 1:\n")
        else:
            lines.append(line)
    indexer = "".join(lines)
    indexer = replace_once(
        indexer,
        "        if query_count >= LONG_S6_MIN_QUERY_ROWS:\n",
        "        merge_s6_group = indexer_use_long_s6(query_count)\n        if merge_s6_group == 1:\n",
    )
    marker = "@pl.jit.inline\ndef indexer_long_leaf_plan("
    indexer = replace_once(
        indexer,
        marker,
        "@pl.jit.inline\n"
        "def indexer_use_long_s6(query_count: pl.Scalar[pl.INDEX]):\n"
        "    use_group = pl.cast(0, pl.INT32)\n"
        "    if query_count >= LONG_S6_MIN_QUERY_ROWS:\n"
        "        use_group = pl.cast(1, pl.INT32)\n"
        "    if query_count == S:\n"
        "        use_group = pl.cast(1, pl.INT32)\n"
        "    return use_group\n\n\n" + marker,
    )
    indexer = replace_once(
        indexer,
        "    if 3 * query_groups == 2 * TOPK_SCORE_WORKERS:  # noqa: SIM114 - explicit scalar branches for PyPTO\n",
        "    # 用户允许B1专门策略：整请求复用Key，增加独立叶覆盖24个Cube worker。\n"
        "    if query_groups == 1:\n"
        "        leaf_multiple = TOPK_SCORE_WORKERS\n"
        "    elif 3 * query_groups == 2 * TOPK_SCORE_WORKERS:  # noqa: SIM114 - PyPTO Scalar branches\n",
    )
    indexer = replace_once(
        indexer,
        "        balanced_count = (leaf_count + leaf_multiple - 1) // leaf_multiple * leaf_multiple\n",
        "        balanced_count = (leaf_count + leaf_multiple - 1) // leaf_multiple * leaf_multiple\n"
        "        if query_groups == 1:\n"
        "            # 至少一叶避免空历史除零，也不创建超过实际1024列段数的叶。\n"
        "            balanced_count = pl.min(balanced_count, pl.max(tile_count, 1))\n",
    )
    marker = 'name_hint="indexer_scale_layout"'
    before = entry[: entry.index(marker)]
    after = entry[entry.index(marker) :]
    before = replace_once(
        before,
        "            if t_dim >= LONG_S6_MIN_QUERY_ROWS:\n",
        "            early_s6_group = indexer_use_long_s6(t_dim)\n            if early_s6_group == 1:\n",
    )
    entry = before + after
    imports = [line for line in entry.splitlines() if line.startswith("from .decode_indexer import ")]
    if len(imports) != 1:
        raise ValueError("Indexer入口import不唯一")
    entry = replace_once(
        entry,
        imports[0] + "\n",
        "from .decode_indexer import LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF, indexer, indexer_use_long_s6\n",
    )
    ast.parse(indexer)
    ast.parse(entry)
    return indexer, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="b1_request_score_v2")
    args = parser.parse_args()
    if not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    base = HERE / "sources/hc_post_workers_v1"
    target = HERE / "sources" / args.candidate
    indexer, entry = migrate((base / "decode_indexer.py").read_text(), (base / "decode_csa.py").read_text())
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(indexer)
    (target / "decode_csa.py").write_text(entry)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base="hc_post_workers_v1",
                donor="13259f92及当前已验证长G6分支的资源适配；用户2026-10-21明确允许B1专门if",
                changed_files=["decode_indexer.py", "decode_csa.py"],
                policy="只在长历史T=S=6选择G6；128K用24叶/24个AIC而非G2的12个有工作AIC",
                linked_changes="Score分组、单根归并、merge叶计划、设备scale布局证明统一条件",
                preserved="G6既有核内算术、N64、原短前缀helper、原QA策略、其它batch与短历史路径",
                numerical_check="G2到G6的算术分组改变必须由完整逐元素检查验证，未先称bit相同",
                status="private_candidate_no_device_evidence",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
