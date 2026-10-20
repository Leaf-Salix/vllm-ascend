# SPDX-License-Identifier: Apache-2.0
"""派生单请求G2历史分片候选；保持旧半叶边界和逐根归并顺序。"""

import ast
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"上下文改变：{old[:90]}")
    return text.replace(old, new)


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_b1_leaf4k"
    shutil.copytree(here / "source_after_wait150", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = target / "policy.py"
    text = file.read_text()
    text = replace_once(
        text, "    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128",
        "    # 用户允许单请求独立分档。仅改变已有G2的历史工作分配，\n"
        "    # 保留动态Q/L和4096半叶边界；较大容量及S1维持原策略。\n"
        "    single_request_leaves = (\n"
        "        requests == 1 and long_history and cache_rows <= 65536\n"
        "        and 5 <= max_query_tokens <= 6\n"
        "    )\n"
        "    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128",
    )
    text = replace_once(
        text, "balanced=long_history and group == 6,",
        "balanced=(long_history and group == 6) or single_request_leaves,",
    )
    text = replace_once(
        text, "roots=1 if long_history and group == 6 else 2,",
        "roots=1 if (long_history and group == 6) or single_request_leaves else 2,",
    )
    ast.parse(text)
    file.write_text(text)
    file = target / "indexer_group_stages.py"
    text = replace_once(
        file.read_text(),
        "        if BALANCED:\n"
        "            leaves, leaf_tiles, extra = indexer_long_leaf_plan(max_history, groups)",
        "        if BALANCED:\n"
        "            if GROUP_SIZE == 2:\n"
        "                # B1策略：原8192叶的前后4096半叶变成独立AIC工作项。\n"
        "                # 根顺序不变；最大历史和有效组数仍来自本轮device输入。\n"
        "                leaf_tiles = pl.cast(CANDIDATES_PER_LEAF // (2 * SCORE_TILE), pl.INDEX)\n"
        "                leaves = pl.max((max_history + CANDIDATES_PER_LEAF // 2 - 1)\n"
        "                                // (CANDIDATES_PER_LEAF // 2), 1)\n"
        "            else:\n"
        "                leaves, leaf_tiles, extra = indexer_long_leaf_plan(max_history, groups)",
    )
    ast.parse(text)
    file.write_text(text)
    file = target / "indexer_publish_stages.py"
    text = replace_once(
        file.read_text(),
        "            if BALANCED:\n"
        "                # 多路归并helper保持incoming3/2/1/root顺序及最后不足四路分支。",
        "            if BALANCED and QUERY_GROUP != 2:\n"
        "                # G2独立半叶仍走下方原顺序左折叠，保持同分规则。\n"
        "                # 多路归并helper保持incoming3/2/1/root顺序及最后不足四路分支。",
    )
    ast.parse(text)
    file.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
