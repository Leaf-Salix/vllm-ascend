# SPDX-License-Identifier: Apache-2.0
"""B1并行化原半叶，保留原排序分支，而不是重定义固定4K叶的同分语义。

旧8192叶最后不足一叶时，两个半叶的边界按ceil(valid/1024)*512计算。
将每个原半叶独立交给AIC；AIV按query分工，复用原indexer_topk_half_leaf。
保留UB累计根方案。旧失败候选和证据均留在原目录，不覆盖。
"""

import ast
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"上下文改变：{old[:110]}")
    return text.replace(old, new)


def main():
    here = Path(__file__).resolve().parent
    target = here / "source_b1_legacy_half"
    shutil.copytree(here / "source_b1_root_resident", target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = target / "indexer_group_stages.py"
    text = file.read_text()
    old = """                # B1策略：原8192叶的前后4096半叶变成独立AIC工作项。
                # 根顺序不变；最大历史和有效组数仍来自本轮device输入。
                leaf_tiles = pl.cast(CANDIDATES_PER_LEAF // (2 * SCORE_TILE), pl.INDEX)
                leaves = pl.max((max_history + CANDIDATES_PER_LEAF // 2 - 1)
                                // (CANDIDATES_PER_LEAF // 2), 1)"""
    new = """                # B1把原每个8192叶的两个半叶独立调度，包含尾部空根。
                # 末叶边界依赖各组实际visible，不能把全部半叶静态等分为4096。
                leaf_tiles = pl.cast(CANDIDATES_PER_LEAF // (2 * SCORE_TILE), pl.INDEX)
                leaves = 2 * pl.max((max_history + CANDIDATES_PER_LEAF - 1)
                                    // CANDIDATES_PER_LEAF, 1)"""
    text = replace_once(text, old, new)
    text = replace_once(
        text,
        "    visible: pl.Scalar[pl.INDEX],\n"
        "):\n"
        "    \"\"\"先分1024候选的整tile，再按组/单query实际visible截断；返回begin/capacity/valid。\"\"\"",
        "    visible: pl.Scalar[pl.INDEX],\n"
        "    BALANCED: pl.constexpr,\n"
        "    GROUP_SIZE: pl.constexpr,\n"
        "):\n"
        "    \"\"\"普通叶沿1024分片；B1逐项复现原末叶半跨度，再返回begin/capacity/valid。\"\"\"",
    )
    text = replace_once(
        text,
        "    begin = (leaf * leaf_tiles + pl.min(leaf, extra)) * SCORE_TILE\n"
        "    capacity = (leaf_tiles + pl.cast(leaf < extra, pl.INDEX)) * SCORE_TILE\n"
        "    valid = pl.max(pl.min(visible - begin, capacity), 0)",
        "    if BALANCED and GROUP_SIZE == 2:\n"
        "        original_begin = (leaf // 2) * CANDIDATES_PER_LEAF\n"
        "        original_valid = pl.max(pl.min(visible - original_begin, CANDIDATES_PER_LEAF), 0)\n"
        "        original_tiles = (original_valid + SCORE_TILE - 1) // SCORE_TILE\n"
        "        capacity = pl.min(original_tiles * (SCORE_TILE // 2), CANDIDATES_PER_LEAF // 2)\n"
        "        begin = original_begin + (leaf % 2) * capacity\n"
        "    else:\n"
        "        begin = (leaf * leaf_tiles + pl.min(leaf, extra)) * SCORE_TILE\n"
        "        capacity = (leaf_tiles + pl.cast(leaf < extra, pl.INDEX)) * SCORE_TILE\n"
        "    valid = pl.max(pl.min(visible - begin, capacity), 0)",
    )
    ast.parse(text)
    file.write_text(text)

    file = target / "indexer_score_stages.py"
    text = file.read_text()
    call = "            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count\n"
    if text.count(call) != 3:
        raise ValueError("预期Cube两路径及Vector共三个范围调用")
    text = text.replace(
        call, "            buf_leaf, buf_leaf_tiles, buf_extra_leaves, buf_visible_count,\n"
              "            balance_leaves, query_group_size,\n"
    )
    old = """                        single_segments = pl.tile.assemble(
                            single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]
                        )
                    if single_step % 2 == 1 or single_step + 1 == single_steps:"""
    new = """                        if query_group_size == 2:
                            # 复用旧score scratch，完整保留原半叶排序分支及同分顺序。
                            # 每个AIV只负责一个query；row与其它worker/lane不相交。
                            legacy_score_row = buf_worker * 2 * query_group_size + buf_score_lane * 3
                            pl.store(single_scaled, [legacy_score_row, single_step * score_tile], score_arena)
                        else:
                            single_segments = pl.tile.assemble(
                                single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]
                            )
                    if query_group_size != 2 and (single_step % 2 == 1 or single_step + 1 == single_steps):"""
    text = replace_once(text, old, new)
    old = """                        single_root = pl.tile.extract(
                            single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                        )
                        pl.store(
                            single_root,
                            [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0],
                            pair_arena,
                        )"""
    new = """                        if query_group_size == 2:
                            legacy_visible, _legacy_causal = indexer_query_visibility(
                                bounds, kv_seq_lens, position_ids, single_batch, single_query + single_query_lane
                            )
                            legacy_valid = pl.max(pl.min(legacy_visible - single_begin, single_capacity), 0)
                            legacy_slot = (single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf
                            if legacy_valid > 0:
                                indexer_topk_half_leaf(
                                    score_arena, pair_arena,
                                    buf_worker * 2 * query_group_size + buf_score_lane * 3,
                                    single_begin, legacy_valid, legacy_slot,
                                )
                            else:
                                pl.store(pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),
                                         [legacy_slot, 0], pair_arena)
                        else:
                            single_root = pl.tile.extract(
                                single_roots, single_local, 0, [1, TOPK_PAIR_WIDTH], target_memory=pl.MemorySpace.Vec
                            )
                            pl.store(
                                single_root,
                                [(single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf, 0],
                                pair_arena,
                            )"""
    text = replace_once(text, old, new)
    text = replace_once(
        text, "    return score_arena, pair_arena\n",
        "        elif balance_leaves and query_group_size == 2:\n"
        "            # 尾部空半叶不触发Cube/Vector交接，但仍发布原布局的空根。\n"
        "            # query根计数仍包含它，不能让消费者读到上一轮内容。\n"
        "            if buf_score_lane < buf_live:\n"
        "                empty_slot = (buf_query + buf_score_lane) * TOPK_ROWS_PER_QUERY + buf_leaf\n"
        "                pl.store(pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),\n"
        "                         [empty_slot, 0], pair_arena)\n"
        "    return score_arena, pair_arena\n",
    )
    ast.parse(text)
    file.write_text(text)

    file = target / "indexer_publish_stages.py"
    text = replace_once(
        file.read_text(),
        "                    if BALANCED:\n"
        "                        #",
        "                    if BALANCED and pl.read(plan, [5]) == 2:\n"
        "                        # B1独立调度原半叶，保留原2*ceil(visible/8192)个根。\n"
        "                        # 包含末叶的空后半根，叶内与最终归并顺序均不改变。\n"
        "                        if leaf_count > 0 and leaf_count * 2 <= leaves:\n"
        "                            roots = leaf_count * 2\n"
        "                    elif BALANCED:\n"
        "                        #",
    )
    ast.parse(text)
    file.write_text(text)
    file = target / "policy.py"
    text = file.read_text().replace(
        "    # 保留动态Q/L和4096半叶边界；较大容量及S1维持原策略。",
        "    # 保留动态Q/L及原半叶边界（末叶可小于4096）；较大容量及S1维持原策略。",
    )
    file.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
