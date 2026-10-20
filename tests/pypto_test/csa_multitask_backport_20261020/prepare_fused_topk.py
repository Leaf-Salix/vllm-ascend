# SPDX-License-Identifier: Apache-2.0
"""迁移5da1bf0c的真实根生产者等待，将长G6 Score与Top-K合为一个MIX。"""

import argparse
import ast
import json
import re
import shutil
import textwrap

from prepare import HERE, section
from prepare_scale_early import add_calls

SIGNALS = '''
# 与已验证单SPMD相同，512B独占一个通知槽，48个AIV生产者。
SCORE_SIGNAL_SLOT_WORDS = 128
SCORE_SIGNAL_WORDS = TOPK_SCORE_WORKERS * 2 * SCORE_SIGNAL_SLOT_WORDS


@pl.jit.inline(auto_scope=False)
def indexer_fused_topk_finish(
    flags: pl.Tensor[[SCORE_SIGNAL_WORDS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_idxs: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    worker: pl.Scalar[pl.INDEX],
):
    """同一24组MIX内，先发布全部本人根，再按实际生产者等待和归并。

    每个AIV调用前完成本人全部Score根；48个参与者都先发布、后等待，等待图
    不包含后续生产工作。512B槽独占，前置证明task每次清零；只在当前MIX
    使用，生命周期由TMR的Tensor依赖保持，不跨调用假定上次信号有效。
    """
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    pl.system.bar_all()
    pl.system.signal_notify(flags, [worker * SCORE_SIGNAL_SLOT_WORDS], pl.cast(1, pl.INT32))
    query_count = pl.tensor.dim(position_ids, 0)
    query_groups = query_count // S
    max_cache_count = 0
    for batch in pl.range(query_groups):
        max_cache_count = pl.max(max_cache_count, pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO)
    all_leaves, leaf_tiles, extra_leaves = indexer_long_leaf_plan(
        pl.min(max_cache_count, TOPK_MAX_CANDIDATES), query_groups,
    )
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        group = query // S
        local_query = query % S
        position = pl.read(position_ids, [query])
        cache_len = pl.read(kv_seq_lens, [group]) // COMPRESS_RATIO
        visible = pl.min(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES)
        if visible > 0:
            # 与indexer_topk_query_merge_one实际读取的根数完全相同。
            tiles = (visible + BUFFERED_LONG_SCORE_TILE - 1) // BUFFERED_LONG_SCORE_TILE
            larger_count = pl.min(extra_leaves, (tiles + leaf_tiles) // (leaf_tiles + 1))
            remaining = pl.max(tiles - extra_leaves * (leaf_tiles + 1), 0)
            roots = larger_count + (remaining + leaf_tiles - 1) // leaf_tiles
            for leaf in pl.range(roots):
                # 5da1bf0c：小组数用leaf-major，其余用group-major。
                item = group * all_leaves + leaf
                if query_groups < TOPK_SCORE_WORKERS:
                    item = leaf * query_groups + group
                producer = (item % TOPK_SCORE_WORKERS) * 2 + local_query // (S // 2)
                pl.system.signal_wait(
                    flags, [producer * SCORE_SIGNAL_SLOT_WORDS], pl.cast(1, pl.INT32),
                )
            pl.system.fence()
            pl.system.cacheinvalid()
            pl.system.fence()
        indexer_topk_query_merge_one(
            query, position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs,
            True, leaf_tiles, extra_leaves,
        )


'''


def add_parameter(source, name, parameters):
    begin = source.index("def " + name + "(")
    end = source.index("\n)", begin)
    return source[:end] + "\n" + parameters.rstrip() + source[end:]


def migrate_indexer(source):
    marker = "@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_native_cube("
    if source.count(marker) != 1:
        raise AssertionError("需要唯一Score MIX")
    source = source.replace(marker, SIGNALS + marker, 1)
    native = "indexer_score_topk_native_cube"
    source = add_parameter(
        source,
        native,
        (
            "    fused_topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],\n"
            "    fused_topk_idxs: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],\n"
            "    score_flags: pl.Tensor[[SCORE_SIGNAL_WORDS], pl.INT32],\n"
        ),
    )
    source = add_calls(source, (native,), ("topk_scores", "topk_idxs", "score_flags"))
    for name in ("indexer_score_topk_forest", "indexer_weights_score", "indexer"):
        source = add_parameter(source, name, "    score_flags: pl.Tensor[[SCORE_SIGNAL_WORDS], pl.INT32],\n")
    source = add_calls(source, ("indexer_score_topk_forest", "indexer_weights_score"), ("score_flags",))
    finish = """
            if balance_leaves:
                indexer_fused_topk_finish(
                    score_flags, position_ids, kv_seq_lens, pair_arena,
                    fused_topk_scores, fused_topk_idxs, buf_worker * 2 + buf_score_lane,
                )
"""
    marker = "    return buffered_leaf_tid\n"
    if source.count(marker) != 1:
        raise AssertionError("需要唯一Score返回位置")
    source = source.replace(marker, finish + "\n" + marker, 1)
    begin = source.index("def indexer_score_topk_forest(")
    tail = section(source[begin:], "    with pl.scope():\n", "    return topk_scores, topk_idxs, score_tid")
    condition = (
        "    topk_already_fused = pl.cast(0, pl.INT32)\n"
        "    if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:\n"
        "        if pl.tensor.dim(position_ids, 0) >= LONG_S6_MIN_QUERY_ROWS:\n"
        "            topk_already_fused = pl.cast(1, pl.INT32)\n"
        "    if topk_already_fused == 0:\n"
    )
    source = source[:begin] + source[begin:].replace(tail, condition + textwrap.indent(tail, "    "), 1)
    ast.parse(source)
    return source


def migrate_entry(source):
    source = source.replace(
        "from .decode_indexer import indexer, LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF",
        "from .decode_indexer import indexer, LONG_S6_MIN_QUERY_ROWS, TOPK_CANDIDATES_PER_LEAF, "
        "SCORE_SIGNAL_WORDS, SCORE_SIGNAL_SLOT_WORDS, TOPK_SCORE_WORKERS",
        1,
    )
    marker = "        early_scale_layout = pl.create_tensor([SCALE_LAYOUT_CAPACITY], dtype=pl.INT32)\n"
    source = source.replace(
        marker,
        marker
        + (
            "        score_flags_2d = pl.create_tensor([TOPK_SCORE_WORKERS * 2, SCORE_SIGNAL_SLOT_WORDS], "
            "dtype=pl.INT32)\n"
            "        score_flags = pl.reshape(score_flags_2d, [SCORE_SIGNAL_WORDS])\n"
        ),
        1,
    )
    marker = "                    early_scale_layout = prove_scale_layout(\n"
    source = source.replace(
        marker,
        (
            "                    # 每次调用初始化；Score显式等待本task及原有cache/系数生产者。\n"
            "                    score_flags_2d = pl.store(\n"
            "                        pl.tile.full([TOPK_SCORE_WORKERS * 2, SCORE_SIGNAL_SLOT_WORDS], "
            "dtype=pl.INT32, value=0),\n"
            "                        [0, 0], score_flags_2d,\n"
            "                    )\n"
        )
        + marker,
        1,
    )
    source = add_calls(source, ("indexer",), ("score_flags",))
    # 两路径均原位写外部idx_topk；避免条件分支的返回别名在AICPU中越作用域。
    source = source.replace("_index_scores, index_indices = indexer(", "indexer(", 1)
    source = re.sub(r"\bindex_indices\b", "idx_topk", source)
    before = "            # Bound indexer scratch to its own runtime scope.\n"
    source = source.replace(
        before,
        (
            "            # 保存外部连续ND输出的零拷贝GM描述符，避免inner scope返回别名逃逸。\n"
            "            stable_topk_flat = pl.reshape(idx_topk, [t_dim * IDX_TOPK])\n"
        )
        + before,
        1,
    )
    after = "            # sparse_attn_csa folds"
    source = source.replace(
        after, ("            stable_topk = pl.reshape(stable_topk_flat, [t_dim, IDX_TOPK])\n\n") + after, 1
    )
    consumer = r"^ {16}idx_topk,\n"
    if len(re.findall(consumer, source, flags=re.MULTILINE)) != 1:
        raise AssertionError("需要唯一Attention Top-K消费者")
    source = re.sub(consumer, "                stable_topk,\n", source, count=1, flags=re.MULTILINE)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="core_v2")
    parser.add_argument("--candidate", default="fused_topk")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    for name, transform in (("decode_indexer.py", migrate_indexer), ("decode_csa.py", migrate_entry)):
        path = target / name
        path.write_text(transform(path.read_text()))
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                source="5da1bf0c真实根依赖；13259f92通知及数值体",
                changed_files=["decode_csa.py", "decode_indexer.py"],
                lifecycle="每调用前置清零；同一既有24组MIX内每AIV先发布所有根、再等待真实生产者；Task完成后释放",
                scope="仅长G6 balanced，短G2/G6及小T保留分离TopK；完整CSA仍多task",
                preserved="Score数学、FFTS双槽、sync_start与allow_early_resolve设置不变",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
