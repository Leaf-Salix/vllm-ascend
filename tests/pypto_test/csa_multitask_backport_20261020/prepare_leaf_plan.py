# SPDX-License-Identifier: Apache-2.0
"""复用入口最大长度，在多task编排侧一次生成Score/Top-K公共叶计划。"""

import argparse
import ast
import json
import shutil

from prepare import HERE, section
from prepare_fused_topk import add_parameter
from prepare_scale_early import add_calls


def migrate_indexer(source):
    cube_name = "indexer_score_topk_native_cube"
    cube_parameters = (
        "    common_max_leaves: pl.Scalar[pl.INDEX],\n"
        "    balanced_max_leaves: pl.Scalar[pl.INDEX],\n"
        "    common_leaf_tiles: pl.Scalar[pl.INDEX],\n"
        "    common_extra_leaves: pl.Scalar[pl.INDEX],\n"
    )
    source = add_parameter(source, cube_name, cube_parameters)
    source = add_calls(
        source,
        (cube_name,),
        ("common_max_leaves", "balanced_max_leaves", "common_leaf_tiles", "common_extra_leaves"),
    )
    cube_begin = source.index("def " + cube_name + "(")
    cube_end = source.index("@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_forest(", cube_begin)
    cube = source[cube_begin:cube_end]
    old = section(cube, "        buf_max_cache_len = 0\n", "        pl.system.set_ffts(buf_score_ffts)\n")
    cube = cube.replace(
        old,
        "        buf_max_leaves = common_max_leaves\n"
        "        buf_leaf_tiles = common_leaf_tiles\n"
        "        buf_extra_leaves = common_extra_leaves\n"
        "        if balance_leaves:\n"
        "            buf_max_leaves = balanced_max_leaves\n",
        1,
    )
    source = source[:cube_begin] + cube + source[cube_end:]

    merge_name = "indexer_topk_query_merge"
    source = add_parameter(
        source,
        merge_name,
        "    common_leaf_tiles: pl.Scalar[pl.INDEX],\n    common_extra_leaves: pl.Scalar[pl.INDEX],\n",
    )
    source = add_calls(source, (merge_name,), ("common_leaf_tiles", "common_extra_leaves"))
    merge_begin = source.index("def " + merge_name + "(")
    merge_end = source.index("@pl.jit.inline\ndef indexer_topk_leaf_publish(", merge_begin)
    merge = source[merge_begin:merge_end]
    old = section(
        merge,
        "    merge_leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE\n",
        "    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):\n",
    )
    merge = merge.replace(
        old,
        "    merge_leaf_tiles = common_leaf_tiles\n    merge_extra_leaves = common_extra_leaves\n",
        1,
    )
    source = source[:merge_begin] + merge + source[merge_end:]

    forest_begin = source.index("def indexer_score_topk_forest(")
    forest_end = source.index("@pl.jit.inline(auto_scope=False)\ndef indexer_qr_rope(", forest_begin)
    forest = source[forest_begin:forest_end]
    marker = "    max_topk_cache_len = host_max_seq_len // COMPRESS_RATIO\n"
    plan = """    # 2417166f/13259f92的单写者叶计划在多task编排侧复用。
    # NativeCSACall保证lengths容量==T/6，入口max覆盖相同请求范围。
    # 每次调用重新取入口值；保留逐query的真实长度/位置/页表检查。
    common_history = pl.min(max_topk_cache_len, TOPK_MAX_CANDIDATES)
    common_max_leaves = pl.max(
        (common_history + TOPK_CANDIDATES_PER_LEAF - 1) // TOPK_CANDIDATES_PER_LEAF, 1
    )
    balanced_max_leaves = common_max_leaves
    common_leaf_tiles = TOPK_CANDIDATES_PER_LEAF // BUFFERED_LONG_SCORE_TILE
    common_extra_leaves = 0
    if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:
        common_query_count = pl.tensor.dim(position_ids, 0)
        common_groups = common_query_count // 2
        if common_query_count >= LONG_S6_MIN_QUERY_ROWS:
            common_groups = common_query_count // S
        balanced_max_leaves, common_leaf_tiles, common_extra_leaves = indexer_long_leaf_plan(
            common_history, common_groups
        )
"""
    if forest.count(marker) != 1:
        raise AssertionError("需要唯一的入口最大长度引用")
    forest = forest.replace(marker, marker + plan, 1)
    old = section(
        forest, "            max_cache_len = 0\n", "            single_leaf = pl.cast(max_leaves == 1, pl.INDEX)\n"
    )
    forest = forest.replace(old, "            max_leaves = common_max_leaves\n", 1)
    source = source[:forest_begin] + forest + source[forest_end:]
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="qa_m16_v3")
    parser.add_argument("--candidate", default="leaf_plan_v1")
    parser.add_argument("--merge-sync", action="store_true", help="对公共计划后的Top-K merge组整组准入")
    args = parser.parse_args()
    if not args.base.isidentifier() or not args.candidate.isidentifier():
        raise ValueError("私有包名必须是标识符")
    target = HERE / "sources" / args.candidate
    shutil.copytree(HERE / "sources" / args.base, target)
    path = target / "decode_indexer.py"
    source = migrate_indexer(path.read_text())
    if args.merge_sync:
        marker = 'TOPK_QUERY_WORKERS, name_hint="indexer_topk_query_merge", deps=[score_tid], allow_early_resolve=True'
        if source.count(marker) != 2:
            raise AssertionError("需要短叶/多叶两个Top-K merge入口")
        source = source.replace(marker, marker + ", sync_start=True")
    path.write_text(source)
    (target / "candidate.json").write_text(
        json.dumps(
            dict(
                candidate=args.candidate,
                base=args.base,
                donor="2417166f，冻结13259f92的indexer_group_plan_stage",
                changed_files=["decode_indexer.py"],
                merge_sync=args.merge_sync,
                scope="入口max复用；编排一次生成公共叶计划，Scalar入参传Score/Top-K；无新task/GM计划",
                contract="NativeCSACall: T=6*lengths.numel，所有cache组具有相同请求容量",
                preserved="分片策略、逐query有效范围、页检查、数值顺序、原依赖与allow_early_resolve",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(target)


if __name__ == "__main__":
    main()
