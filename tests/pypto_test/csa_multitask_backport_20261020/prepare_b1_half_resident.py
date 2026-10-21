# SPDX-License-Identifier: Apache-2.0
"""迁移a241fa61的B1原半叶分工、分数UB驻留与原序根归并。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO
from prepare_query_ready import replace_once
from prepare_query_score import append_call_arguments, append_parameters, excerpt, function


def replace_span(source, node, replacement):
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [replacement]
    return "".join(lines)


def indent(source, spaces):
    return textwrap.indent(textwrap.dedent(source).strip() + "\n", " " * spaces)


def core_update(source):
    source = append_parameters(source, "indexer_score_topk_native_cube", ["b1_half_resident: pl.constexpr"])
    source = append_call_arguments(source, {"indexer_score_topk_native_cube"}, ["False"])
    node = function(source, "indexer_score_topk_native_cube")
    core = excerpt(source, node)
    core = replace_once(
        core,
        "        if balance_leaves:\n            buf_max_leaves, buf_leaf_tiles, buf_extra_leaves",
        "        if b1_half_resident:\n"
        "            buf_max_leaves = 2 * buf_max_leaves\n"
        "        elif balance_leaves:\n            buf_max_leaves, buf_leaf_tiles, buf_extra_leaves",
    )
    # 两个AIC数学分支和旧AIV分支共用同一范围规则；实际B1只走未拆M的AIC。
    ranges = [
        n
        for n in ast.walk(ast.parse(core))
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "balance_leaves"
        and isinstance(n.body[0], ast.Assign)
        and ast.unparse(n.body[0].targets[0]) == "buf_logical_begin"
    ]
    if len(ranges) != 3:
        raise ValueError("需要核对三处原叶范围")
    for n in sorted(ranges, key=lambda n: n.lineno, reverse=True):
        old = "".join(core.splitlines(keepends=True)[n.lineno - 1 : n.end_lineno])
        prefix = " " * n.col_offset
        replacement = (
            prefix
            + "if b1_half_resident:\n"
            + prefix
            + "    buf_logical_begin, buf_leaf_capacity = indexer_b1_half_range(buf_leaf, buf_visible_count)\n"
            + old.replace(prefix + "if balance_leaves:", prefix + "elif balance_leaves:", 1)
        )
        core = replace_span(core, n, replacement)
    old_range = (
        "                    single_begin = "
        "(single_leaf * buf_leaf_tiles + pl.min(single_leaf, buf_extra_leaves)) * score_tile\n"
        "                    single_capacity = "
        "(buf_leaf_tiles + pl.cast(single_leaf < buf_extra_leaves, pl.INDEX)) * score_tile\n"
    )
    core = replace_once(
        core,
        old_range,
        old_range
        + indent(
            """
        if b1_half_resident:
            single_begin, single_capacity = indexer_b1_half_range(single_leaf, single_visible)
    """,
            20,
        ),
    )
    core = replace_once(
        core,
        "                        for single_step in pl.range(single_steps):\n",
        indent(
            """
        # 固定4×1024描述符使未采用分支也类型合法；只有B1初始化/写入它。
        legacy_half_scores = pl.tile.create([4, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32)
        if b1_half_resident:
            legacy_half_scores = pl.tile.full(
                [4, BUFFERED_LONG_SCORE_TILE], dtype=pl.FP32, value=FP32_NEG_INF
            )
    """,
            24,
        )
        + "                        for single_step in pl.range(single_steps):\n",
    )
    assemble = (
        "                                single_segments = pl.tile.assemble(\n"
        "                                    single_segments, single_scaled, [single_local * 2 + single_step % 2, 0]\n"
        "                                )\n"
    )
    core = replace_once(
        core,
        assemble,
        indent(
            """
        if b1_half_resident:
            legacy_half_scores = pl.tile.assemble(legacy_half_scores, single_scaled, [single_step, 0])
        else:
    """,
            32,
        )
        + textwrap.indent(assemble, "    "),
    )
    core = replace_once(
        core,
        "                            if single_step % 2 == 1 or single_step + 1 == single_steps:\n",
        "                            if not b1_half_resident and "
        "(single_step % 2 == 1 or single_step + 1 == single_steps):\n",
    )
    publish = [
        n
        for n in ast.walk(ast.parse(core))
        if isinstance(n, ast.For)
        and ast.unparse(n.target) == "single_local"
        and len(n.body) == 3
        and isinstance(n.body[-1], ast.Expr)
        and isinstance(n.body[-1].value, ast.Call)
        and ast.unparse(n.body[-1].value.func) == "pl.store"
    ]
    if len(publish) != 1:
        raise ValueError("原根发布循环不唯一")
    n = publish[0]
    lines = core.splitlines(keepends=True)
    prefix = "".join(lines[n.lineno - 1 : n.body[0].end_lineno])
    old = "".join(lines[n.body[1].lineno - 1 : n.end_lineno])
    block = indent(
        """
        if b1_half_resident:
            legacy_position = pl.read(position_ids, [single_query + single_query_lane])
            legacy_visible = pl.max(
                pl.min(pl.min(single_cache_len, (legacy_position + 1) // COMPRESS_RATIO), TOPK_MAX_CANDIDATES), 0
            )
            legacy_valid = pl.max(pl.min(legacy_visible - single_begin, single_capacity), 0)
            legacy_slot = (single_query + single_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf
            if legacy_valid > 0:
                legacy_half_flat = pl.tile.reshape(legacy_half_scores, [1, TOPK_CANDIDATES_PER_LEAF // 2])
                indexer_topk_half_leaf_resident(legacy_half_flat, pair_arena, single_begin, legacy_valid, legacy_slot)
            else:
                pl.store(pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),
                         [legacy_slot, 0], pair_arena)
        else:
    """,
        28,
    )
    core = replace_span(core, n, prefix + block + textwrap.indent(old, "    "))
    guard = [
        n
        for n in ast.walk(ast.parse(core))
        if isinstance(n, ast.If) and ast.unparse(n.test) == "single_begin < single_visible"
    ]
    if len(guard) != 1 or guard[0].orelse:
        raise ValueError("原有效叶分支结构变化")
    n = guard[0]
    lines = core.splitlines(keepends=True)
    lines.insert(
        n.end_lineno,
        indent(
            """
        else:
            if b1_half_resident:
                # 原最后半叶可能为空，归并仍计入该根，必须覆盖上一轮内容。
                empty_query_lane = buf_score_lane
                empty_slot = (single_query + empty_query_lane) * TOPK_ROWS_PER_QUERY + single_leaf
                pl.store(pl.tile.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF),
                         [empty_slot, 0], pair_arena)
    """,
            20,
        ),
    )
    core = "".join(lines)
    old = excerpt(source, node)
    return replace_once(source, old, core)


def add_helpers(source, donor):
    root = excerpt(source, function(source, "indexer_topk_query_merge_one"))
    root = root.replace("def indexer_topk_query_merge_one(", "def indexer_topk_query_merge_b1(", 1)
    start = root.index("            for child in pl.range(1, half_count):\n")
    stop = root.index("        root_scores =", start)
    publish = function(donor, "indexer_pair_publish_stage")
    branch = next(
        n for n in ast.walk(publish) if isinstance(n, ast.If) and ast.unparse(n.test) == "BALANCED and QUERY_GROUP == 2"
    )
    donor_lines = donor.splitlines(keepends=True)
    payload = "".join(donor_lines[branch.body[0].lineno - 1 : branch.body[-1].end_lineno])
    payload = payload.replace("root_count", "half_count")
    root = root[:start] + indent(payload, 12) + root[stop:]
    half = '''
@pl.jit.inline(auto_scope=False)
def indexer_b1_half_range(leaf: pl.Scalar[pl.INDEX], visible: pl.Scalar[pl.INDEX]):
    """a241fa61的原半叶几何，末叶容量由实际可见数决定，不能固定切4096。"""
    original_begin = (leaf // 2) * TOPK_CANDIDATES_PER_LEAF
    original_valid = pl.max(pl.min(visible - original_begin, TOPK_CANDIDATES_PER_LEAF), 0)
    original_tiles = (original_valid + BUFFERED_LONG_SCORE_TILE - 1) // BUFFERED_LONG_SCORE_TILE
    capacity = pl.min(original_tiles * (BUFFERED_LONG_SCORE_TILE // 2), TOPK_CANDIDATES_PER_LEAF // 2)
    begin = original_begin + (leaf % 2) * capacity
    return begin, capacity
'''
    wrapper = """
@pl.jit.incore
def indexer_topk_query_merge_b1_incore(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
):
    worker = pl.tile.get_block_idx()
    query_count = pl.tensor.dim(position_ids, 0)
    for query in pl.range(worker, query_count, TOPK_QUERY_WORKERS):
        indexer_topk_query_merge_b1(
            query, position_ids, kv_seq_lens, pair_arena, topk_scores, topk_indices, False, 8, 0,
        )
"""
    point = source.index("@pl.jit.inline(auto_scope=False)\ndef indexer_score_topk_native_cube")
    return source[:point] + half + "\n\n" + root + "\n\n" + wrapper + "\n\n" + source[point:]


def dispatch(source, flat_predicate=False, inline_merge=False):
    node = function(source, "indexer_score_topk_forest")
    forest = excerpt(source, node)
    forest = replace_once(
        forest,
        "    max_topk_cache_len = host_max_seq_len // COMPRESS_RATIO\n",
        "    max_topk_cache_len = host_max_seq_len // COMPRESS_RATIO\n"
        + indent(
            """
        use_b1_half = pl.cast(0, pl.INT32)
        if pl.tensor.dim(position_ids, 0) == S:
            if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF and max_topk_cache_len <= TOPK_MAX_CANDIDATES:
                use_b1_half = pl.cast(1, pl.INT32)
    """,
            4,
        ),
    )
    if not flat_predicate:
        forest = replace_once(
            forest,
            "        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF and max_topk_cache_len <= TOPK_MAX_CANDIDATES:\n"
            "            use_b1_half = pl.cast(1, pl.INT32)\n",
            "        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:\n"
            "            if max_topk_cache_len <= TOPK_MAX_CANDIDATES:\n"
            "                use_b1_half = pl.cast(1, pl.INT32)\n",
        )
    calls = [
        n
        for n in ast.walk(ast.parse(forest))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "indexer_score_topk_native_cube"
        and ast.unparse(n.args[12]) == "2"
        and ast.unparse(n.args[14]) == "BUFFERED_LONG_SCORE_TILE"
    ]
    if len(calls) != 1:
        raise ValueError("需要唯一原长G2调用")
    call = calls[0]
    call.args[19] = ast.Constant(value=True)
    call.args[-1] = ast.Constant(value=True)
    new_call = "score_tid = " + ast.unparse(call) + "\n"
    forest = replace_once(
        forest,
        "        if max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:\n            # 长档整组准入",
        "        if use_b1_half == 1:\n"
        + indent(new_call, 12)
        + "        elif max_topk_cache_len > TOPK_CANDIDATES_PER_LEAF:\n            # 长档整组准入",
    )
    merge = "        if max_topk_cache_len < INDEXER_NATIVE_CUBE_MIN_ROWS:\n"
    replacement = indent(
        """
        if use_b1_half == 1:
            with pl.spmd(TOPK_QUERY_WORKERS, name_hint="indexer_topk_query_merge_b1",
                         deps=[score_tid], allow_early_resolve=True):
                for query in pl.range(pl.tile.get_block_idx(), pl.tensor.dim(position_ids, 0), TOPK_QUERY_WORKERS):
                    indexer_topk_query_merge_b1(position_ids=position_ids, kv_seq_lens=kv_seq_lens,
                        query=query, pair_arena=pair_arena, topk_scores=topk_scores, topk_indices=topk_idxs,
                        multiway=False, leaf_tiles=8, extra_leaves=0)
        elif max_topk_cache_len < INDEXER_NATIVE_CUBE_MIN_ROWS:
    """,
        8,
    )
    forest = replace_once(forest, merge, replacement)
    if not inline_merge:
        direct = indent(
            """
            for query in pl.range(pl.tile.get_block_idx(), pl.tensor.dim(position_ids, 0), TOPK_QUERY_WORKERS):
                indexer_topk_query_merge_b1(position_ids=position_ids, kv_seq_lens=kv_seq_lens,
                    query=query, pair_arena=pair_arena, topk_scores=topk_scores, topk_indices=topk_idxs,
                    multiway=False, leaf_tiles=8, extra_leaves=0)
        """,
            16,
        )
        forest = replace_once(
            forest,
            direct,
            indent(
                """
            indexer_topk_query_merge_b1_incore(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs)
        """,
                16,
            ),
        )
    return replace_once(source, excerpt(source, node), forest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="b1_half_resident_v4")
    parser.add_argument("--flat-predicate", action="store_true", help="仅复现v1编排不支持And表达式的失败")
    parser.add_argument("--keep-output-alias", action="store_true", help="仅复现v2的跨scope返回别名失败")
    parser.add_argument("--inline-merge", action="store_true", help="仅复现v1至v3的编排直接内联归并")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须为标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))

    def donor(name):
        return subprocess.check_output(
            ["git", "show", f"a241fa61:vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/{name}.py"], cwd=REPO, text=True
        )

    stage_source = donor("indexer_topk_stages")
    license_prefix = "".join(stage_source.splitlines(keepends=True)[: ast.parse(stage_source).body[0].lineno - 1])
    stage = (
        license_prefix + '"""a241fa61原半叶排序数学，迁移到多task的B1专用路径。"""\n'
        "import pypto.language as pl\nfrom .config import FP32_NEG_INF\n"
        'PAIR_ROWS_DYN = pl.dynamic("CSA_B1_PAIR_ROWS")\nTOPK_PAIR_WIDTH = 1024\n\n'
        + excerpt(stage_source, function(stage_source, "indexer_topk_half_leaf_resident"))
        + "\n"
    )
    assert ast.dump(function(stage, "indexer_topk_half_leaf_resident")) == ast.dump(
        function(stage_source, "indexer_topk_half_leaf_resident")
    )
    source = (base / "decode_indexer.py").read_text()
    source = core_update(source)
    source = add_helpers(source, donor("indexer_publish_stages"))
    source = dispatch(source, args.flat_predicate, args.inline_merge)
    source = replace_once(
        source,
        "from .topk_threshold_stages import threshold_merge_top512\n",
        "from .topk_threshold_stages import threshold_merge_top512\n"
        "from .b1_topk_stages import indexer_topk_half_leaf_resident\n",
    )
    ast.parse(source)
    entry = (base / "decode_csa.py").read_text()
    if args.inline_merge and not args.keep_output_alias:
        entry = replace_once(entry, "_index_scores, index_indices =", "_index_scores, _index_indices =")
        entry = replace_once(entry, "                index_indices,\n", "                idx_topk,\n")
    ast.parse(entry)
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(source)
    (target / "b1_topk_stages.py").write_text(stage)
    (target / "decode_csa.py").write_text(entry)
    manifest = dict(
        candidate=args.candidate,
        base=args.base,
        donor="a241fa61",
        changed_files=["decode_indexer.py", "b1_topk_stages.py"]
        + (["decode_csa.py"] if args.inline_merge and not args.keep_output_alias else []),
        mechanism="B1用G2独立调度原半叶，分数保留UB并按原半叶六分支排序，根按incoming优先左折叠保留UB",
        scope="当前多task固定S6合同，实际T==S且8192<compressed_history<=65536；保留现有动态页表/长度/位置处理",
        source_proof="已提交a241fa61及B1_HALF_SCORE_ADOPTION；不沿用另一个session的10%验收门槛",
        caveats=["不同于当前正式B1的G6/24叶，必须重新配对", "原SPMD的动态Q/L接口不直接等价于当前多task固定S6"],
        context_switch_history=32765,
        repro_only_flat_predicate=args.flat_predicate,
        repro_only_output_alias=args.keep_output_alias,
        repro_only_inline_merge=args.inline_merge,
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
