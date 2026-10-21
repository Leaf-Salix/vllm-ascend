# SPDX-License-Identifier: Apache-2.0
"""按Score资源区间联动已验证WS列池和成对压缩，不改变短档策略。"""

import argparse
import ast
import json
import shutil
import subprocess
import textwrap

from prepare import HERE, REPO
from prepare_query_ready import replace_once
from prepare_ws_pool import DONOR_COMMIT, DONOR_FILE, assignment_block, excerpt


def migrate(source, donor):
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube")
    pooled = next(
        n
        for n in ast.parse(donor).body
        if isinstance(n, ast.FunctionDef) and n.name == "_indexer_score_cube_split_store"
    )
    pool_create = [
        n
        for n in ast.walk(pooled)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "split_ws_pool" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "pl.tile.create"
    ]
    if len(pool_create) != 1:
        raise ValueError("来源WS列池创建必须唯一")
    edits = [
        (assignment_block(kernel, "split_ws_acc", 1), pool_create),
        (assignment_block(kernel, "split_ws_first", 3), assignment_block(pooled, "split_ws_col", 6)),
        (assignment_block(kernel, "split_ws_first_last", 3), assignment_block(pooled, "split_ws_last_col", 6)),
    ]
    lines = source.splitlines(keepends=True)
    for old, new in sorted(edits, key=lambda pair: pair[0][0].lineno, reverse=True):
        original = excerpt(source, old)
        replacement = excerpt(donor, new).replace("SPLIT_WS_POOL_PANELS", "(score_tile // (2 * qk_panel_cols))")
        if len(old) == 1:
            # SSA验证早于constexpr折叠。创建需位于公共作用域；折叠后未使用
            # 的Acc声明应被消除，随后检查实际生成代码，不假定必然零成本。
            conditional = original + replacement
        else:
            conditional = (
                "if use_ws_pool:\n"
                + textwrap.indent(replacement, "    ")
                + "else:\n"
                + textwrap.indent(original, "    ")
            )
        lines[old[0].lineno - 1 : old[-1].end_lineno] = [textwrap.indent(conditional, " " * old[0].col_offset)]
    source = "".join(lines)
    tree = ast.parse(source)
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_native_cube")
    lines = source.splitlines(keepends=True)
    header_end = next(i for i in range(kernel.lineno, kernel.end_lineno) if lines[i] == "):\n")
    lines.insert(header_end, "    use_ws_pool: pl.constexpr,\n    use_paired_prefix: pl.constexpr,\n")
    source = "".join(lines)
    source = replace_once(
        source,
        "from .topk_threshold_stages import threshold_merge_top512\n",
        "from .topk_compact_stages import threshold_merge_paired_top512\n"
        "from .topk_threshold_stages import threshold_merge_top512\n",
    )
    merges = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "threshold_merge_top512"
    ]
    if len(merges) != 1:
        raise ValueError("原阈值归并必须唯一")
    node = merges[0]
    lines = source.splitlines(keepends=True)
    old = textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))
    new = old.replace("threshold_merge_top512(", "threshold_merge_paired_top512(", 1)
    body = "if use_paired_prefix:\n" + textwrap.indent(new, "    ") + "else:\n" + textwrap.indent(old, "    ")
    lines[node.lineno - 1 : node.end_lineno] = [textwrap.indent(body, " " * node.col_offset)]
    source = "".join(lines)
    tree = ast.parse(source)
    forest = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "indexer_score_topk_forest")
    split = [
        n
        for n in ast.walk(forest)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "pl.tensor.dim(position_ids, 0) >= SCORE_SPLIT_MIN_QUERY_ROWS"
    ]
    if len(split) != 1 or len(split[0].body) != 1:
        raise ValueError("只修改既有长G6拆M路径的单个调用")
    call = split[0].body[0]
    lines = source.splitlines(keepends=True)
    original = textwrap.dedent("".join(lines[call.lineno - 1 : call.end_lineno]))

    def variant(pool):
        return replace_once(
            original,
            "    layout_tid,\n)",
            f"    layout_tid,\n    use_ws_pool={pool},\n    use_paired_prefix=True,\n)",
        )

    policy = (
        "# 仅扩展既有拆M长G6：按请求波数和入口长度上界选择等价实现。\n"
        "# 原短档、G2、未拆M小批及两波调用显式关闭新constexpr，不改变策略。\n"
        "region_groups = pl.tensor.dim(position_ids, 0) // S\n"
        "region_leaves, region_tiles, region_extra = indexer_long_leaf_plan(\n"
        "    pl.min(max_topk_cache_len, TOPK_MAX_CANDIDATES), region_groups\n"
        ")\n"
        "if region_groups * region_leaves > 2 * TOPK_SCORE_WORKERS:\n"
        "    if region_groups <= TOPK_SCORE_WORKERS:\n"
        + textwrap.indent(variant(True), "        ")
        + "    else:\n"
        + textwrap.indent(variant(False), "        ")
        + "else:\n"
        + textwrap.indent(original, "    ")
    )
    lines[call.lineno - 1 : call.end_lineno] = [textwrap.indent(policy, " " * call.col_offset)]
    result = "".join(lines)
    # 依赖调用不读取constexpr签名默认值，必须在每个原调用显式绑定False。
    calls = [
        n
        for n in ast.walk(ast.parse(result))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "indexer_score_topk_native_cube"
        and not n.keywords
    ]
    if len(calls) != 6:
        raise ValueError("必须保留六个原调用并显式关闭新能力")
    lines = result.splitlines(keepends=True)
    for node in sorted(calls, key=lambda n: n.end_lineno, reverse=True):
        indent = len(lines[node.end_lineno - 1]) - len(lines[node.end_lineno - 1].lstrip()) + 4
        lines.insert(
            node.end_lineno - 1,
            " " * indent + "use_ws_pool=False,\n" + " " * indent + "use_paired_prefix=False,\n",
        )
    result = "".join(lines)
    ast.parse(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="formal_4d5b2ed9")
    parser.add_argument("--candidate", default="score_region_v3")
    args = parser.parse_args()
    if not all(n.isidentifier() for n in (args.base, args.candidate)):
        raise ValueError("私有包名必须是标识符")
    base, target = (HERE / "sources" / n for n in (args.base, args.candidate))
    donor = subprocess.check_output(["git", "show", f"{DONOR_COMMIT}:{DONOR_FILE}"], cwd=REPO, text=True)
    changed = migrate((base / "decode_indexer.py").read_text(), donor)
    helper = HERE / "sources/paired_wave_policy_v1/topk_compact_stages.py"
    shutil.copytree(base, target)
    (target / "decode_indexer.py").write_text(changed)
    (target / helper.name).write_text(helper.read_text())
    manifest = dict(
        candidate=args.candidate,
        base=args.base,
        sources={"ws_pool": DONOR_COMMIT, "paired_compact": "c0d27e9a"},
        changed_files=["decode_indexer.py", "topk_compact_stages.py"],
        policy="长G6拆M且Score工作项>48启用成对压缩；请求组<=24另启用WS列池",
        rationale="单请求波区间B16已有WS+短前缀五窗Score收益；跨请求波B28不叠加列池",
        caveat="按资源边界的待测组合，不声称所有同区间形状必然受益；实际长度与页表处理不变",
        unchanged="短档/G2/未拆M小批仅显式补False constexpr；原策略、数学、同分规则、依赖和硬件握手不改",
        context_switch_history=32765,
        status="private_candidate_no_device_evidence",
    )
    (target / "candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
